"""
Flight Maneuver Classifier — ルールベース仮ラベラー (14クラス / G-Load対応版)

スライディングウィンドウ特徴量から、航空機の機動を14クラスに分類する
ルールベースのラベラー。G-Load を主要な判定基準として組み込む。

クラス体系:
    基本飛行 (5):
        0: Straight & Level  — 直線水平飛行 (g ≈ 1)
        1: Acceleration       — 加速 (水平, g ≈ 1)
        2: Deceleration       — 減速 (水平, g ≈ 1)
        3: Steady Climb       — 定常上昇
        4: Steady Descent     — 定常降下
    旋回系 (4):
        5: Level Turn         — 水平旋回 (g ≈ 1.5–3)
        6: Climbing Turn      — 上昇旋回 / High Yo-Yo
        7: Descending Turn    — 降下旋回 / Low Yo-Yo
        8: High-G Turn        — 高G旋回 / ブレイクターン (g > 4)
    戦術機動 (5):
        9: Dive               — 急降下 (g < 1, unloaded)
       10: Zoom Climb         — 急上昇 / ズームクライム (g > 2)
       11: Reversal           — 方向転換 (Split-S / Immelmann)
       12: Jinking            — 回避機動 (g_load_std 大)
       13: Extension          — 離脱 / エネルギー回復 (g ≈ 1)
"""

import numpy as np
import pandas as pd


# ============================================================
# クラス定義
# ============================================================

MANEUVER_CLASSES = {
    # 基本飛行
    0:  "Straight & Level",
    1:  "Acceleration",
    2:  "Deceleration",
    3:  "Steady Climb",
    4:  "Steady Descent",
    # 旋回系
    5:  "Level Turn",
    6:  "Climbing Turn",
    7:  "Descending Turn",
    8:  "High-G Turn",
    # 戦術機動
    9:  "Dive",
    10: "Zoom Climb",
    11: "Reversal",
    12: "Jinking",
    13: "Extension",
}

CLASS_NAMES = list(MANEUVER_CLASSES.values())
NUM_CLASSES = len(MANEUVER_CLASSES)


# ============================================================
# デフォルト閾値
# ============================================================

DEFAULT_THRESHOLDS = {
    # --- 旋回判定 ---
    "heading_delta_threshold": 5.0,         # [deg]   これ以上で「旋回」
    "heading_reversal_threshold": 120.0,    # [deg]   これ以上で「方向転換」

    # --- 高度変化判定 ---
    "altitude_slope_threshold": 2.0,        # [m/s]   上昇/降下判定
    "altitude_slope_steep": 10.0,           # [m/s]   急降下/急上昇判定

    # --- 速度変化判定 ---
    "speed_delta_threshold": 5.0,           # [m/s]   加速/減速判定

    # --- G-Load 判定 ---
    "g_load_high_g": 7.0,                   # [G]     これ以上で High-G Turn
    "g_load_unloaded": 0.5,                 # [G]     これ以下で unloaded (Dive)
    "g_load_zoom": 2.0,                     # [G]     これ以上で Zoom Climb の pull
    "g_load_std_jinking": 1.0,              # [G]     g_load_std がこれ以上 → Jinking
    "g_load_level": 1.5,                    # [G]     g_mean がこれ以下 → level 相当

    # --- ロール / ジンキング ---
    "roll_std_jinking": 50.0,              # [deg]   roll_std がこれ以上でジンキング補助

    # --- Dive / Zoom ---
    "pitch_dive_threshold": -15.0,          # [deg]   pitch がこれ以下 → Dive
    "pitch_zoom_threshold": 15.0,           # [deg]   pitch がこれ以上 → Zoom Climb

    # --- Extension ---
    "extension_speed_gain": 5.0,            # [m/s]   speed 増加
}


# ============================================================
# ラベラー
# ============================================================

class ManeuverLabeler:
    """
    ルールベースで14クラスの機動ラベルを付与する (G-Load 対応版)。

    G-Load の活用方針:
        - g_load_mean: 旋回荷重の強度（High-G Turn ≥ 4G, Level Turn ≈ 1.5–3G）
        - g_load_std:  荷重の変動（大 → Jinking / 不規則な機動）
        - g_load_mean < 1:  unloaded → Dive / pushover
        - g_load_mean ≈ 1:  直線飛行系（Level, Accel, Decel, Extension）

    判定の優先度 (高→低):
        1. Jinking         — g_load_std 大 & (roll_std 大 or heading_std 大)
        2. Reversal        — |heading_delta| ≥ 120°
        3. High-G Turn     — g_load_mean ≥ 4G & heading 変化
        4. Dive            — pitch 大きく負 & g < 1 (unloaded) & altitude 急降下
        5. Zoom Climb      — pitch 大きく正 & g ≥ 2 & altitude 急上昇
        6. Climbing Turn   — heading 変化 & altitude 上昇
        7. Descending Turn — heading 変化 & altitude 降下
        8. Level Turn      — heading 変化 & altitude 概ねフラット
        9. Steady Climb    — altitude 上昇 (旋回なし)
       10. Steady Descent  — altitude 降下 (旋回なし)
       11. Extension       — speed 増加 & 直線 & 微降下 & g ≈ 1
       12. Acceleration    — speed 増加
       13. Deceleration    — speed 減少
       14. Straight & Level — デフォルト (g ≈ 1)
    """

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = {**DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

    def label_single(self, row: dict | pd.Series) -> int:
        """
        1窓分の特徴量からクラスID (0–13) を返す。

        使用する特徴量:
            heading_delta, heading_std, altitude_slope,
            speed_delta, roll_std, pitch_mean,
            g_load_mean, g_load_std
        """
        heading_delta = row.get("heading_delta", 0.0)
        heading_std = row.get("heading_std", 0.0)
        altitude_slope = row.get("altitude_slope", 0.0)
        speed_delta = row.get("speed_delta", 0.0)
        roll_std = row.get("roll_std", 0.0)
        pitch_mean = row.get("pitch_mean", 0.0)
        g_mean = row.get("g_load_mean", 1.0)
        g_std = row.get("g_load_std", 0.0)

        abs_heading_delta = abs(heading_delta)
        th = self.thresholds

        # ---- 1. Jinking: G荷重が不規則に変動 + 姿勢も不安定 ----
        if (g_std > th["g_load_std_jinking"]
                and (roll_std > th["roll_std_jinking"]
                     or heading_std > th["heading_delta_threshold"] * 2)):
            return 12  # Jinking

        # ---- 2. Reversal: 大きな方向転換 ----
        if abs_heading_delta >= th["heading_reversal_threshold"]:
            return 11  # Reversal

        # ---- 3. High-G Turn: 高荷重 + heading 変化 ----
        if (g_mean >= th["g_load_high_g"]
                and abs_heading_delta > th["heading_delta_threshold"]):
            return 8  # High-G Turn

        # ---- 4. Dive: unloaded, pitch 負, altitude 急降下 ----
        if (g_mean < th["g_load_unloaded"]
                and pitch_mean < th["pitch_dive_threshold"]
                and altitude_slope < -th["altitude_slope_steep"]):
            return 9  # Dive

        # ---- 5. Zoom Climb: 高G pull, pitch 正, altitude 急上昇 ----
        if (g_mean >= th["g_load_zoom"]
                and pitch_mean > th["pitch_zoom_threshold"]
                and altitude_slope > th["altitude_slope_steep"]):
            return 10  # Zoom Climb

        # ---- 旋回系 (heading_delta が閾値以上) ----
        is_turning = abs_heading_delta > th["heading_delta_threshold"]

        if is_turning:
            # 6. Climbing Turn: 旋回 + 上昇
            if altitude_slope > th["altitude_slope_threshold"]:
                return 6  # Climbing Turn

            # 7. Descending Turn: 旋回 + 降下
            if altitude_slope < -th["altitude_slope_threshold"]:
                return 7  # Descending Turn

            # 8. Level Turn: 旋回のみ
            return 5  # Level Turn

        # ---- 非旋回・縦方向 ----
        # 9. Steady Climb
        if altitude_slope > th["altitude_slope_threshold"]:
            return 3  # Steady Climb

        # 10. Steady Descent
        if altitude_slope < -th["altitude_slope_threshold"]:
            return 4  # Steady Descent

        # ---- 非旋回・水平・速度変化 ----
        # 11. Extension: 速度増加 + 直線 + 微降下 + 低G (離脱)
        if (speed_delta > th["extension_speed_gain"]
                and abs_heading_delta <= th["heading_delta_threshold"]
                and altitude_slope < 0
                and altitude_slope >= -th["altitude_slope_threshold"]
                and g_mean < th["g_load_level"]):
            return 13  # Extension

        # 12. Acceleration: 速度増加
        if speed_delta > th["speed_delta_threshold"]:
            return 1  # Acceleration

        # 13. Deceleration: 速度減少
        if speed_delta < -th["speed_delta_threshold"]:
            return 2  # Deceleration

        # ---- 14. Straight & Level (デフォルト) ----
        return 0

    def label_dataframe(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量 DataFrame 全体にラベルを付与する。

        Returns:
            元の DataFrame に 'label' (int) と 'label_name' (str) 列を追加した
            DataFrame のコピー。
        """
        if features_df.empty:
            df = features_df.copy()
            df["label"] = pd.Series(dtype=int)
            df["label_name"] = pd.Series(dtype=str)
            return df

        df = features_df.copy()
        labels = df.apply(lambda row: self.label_single(row), axis=1)
        df["label"] = labels.astype(int)
        df["label_name"] = df["label"].map(MANEUVER_CLASSES)
        return df

    def print_distribution(self, df: pd.DataFrame):
        """ラベル分布を表示する。"""
        if "label" not in df.columns:
            print("[ManeuverLabeler] ラベル列がありません")
            return

        total = len(df)
        print(f"\n[ManeuverLabeler] === ラベル分布 (N={total}) ===")
        for cls_id, cls_name in MANEUVER_CLASSES.items():
            count = (df["label"] == cls_id).sum()
            pct = count / total * 100 if total > 0 else 0
            print(f"  {cls_id:2d}: {cls_name:20s}  {count:6d}  ({pct:5.1f}%)")
