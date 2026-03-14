"""
Flight Maneuver Classifier — ルールベース仮ラベラー

スライディングウィンドウ特徴量から、航空機の機動を6クラスに分類する
ルールベースのラベラー。

クラス:
    0: Level Flight  — 水平飛行
    1: Climb          — 上昇
    2: Descent        — 降下
    3: Left Turn      — 左旋回
    4: Right Turn     — 右旋回
    5: Roll Maneuver  — ロール機動
"""

import numpy as np
import pandas as pd


# ============================================================
# クラス定義
# ============================================================

MANEUVER_CLASSES = {
    0: "Level Flight",
    1: "Climb",
    2: "Descent",
    3: "Left Turn",
    4: "Right Turn",
    5: "Roll Maneuver",
}

CLASS_NAMES = list(MANEUVER_CLASSES.values())
NUM_CLASSES = len(MANEUVER_CLASSES)


# ============================================================
# デフォルト閾値
# ============================================================

DEFAULT_THRESHOLDS = {
    # Roll Maneuver: roll_std が大きい or roll_delta が大きい
    "roll_std_threshold": 15.0,       # deg
    "roll_delta_threshold": 30.0,     # deg

    # Turn: heading_delta の閾値
    "heading_delta_threshold": 5.0,   # deg

    # Climb/Descent: altitude_slope の閾値
    "altitude_slope_threshold": 2.0,  # m/s
}


# ============================================================
# ラベラー
# ============================================================

class ManeuverLabeler:
    """
    ルールベースで機動クラスの仮ラベルを付与する。

    優先度 (高→低):
        1. Roll Maneuver — roll_std > 閾値 or |roll_delta| > 閾値
        2. Left Turn     — heading_delta < -閾値 (left = 負方向)
        3. Right Turn    — heading_delta > +閾値
        4. Climb         — altitude_slope > 閾値 かつ |heading_delta| ≤ 閾値
        5. Descent       — altitude_slope < -閾値 かつ |heading_delta| ≤ 閾値
        6. Level Flight  — デフォルト
    """

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = {**DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

    def label_single(self, row: dict | pd.Series) -> int:
        """
        1窓分の特徴量からクラスIDを返す。

        Args:
            row: 特徴量辞書 / Series (roll_std, roll_delta, heading_delta,
                 altitude_slope 等を含む)

        Returns:
            クラスID (0–5)
        """
        roll_std = abs(row.get("roll_std", 0.0))
        roll_delta = abs(row.get("roll_delta", 0.0))
        heading_delta = row.get("heading_delta", 0.0)
        altitude_slope = row.get("altitude_slope", 0.0)

        th = self.thresholds

        # 1. Roll Maneuver
        if roll_std > th["roll_std_threshold"] or roll_delta > th["roll_delta_threshold"]:
            return 5  # Roll Maneuver

        # 2. Left Turn (heading が負方向に変化 = 左旋回)
        if heading_delta < -th["heading_delta_threshold"]:
            return 3  # Left Turn

        # 3. Right Turn
        if heading_delta > th["heading_delta_threshold"]:
            return 4  # Right Turn

        # 4. Climb (旋回でない場合のみ)
        if altitude_slope > th["altitude_slope_threshold"]:
            return 1  # Climb

        # 5. Descent (旋回でない場合のみ)
        if altitude_slope < -th["altitude_slope_threshold"]:
            return 2  # Descent

        # 6. Level Flight (デフォルト)
        return 0

    def label_dataframe(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        特徴量 DataFrame 全体にラベルを付与する。

        Args:
            features_df: extract_window_features() の出力

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
            print(f"  {cls_id}: {cls_name:20s}  {count:6d}  ({pct:5.1f}%)")
