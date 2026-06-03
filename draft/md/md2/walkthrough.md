# Walkthrough — Flight Maneuver Classifier

## 概要

Tacview ACMI ログから航空機機動を分類するパイプライン。Phase 1 で6クラス版を構築し、Phase 2 で14クラスの戦術機動体系に拡張。

## 14クラス体系

| ID | クラス | カテゴリ | 検出キー |
|----|--------|---------|---------|
| 0 | Straight & Level | 基本 | 全変化小 |
| 1 | Acceleration | 基本 | speed↑, level |
| 2 | Deceleration | 基本 | speed↓, level |
| 3 | Steady Climb | 基本 | alt↑, 直線 |
| 4 | Steady Descent | 基本 | alt↓, 直線 |
| 5 | Level Turn | 旋回 | heading変化, alt≈0 |
| 6 | Climbing Turn | 旋回 | heading変化 + alt↑ |
| 7 | Descending Turn | 旋回 | heading変化 + alt↓ |
| 8 | High-G Turn | 旋回 | heading変化 + speed大幅↓ |
| 9 | Dive | 戦術 | pitch<<0 + alt急↓ + speed↑ |
| 10 | Zoom Climb | 戦術 | pitch>>0 + alt急↑ + speed↓ |
| 11 | Reversal | 戦術 | \|Δheading\|≥120° |
| 12 | Jinking | 戦術 | roll_std大 + heading_std大 |
| 13 | Extension | 戦術 | speed↑ + 直線 + 微降下 |

## 変更ファイル

```diff:maneuver_labeler.py
===
"""
Flight Maneuver Classifier — ルールベース仮ラベラー (14クラス版)

スライディングウィンドウ特徴量から、航空機の機動を14クラスに分類する
ルールベースのラベラー。

クラス体系:
    基本飛行 (5):
        0: Straight & Level  — 直線水平飛行
        1: Acceleration       — 加速 (水平)
        2: Deceleration       — 減速 (水平)
        3: Steady Climb       — 定常上昇
        4: Steady Descent     — 定常降下
    旋回系 (4):
        5: Level Turn         — 水平旋回
        6: Climbing Turn      — 上昇旋回 (High Yo-Yo 的)
        7: Descending Turn    — 降下旋回 (Low Yo-Yo 的)
        8: High-G Turn        — 高G旋回 / ブレイクターン
    戦術機動 (5):
        9: Dive               — 急降下
       10: Zoom Climb         — 急上昇 (ズームクライム)
       11: Reversal           — 方向転換 (Split-S / Immelmann)
       12: Jinking            — 回避機動 (不規則な変化)
       13: Extension          — 離脱 / エネルギー回復
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
    "heading_delta_threshold": 5.0,         # deg: これ以上で「旋回」
    "heading_reversal_threshold": 120.0,    # deg: これ以上で「方向転換」

    # --- 高度変化判定 ---
    "altitude_slope_threshold": 2.0,        # m/s: 上昇/降下判定
    "altitude_slope_steep": 10.0,           # m/s: 急降下/急上昇判定

    # --- 速度変化判定 ---
    "speed_delta_threshold": 5.0,           # m/s: 加速/減速判定
    "speed_loss_highg": 15.0,               # m/s: 高G旋回での速度損失

    # --- ロール / ジンキング ---
    "roll_std_jinking": 20.0,               # deg: roll_std がこれ以上 → ジンキング
    "heading_std_jinking": 10.0,            # deg: heading_std がこれ以上 → ジンキング

    # --- Dive / Zoom ---
    "pitch_dive_threshold": -15.0,          # deg: pitch がこれ以下 → Dive
    "pitch_zoom_threshold": 15.0,           # deg: pitch がこれ以上 → Zoom Climb

    # --- Extension ---
    "extension_speed_gain": 5.0,            # m/s: speed 増加
}


# ============================================================
# ラベラー
# ============================================================

class ManeuverLabeler:
    """
    ルールベースで14クラスの機動ラベルを付与する。

    判定の優先度 (高→低):
        1. Jinking         — roll_std & heading_std が大きい (不規則な回避機動)
        2. Reversal        — |heading_delta| ≥ 120° (方向転換)
        3. Dive            — pitch 大きく負 & altitude 急降下 & speed 増加
        4. Zoom Climb      — pitch 大きく正 & altitude 急上昇 & speed 減少
        5. High-G Turn     — heading 変化 大 & speed 大きく減少
        6. Climbing Turn   — heading 変化 & altitude 上昇
        7. Descending Turn — heading 変化 & altitude 降下
        8. Level Turn      — heading 変化 & altitude 概ねフラット
        9. Steady Climb    — altitude 上昇 (旋回なし)
       10. Steady Descent  — altitude 降下 (旋回なし)
       11. Extension       — speed 増加 & 直線 & altitude フラット～微降下
       12. Acceleration    — speed 増加 & 直線 & altitude フラット
       13. Deceleration    — speed 減少 & 直線 & altitude フラット
       14. Straight & Level — デフォルト
    """

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = {**DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

    def label_single(self, row: dict | pd.Series) -> int:
        """
        1窓分の特徴量からクラスID (0–13) を返す。

        使用する特徴量:
            heading_delta, heading_std, altitude_slope, altitude_delta,
            speed_delta, speed_slope, roll_std, pitch_mean
        """
        heading_delta = row.get("heading_delta", 0.0)
        heading_std = row.get("heading_std", 0.0)
        altitude_slope = row.get("altitude_slope", 0.0)
        speed_delta = row.get("speed_delta", 0.0)
        roll_std = row.get("roll_std", 0.0)
        pitch_mean = row.get("pitch_mean", 0.0)

        abs_heading_delta = abs(heading_delta)
        th = self.thresholds

        # ---- 1. Jinking: 非常に不規則な動き ----
        if (roll_std > th["roll_std_jinking"]
                and heading_std > th["heading_std_jinking"]):
            return 12  # Jinking

        # ---- 2. Reversal: 大きな方向転換 ----
        if abs_heading_delta >= th["heading_reversal_threshold"]:
            return 11  # Reversal

        # ---- 3. Dive: 急降下 ----
        if (pitch_mean < th["pitch_dive_threshold"]
                and altitude_slope < -th["altitude_slope_steep"]
                and speed_delta > 0):
            return 9  # Dive

        # ---- 4. Zoom Climb: 急上昇 ----
        if (pitch_mean > th["pitch_zoom_threshold"]
                and altitude_slope > th["altitude_slope_steep"]
                and speed_delta < 0):
            return 10  # Zoom Climb

        # ---- 旋回系 (heading_delta が閾値以上) ----
        is_turning = abs_heading_delta > th["heading_delta_threshold"]

        if is_turning:
            # 5. High-G Turn: 旋回 + 大きな速度損失
            if speed_delta < -th["speed_loss_highg"]:
                return 8  # High-G Turn

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
        # 11. Extension: 速度増加 + 直線 + 微降下 (離脱/エネルギー回復)
        #     altitude_slope が 0 未満 (微降下) だが急降下ではない場合
        if (speed_delta > th["extension_speed_gain"]
                and abs_heading_delta <= th["heading_delta_threshold"]
                and altitude_slope < 0
                and altitude_slope >= -th["altitude_slope_threshold"]):
            return 13  # Extension

        # 12. Acceleration: 速度増加 + 直線 + 水平〜微上昇
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
```

```diff:maneuver_main.py
===
"""
Flight Maneuver Classifier — メインエントリポイント

Tacview ACMI ログから航空機機動を分類するパイプラインを実行する。

Usage:
    python src/maneuver_main.py flight.acmi
    python src/maneuver_main.py flight.acmi --window 5 --step 1 --output-dir results/
    python src/maneuver_main.py flight.acmi --list-aircraft   # 航空機一覧のみ表示

処理フロー:
    1. ACMI パース (acmi_parser.ACMIParser)
    2. 航空機ごとの時系列 DataFrame 構築
    3. 前処理 (角度 unwrap, 欠損補間)
    4. スライディングウィンドウ特徴量抽出
    5. ルールベース仮ラベル付与
    6. Random Forest 学習・評価
    7. 結果出力 (コンソール + 画像 + CSV)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# プロジェクト src をパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

from acmi_parser import ACMIParser
from maneuver_feature_engine import (
    FEATURE_COLUMNS,
    build_feature_matrix,
    get_all_aircraft_ids,
)
from maneuver_labeler import ManeuverLabeler, MANEUVER_CLASSES
from maneuver_classifier import train_and_evaluate, save_model


def run_pipeline(
    acmi_path: str,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    output_dir: str = "results",
    aircraft_filter: str | None = None,
    thresholds: dict | None = None,
) -> dict:
    """
    機動分類パイプラインを一括実行する。

    Args:
        acmi_path: ACMI ファイルパス
        window_sec: スライディングウィンドウ幅 (秒)
        step_sec: ステップ (秒)
        output_dir: 結果出力先ディレクトリ
        aircraft_filter: 特定の航空機IDのみ処理 (None=全て)
        thresholds: ルールベースラベラーの閾値 (None=デフォルト)

    Returns:
        評価結果辞書 (maneuver_classifier.train_and_evaluate の戻り値)
    """
    # ========== 1. ACMI パース ==========
    print(f"[ManeuverMain] Parsing: {acmi_path}")
    parser = ACMIParser()
    frames, _ = parser.parse_file(acmi_path)
    print(f"[ManeuverMain] Title: {parser.title}")
    print(f"[ManeuverMain] Frames: {len(frames)}")

    # ========== 2. 航空機抽出 ==========
    aircraft = get_all_aircraft_ids(frames)
    if not aircraft:
        print("[ManeuverMain] ⚠ 航空機が見つかりません")
        return {}

    print(f"\n[ManeuverMain] === 検出された航空機 ({len(aircraft)}機) ===")
    for oid, name in aircraft.items():
        print(f"  ID={oid}  {name}")

    # フィルタ
    if aircraft_filter:
        aircraft = {k: v for k, v in aircraft.items() if k == aircraft_filter}
        if not aircraft:
            print(f"[ManeuverMain] ⚠ 指定された航空機ID '{aircraft_filter}' が見つかりません")
            return {}

    # ========== 3–4. 特徴量抽出 (航空機ごと) ==========
    all_features = []
    for oid, name in aircraft.items():
        print(f"\n[ManeuverMain] 特徴量抽出: {name} (ID={oid})")
        feat_df = build_feature_matrix(frames, oid, window_sec, step_sec)
        if feat_df.empty:
            print(f"  → データ不足、スキップ")
            continue
        feat_df["aircraft_name"] = name
        all_features.append(feat_df)
        print(f"  → {len(feat_df)} 窓を抽出")

    if not all_features:
        print("[ManeuverMain] ⚠ 抽出可能な特徴量がありません")
        return {}

    features_df = pd.concat(all_features, ignore_index=True)
    print(f"\n[ManeuverMain] 全特徴量: {len(features_df)} 窓 × {len(FEATURE_COLUMNS)} 特徴量")

    # ========== 5. ルールベース仮ラベル ==========
    labeler = ManeuverLabeler(thresholds=thresholds)
    labeled_df = labeler.label_dataframe(features_df)
    labeler.print_distribution(labeled_df)

    # ========== データ十分性チェック ==========
    unique_labels = labeled_df["label"].nunique()
    if unique_labels < 2:
        print(f"\n[ManeuverMain] ⚠ ラベルが {unique_labels} 種類しかありません（最低2種類必要）")
        print("[ManeuverMain] ルールの閾値を調整するか、データを増やしてください")
        # CSV は保存する
        _save_csv(labeled_df, output_dir)
        return {}

    # 各クラスで最低2サンプル必要（stratify のため）
    label_counts = labeled_df["label"].value_counts()
    valid_labels = label_counts[label_counts >= 2].index.tolist()
    if len(valid_labels) < 2:
        print(f"\n[ManeuverMain] ⚠ 2サンプル以上のクラスが2種類未満です")
        _save_csv(labeled_df, output_dir)
        return {}

    # サンプル不足クラスを除外
    if len(valid_labels) < unique_labels:
        removed = set(labeled_df["label"].unique()) - set(valid_labels)
        removed_names = [MANEUVER_CLASSES.get(l, str(l)) for l in removed]
        print(f"\n[ManeuverMain] ⚠ サンプル不足で除外: {removed_names}")
        labeled_df = labeled_df[labeled_df["label"].isin(valid_labels)].copy()

    # ========== 6. Random Forest 学習・評価 ==========
    X = labeled_df[FEATURE_COLUMNS].values.astype(np.float64)
    y = labeled_df["label"].values.astype(int)

    results = train_and_evaluate(
        X, y,
        feature_names=FEATURE_COLUMNS,
        output_dir=output_dir,
    )

    # ========== 7. 結果保存 ==========
    _save_csv(labeled_df, output_dir)

    if results.get("model"):
        model_path = str(Path(output_dir) / "maneuver_rf_model.joblib")
        save_model(results["model"], FEATURE_COLUMNS, model_path)

    print(f"\n[ManeuverMain] === 完了 ===")
    print(f"  結果ディレクトリ: {output_dir}")

    return results


def _save_csv(df: pd.DataFrame, output_dir: str):
    """特徴量+ラベル CSV を保存する。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "maneuver_features_labeled.csv"
    df.to_csv(csv_path, index=False)
    print(f"[ManeuverMain] CSV saved → {csv_path}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Flight Maneuver Classifier — Tacview ACMI ログからの機動分類"
    )
    parser.add_argument("acmi_file", help="Tacview ACMI ファイルパス")
    parser.add_argument(
        "--window", type=float, default=5.0,
        help="スライディングウィンドウ幅 (秒, default: 5.0)",
    )
    parser.add_argument(
        "--step", type=float, default=1.0,
        help="スライディングウィンドウステップ (秒, default: 1.0)",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="結果出力ディレクトリ (default: results/)",
    )
    parser.add_argument(
        "--aircraft-id", default=None,
        help="特定の航空機IDのみ処理",
    )
    parser.add_argument(
        "--list-aircraft", action="store_true",
        help="航空機一覧を表示して終了",
    )

    # ルールベースラベラーの閾値 (14クラス版)
    th_group = parser.add_argument_group("ラベラー閾値 (上級)")
    th_group.add_argument(
        "--heading-delta-th", type=float, default=5.0,
        help="旋回判定 heading_delta 閾値 (deg, default: 5.0)",
    )
    th_group.add_argument(
        "--heading-reversal-th", type=float, default=120.0,
        help="Reversal 判定 heading_delta 閾値 (deg, default: 120.0)",
    )
    th_group.add_argument(
        "--altitude-slope-th", type=float, default=2.0,
        help="Climb/Descent 判定 altitude_slope 閾値 (m/s, default: 2.0)",
    )
    th_group.add_argument(
        "--altitude-slope-steep", type=float, default=10.0,
        help="Dive/Zoom 判定 altitude_slope 閾値 (m/s, default: 10.0)",
    )
    th_group.add_argument(
        "--speed-delta-th", type=float, default=5.0,
        help="加速/減速判定 speed_delta 閾値 (m/s, default: 5.0)",
    )
    th_group.add_argument(
        "--speed-loss-highg", type=float, default=15.0,
        help="High-G Turn 判定 speed 損失閾値 (m/s, default: 15.0)",
    )
    th_group.add_argument(
        "--roll-std-jinking", type=float, default=20.0,
        help="Jinking 判定 roll_std 閾値 (deg, default: 20.0)",
    )
    th_group.add_argument(
        "--heading-std-jinking", type=float, default=10.0,
        help="Jinking 判定 heading_std 閾値 (deg, default: 10.0)",
    )

    args = parser.parse_args()

    # ---- 航空機一覧モード ----
    if args.list_aircraft:
        acmi_parser = ACMIParser()
        frames, _ = acmi_parser.parse_file(args.acmi_file)
        aircraft = get_all_aircraft_ids(frames)
        print(f"\n{'ID':>12} {'Name'}")
        print("-" * 40)
        for oid, name in sorted(aircraft.items()):
            print(f"{oid:>12} {name}")
        return

    # ---- パイプライン実行 ----
    thresholds = {
        "heading_delta_threshold": args.heading_delta_th,
        "heading_reversal_threshold": args.heading_reversal_th,
        "altitude_slope_threshold": args.altitude_slope_th,
        "altitude_slope_steep": args.altitude_slope_steep,
        "speed_delta_threshold": args.speed_delta_th,
        "speed_loss_highg": args.speed_loss_highg,
        "roll_std_jinking": args.roll_std_jinking,
        "heading_std_jinking": args.heading_std_jinking,
    }

    run_pipeline(
        acmi_path=args.acmi_file,
        window_sec=args.window,
        step_sec=args.step,
        output_dir=args.output_dir,
        aircraft_filter=args.aircraft_id,
        thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
```

```diff:test_maneuver_labeler.py
===
"""
Flight Maneuver Classifier — maneuver_labeler のテスト (14クラス版)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maneuver_labeler import ManeuverLabeler, MANEUVER_CLASSES, NUM_CLASSES


# ============================================================
# 基本飛行クラス (0–4)
# ============================================================

class TestBasicFlightClasses:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_straight_and_level(self):
        """全指標が閾値以下 → Straight & Level (0)"""
        row = {
            "heading_delta": 1.0, "heading_std": 1.0,
            "altitude_slope": 0.5, "speed_delta": 2.0,
            "roll_std": 2.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 0

    def test_acceleration(self):
        """speed 増加 + 直線 + level → Acceleration (1)"""
        row = {
            "heading_delta": 1.0, "heading_std": 1.0,
            "altitude_slope": 0.0, "speed_delta": 10.0,
            "roll_std": 2.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 1

    def test_deceleration(self):
        """speed 減少 + 直線 + level → Deceleration (2)"""
        row = {
            "heading_delta": 1.0, "heading_std": 1.0,
            "altitude_slope": 0.0, "speed_delta": -10.0,
            "roll_std": 2.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 2

    def test_steady_climb(self):
        """altitude 上昇 + 旋回なし → Steady Climb (3)"""
        row = {
            "heading_delta": 2.0, "heading_std": 1.0,
            "altitude_slope": 5.0, "speed_delta": 0.0,
            "roll_std": 2.0, "pitch_mean": 5.0,
        }
        assert self.labeler.label_single(row) == 3

    def test_steady_descent(self):
        """altitude 降下 + 旋回なし → Steady Descent (4)"""
        row = {
            "heading_delta": 2.0, "heading_std": 1.0,
            "altitude_slope": -5.0, "speed_delta": 0.0,
            "roll_std": 2.0, "pitch_mean": -3.0,
        }
        assert self.labeler.label_single(row) == 4


# ============================================================
# 旋回系クラス (5–8)
# ============================================================

class TestTurnClasses:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_level_turn(self):
        """heading 変化 + altitude flat → Level Turn (5)"""
        row = {
            "heading_delta": 15.0, "heading_std": 5.0,
            "altitude_slope": 0.0, "speed_delta": -3.0,
            "roll_std": 5.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 5

    def test_level_turn_left(self):
        """heading 負方向 → Level Turn (5)"""
        row = {
            "heading_delta": -15.0, "heading_std": 5.0,
            "altitude_slope": 0.0, "speed_delta": -3.0,
            "roll_std": 5.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 5

    def test_climbing_turn(self):
        """heading 変化 + altitude 上昇 → Climbing Turn (6)"""
        row = {
            "heading_delta": 20.0, "heading_std": 5.0,
            "altitude_slope": 5.0, "speed_delta": -5.0,
            "roll_std": 5.0, "pitch_mean": 5.0,
        }
        assert self.labeler.label_single(row) == 6

    def test_descending_turn(self):
        """heading 変化 + altitude 降下 → Descending Turn (7)"""
        row = {
            "heading_delta": -20.0, "heading_std": 5.0,
            "altitude_slope": -5.0, "speed_delta": -3.0,
            "roll_std": 5.0, "pitch_mean": -3.0,
        }
        assert self.labeler.label_single(row) == 7

    def test_high_g_turn(self):
        """heading 変化 + speed 大幅減少 → High-G Turn (8)"""
        row = {
            "heading_delta": 30.0, "heading_std": 8.0,
            "altitude_slope": 0.0, "speed_delta": -25.0,
            "roll_std": 10.0, "pitch_mean": 2.0,
        }
        assert self.labeler.label_single(row) == 8


# ============================================================
# 戦術機動クラス (9–13)
# ============================================================

class TestTacticalClasses:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_dive(self):
        """pitch 大きく負 + altitude 急降下 + speed 増加 → Dive (9)"""
        row = {
            "heading_delta": 2.0, "heading_std": 2.0,
            "altitude_slope": -20.0, "speed_delta": 15.0,
            "roll_std": 3.0, "pitch_mean": -25.0,
        }
        assert self.labeler.label_single(row) == 9

    def test_zoom_climb(self):
        """pitch 大きく正 + altitude 急上昇 + speed 減少 → Zoom Climb (10)"""
        row = {
            "heading_delta": 2.0, "heading_std": 2.0,
            "altitude_slope": 20.0, "speed_delta": -15.0,
            "roll_std": 3.0, "pitch_mean": 25.0,
        }
        assert self.labeler.label_single(row) == 10

    def test_reversal(self):
        """heading 大幅変化 (≥120°) → Reversal (11)"""
        row = {
            "heading_delta": 150.0, "heading_std": 30.0,
            "altitude_slope": -5.0, "speed_delta": -10.0,
            "roll_std": 15.0, "pitch_mean": -10.0,
        }
        assert self.labeler.label_single(row) == 11

    def test_reversal_negative(self):
        """heading 大幅変化 (負方向) → Reversal (11)"""
        row = {
            "heading_delta": -130.0, "heading_std": 30.0,
            "altitude_slope": 5.0, "speed_delta": -10.0,
            "roll_std": 15.0, "pitch_mean": 10.0,
        }
        assert self.labeler.label_single(row) == 11

    def test_jinking(self):
        """roll_std + heading_std が大きい → Jinking (12)"""
        row = {
            "heading_delta": 5.0, "heading_std": 15.0,
            "altitude_slope": 0.0, "speed_delta": 0.0,
            "roll_std": 25.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 12

    def test_extension(self):
        """speed 増加 + 直線 + 微降下 → Extension (13)"""
        row = {
            "heading_delta": 2.0, "heading_std": 2.0,
            "altitude_slope": -1.0, "speed_delta": 10.0,
            "roll_std": 3.0, "pitch_mean": -1.0,
        }
        assert self.labeler.label_single(row) == 13


# ============================================================
# 優先度テスト
# ============================================================

class TestPriorityRules:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_jinking_over_turn(self):
        """Jinking が Turn より優先"""
        row = {
            "heading_delta": 20.0, "heading_std": 15.0,
            "altitude_slope": 0.0, "speed_delta": 0.0,
            "roll_std": 25.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 12

    def test_reversal_over_high_g(self):
        """Reversal が High-G Turn より優先"""
        row = {
            "heading_delta": 150.0, "heading_std": 20.0,
            "altitude_slope": 0.0, "speed_delta": -30.0,
            "roll_std": 10.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 11

    def test_high_g_over_level_turn(self):
        """High-G Turn が Level Turn より優先"""
        row = {
            "heading_delta": 30.0, "heading_std": 8.0,
            "altitude_slope": 0.0, "speed_delta": -20.0,
            "roll_std": 10.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 8

    def test_climb_over_acceleration(self):
        """Steady Climb が Acceleration より優先"""
        row = {
            "heading_delta": 1.0, "heading_std": 1.0,
            "altitude_slope": 5.0, "speed_delta": 10.0,
            "roll_std": 2.0, "pitch_mean": 5.0,
        }
        assert self.labeler.label_single(row) == 3

    def test_dive_over_descent(self):
        """Dive が Steady Descent より優先"""
        row = {
            "heading_delta": 2.0, "heading_std": 2.0,
            "altitude_slope": -15.0, "speed_delta": 10.0,
            "roll_std": 3.0, "pitch_mean": -20.0,
        }
        assert self.labeler.label_single(row) == 9


# ============================================================
# DataFrame / 設定テスト
# ============================================================

class TestLabelDataFrame:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_labels_added(self):
        """DataFrame に label / label_name 列が追加される"""
        df = pd.DataFrame({
            "heading_delta": [1.0, 150.0],
            "heading_std": [1.0, 30.0],
            "altitude_slope": [0.0, -5.0],
            "speed_delta": [0.0, -10.0],
            "roll_std": [2.0, 15.0],
            "pitch_mean": [0.0, -10.0],
        })
        result = self.labeler.label_dataframe(df)
        assert "label" in result.columns
        assert "label_name" in result.columns
        assert result.iloc[0]["label"] == 0   # Straight & Level
        assert result.iloc[1]["label"] == 11  # Reversal

    def test_empty_dataframe(self):
        """空 DataFrame"""
        df = pd.DataFrame()
        result = self.labeler.label_dataframe(df)
        assert "label" in result.columns

    def test_custom_thresholds(self):
        """閾値をカスタマイズ"""
        labeler = ManeuverLabeler(thresholds={
            "heading_delta_threshold": 30.0,
        })
        row = {
            "heading_delta": 20.0, "heading_std": 5.0,
            "altitude_slope": 0.0, "speed_delta": 0.0,
            "roll_std": 2.0, "pitch_mean": 0.0,
        }
        # 20° < threshold 30° → not a turn → Straight & Level
        assert labeler.label_single(row) == 0

    def test_num_classes(self):
        """14クラスが定義されている"""
        assert NUM_CLASSES == 14
        assert len(MANEUVER_CLASSES) == 14
```

```diff:test_maneuver_acmi_export.py
"""
Tests for maneuver_acmi_export.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maneuver_acmi_export import (
    build_annotation_schedule,
    extract_frame_markers,
    extract_object_first_frames,
    inject_annotation_lines,
    load_labeled_windows,
)


SAMPLE_ACMI = """\
FileType=text/acmi/tacview
FileVersion=2.2
0,ReferenceTime=2024-01-15T12:00:00Z
#0
1,T=44.1|41.1|5000,Name=F-16C,Type=Air+FixedWing
#1
1,T=44.2|41.2|5100
#2
1,T=44.3|41.3|5200
"""


class TestBuildAnnotationSchedule:
    def test_assigns_to_nearest_frame_from_center(self):
        frame_markers = extract_frame_markers(SAMPLE_ACMI)
        labels_df = pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 1.2,
                    "aircraft_id": "1",
                    "label": 1,
                    "label_name": "Climb",
                }
            ]
        )

        schedule = build_annotation_schedule(frame_markers, labels_df, time_anchor="center")

        assert "1" in schedule
        assert schedule["1"] == [
            "1,ManeuverLabel=Climb,ManeuverLabelId=1,"
            "ManeuverWindowStart=0,ManeuverWindowEnd=1.2,ManeuverSampleTime=0.6"
        ]

    def test_last_update_wins_for_same_frame_and_object(self):
        frame_markers = extract_frame_markers(SAMPLE_ACMI)
        labels_df = pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 1.0,
                    "aircraft_id": "1",
                    "label": 0,
                    "label_name": "Level Flight",
                },
                {
                    "window_start": 0.1,
                    "window_end": 1.1,
                    "aircraft_id": "1",
                    "label": 3,
                    "label_name": "Left Turn",
                },
            ]
        )

        schedule = build_annotation_schedule(frame_markers, labels_df, time_anchor="start")

        assert schedule["0"] == [
            "1,ManeuverLabel=Left Turn,ManeuverLabelId=3,"
            "ManeuverWindowStart=0.1,ManeuverWindowEnd=1.1,ManeuverSampleTime=0.1"
        ]

    def test_skips_unknown_objects_and_respects_first_frame(self):
        frame_markers = extract_frame_markers(SAMPLE_ACMI)
        first_frames = extract_object_first_frames(SAMPLE_ACMI)
        labels_df = pd.DataFrame(
            [
                {
                    "window_start": -2.0,
                    "window_end": 0.0,
                    "aircraft_id": "1",
                    "label": 0,
                    "label_name": "Level Flight",
                },
                {
                    "window_start": 0.0,
                    "window_end": 1.0,
                    "aircraft_id": "999",
                    "label": 1,
                    "label_name": "Climb",
                },
            ]
        )

        schedule = build_annotation_schedule(
            frame_markers,
            labels_df,
            time_anchor="start",
            first_frame_by_object=first_frames,
        )

        assert list(schedule.keys()) == ["0"]
        assert schedule["0"] == [
            "1,ManeuverLabel=Level Flight,ManeuverLabelId=0,"
            "ManeuverWindowStart=-2,ManeuverWindowEnd=0,ManeuverSampleTime=-2"
        ]


class TestInjectAnnotationLines:
    def test_inserts_updates_immediately_after_frame_marker(self):
        annotated = inject_annotation_lines(
            SAMPLE_ACMI,
            {
                "1": [
                    "1,ManeuverLabel=Climb,ManeuverLabelId=1,"
                    "ManeuverWindowStart=0,ManeuverWindowEnd=1.2,ManeuverSampleTime=0.6"
                ]
            },
        )

        expected = """\
FileType=text/acmi/tacview
FileVersion=2.2
0,ReferenceTime=2024-01-15T12:00:00Z
#0
1,T=44.1|41.1|5000,Name=F-16C,Type=Air+FixedWing
#1
1,ManeuverLabel=Climb,ManeuverLabelId=1,ManeuverWindowStart=0,ManeuverWindowEnd=1.2,ManeuverSampleTime=0.6
1,T=44.2|41.2|5100
#2
1,T=44.3|41.3|5200
"""
        assert annotated == expected


class TestLoadLabeledWindows:
    def test_fills_label_name_from_label_column(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 5.0,
                    "aircraft_id": "302",
                    "label": 4,
                }
            ]
        ).to_csv(csv_path, index=False)

        labels_df = load_labeled_windows(str(csv_path))

        assert labels_df.iloc[0]["label_name"] == "Right Turn"
===
"""
Tests for maneuver_acmi_export.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maneuver_acmi_export import (
    build_annotation_schedule,
    extract_frame_markers,
    extract_object_first_frames,
    inject_annotation_lines,
    load_labeled_windows,
)


SAMPLE_ACMI = """\
FileType=text/acmi/tacview
FileVersion=2.2
0,ReferenceTime=2024-01-15T12:00:00Z
#0
1,T=44.1|41.1|5000,Name=F-16C,Type=Air+FixedWing
#1
1,T=44.2|41.2|5100
#2
1,T=44.3|41.3|5200
"""


class TestBuildAnnotationSchedule:
    def test_assigns_to_nearest_frame_from_center(self):
        frame_markers = extract_frame_markers(SAMPLE_ACMI)
        labels_df = pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 1.2,
                    "aircraft_id": "1",
                    "label": 1,
                    "label_name": "Climb",
                }
            ]
        )

        schedule = build_annotation_schedule(frame_markers, labels_df, time_anchor="center")

        assert "1" in schedule
        assert schedule["1"] == [
            "1,ManeuverLabel=Climb,ManeuverLabelId=1,"
            "ManeuverWindowStart=0,ManeuverWindowEnd=1.2,ManeuverSampleTime=0.6"
        ]

    def test_last_update_wins_for_same_frame_and_object(self):
        frame_markers = extract_frame_markers(SAMPLE_ACMI)
        labels_df = pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 1.0,
                    "aircraft_id": "1",
                    "label": 0,
                    "label_name": "Level Flight",
                },
                {
                    "window_start": 0.1,
                    "window_end": 1.1,
                    "aircraft_id": "1",
                    "label": 3,
                    "label_name": "Left Turn",
                },
            ]
        )

        schedule = build_annotation_schedule(frame_markers, labels_df, time_anchor="start")

        assert schedule["0"] == [
            "1,ManeuverLabel=Left Turn,ManeuverLabelId=3,"
            "ManeuverWindowStart=0.1,ManeuverWindowEnd=1.1,ManeuverSampleTime=0.1"
        ]

    def test_skips_unknown_objects_and_respects_first_frame(self):
        frame_markers = extract_frame_markers(SAMPLE_ACMI)
        first_frames = extract_object_first_frames(SAMPLE_ACMI)
        labels_df = pd.DataFrame(
            [
                {
                    "window_start": -2.0,
                    "window_end": 0.0,
                    "aircraft_id": "1",
                    "label": 0,
                    "label_name": "Level Flight",
                },
                {
                    "window_start": 0.0,
                    "window_end": 1.0,
                    "aircraft_id": "999",
                    "label": 1,
                    "label_name": "Climb",
                },
            ]
        )

        schedule = build_annotation_schedule(
            frame_markers,
            labels_df,
            time_anchor="start",
            first_frame_by_object=first_frames,
        )

        assert list(schedule.keys()) == ["0"]
        assert schedule["0"] == [
            "1,ManeuverLabel=Level Flight,ManeuverLabelId=0,"
            "ManeuverWindowStart=-2,ManeuverWindowEnd=0,ManeuverSampleTime=-2"
        ]


class TestInjectAnnotationLines:
    def test_inserts_updates_immediately_after_frame_marker(self):
        annotated = inject_annotation_lines(
            SAMPLE_ACMI,
            {
                "1": [
                    "1,ManeuverLabel=Climb,ManeuverLabelId=1,"
                    "ManeuverWindowStart=0,ManeuverWindowEnd=1.2,ManeuverSampleTime=0.6"
                ]
            },
        )

        expected = """\
FileType=text/acmi/tacview
FileVersion=2.2
0,ReferenceTime=2024-01-15T12:00:00Z
#0
1,T=44.1|41.1|5000,Name=F-16C,Type=Air+FixedWing
#1
1,ManeuverLabel=Climb,ManeuverLabelId=1,ManeuverWindowStart=0,ManeuverWindowEnd=1.2,ManeuverSampleTime=0.6
1,T=44.2|41.2|5100
#2
1,T=44.3|41.3|5200
"""
        assert annotated == expected


class TestLoadLabeledWindows:
    def test_fills_label_name_from_label_column(self, tmp_path):
        csv_path = tmp_path / "labels.csv"
        pd.DataFrame(
            [
                {
                    "window_start": 0.0,
                    "window_end": 5.0,
                    "aircraft_id": "302",
                    "label": 4,
                }
            ]
        ).to_csv(csv_path, index=False)

        labels_df = load_labeled_windows(str(csv_path))

        assert labels_df.iloc[0]["label_name"] == "Steady Descent"
```

## テスト結果

```
96 passed in 5.07s
```

## 使い方

```bash
# 基本
python src/maneuver_main.py flight.acmi

# 閾値カスタマイズ例
python src/maneuver_main.py flight.acmi \
  --heading-delta-th 8 \
  --altitude-slope-steep 12 \
  --speed-loss-highg 20
```
