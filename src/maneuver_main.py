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
    6. Random Forest / XGBoost 学習・評価
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
from maneuver_acmi_export import annotate_acmi_file
from maneuver_labeler import ManeuverLabeler, MANEUVER_CLASSES
from maneuver_classifier import train_and_evaluate, save_model


def run_pipeline(
    acmi_path: str,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    output_dir: str = "results",
    aircraft_filter: str | None = None,
    thresholds: dict | None = None,
    annotate_acmi: bool = True,
    annotated_acmi_path: str | None = None,
    model_type: str = "random_forest",
    use_gpu: bool = False,
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
        annotate_acmi: True の場合、注釈付き ACMI も自動出力する
        annotated_acmi_path: 出力先 ACMI パス (None=output_dir 配下に自動決定)
        model_type: "random_forest" または "xgboost"
        use_gpu: True の場合、XGBoost で CUDA GPU を使用する

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
    export_df = labeled_df.copy()

    # ========== データ十分性チェック ==========
    unique_labels = labeled_df["label"].nunique()
    if unique_labels < 2:
        print(f"\n[ManeuverMain] ⚠ ラベルが {unique_labels} 種類しかありません（最低2種類必要）")
        print("[ManeuverMain] ルールの閾値を調整するか、データを増やしてください")
        # CSV は保存する
        csv_path = _save_csv(export_df, output_dir)
        _maybe_export_annotated_acmi(
            acmi_path=acmi_path,
            labels_csv_path=csv_path,
            output_dir=output_dir,
            annotate_acmi=annotate_acmi,
            annotated_acmi_path=annotated_acmi_path,
        )
        return {}

    # 各クラスで最低2サンプル必要（stratify のため）
    label_counts = labeled_df["label"].value_counts()
    valid_labels = label_counts[label_counts >= 2].index.tolist()
    if len(valid_labels) < 2:
        print(f"\n[ManeuverMain] ⚠ 2サンプル以上のクラスが2種類未満です")
        csv_path = _save_csv(export_df, output_dir)
        _maybe_export_annotated_acmi(
            acmi_path=acmi_path,
            labels_csv_path=csv_path,
            output_dir=output_dir,
            annotate_acmi=annotate_acmi,
            annotated_acmi_path=annotated_acmi_path,
        )
        return {}

    # サンプル不足クラスを除外
    if len(valid_labels) < unique_labels:
        removed = set(labeled_df["label"].unique()) - set(valid_labels)
        removed_names = [MANEUVER_CLASSES.get(l, str(l)) for l in removed]
        print(f"\n[ManeuverMain] ⚠ サンプル不足で除外: {removed_names}")
        labeled_df = labeled_df[labeled_df["label"].isin(valid_labels)].copy()

    # ========== 6. 学習・評価 ==========
    X = labeled_df[FEATURE_COLUMNS].values.astype(np.float64)
    y = labeled_df["label"].values.astype(int)

    results = train_and_evaluate(
        X, y,
        feature_names=FEATURE_COLUMNS,
        output_dir=output_dir,
        model_type=model_type,
        use_gpu=use_gpu,
    )

    # ========== 7. 結果保存 ==========
    csv_path = _save_csv(labeled_df, output_dir)

    if results.get("model"):
        model_path = str(Path(output_dir) / _default_model_filename(model_type))
        save_model(
            results["model"],
            FEATURE_COLUMNS,
            model_path,
            metadata={"model_type": model_type, "use_gpu": use_gpu},
        )
        export_df["predicted_label"] = results["model"].predict(
            export_df[FEATURE_COLUMNS].values.astype(np.float64)
        ).astype(int)
        export_df["predicted_label_name"] = export_df["predicted_label"].map(
            MANEUVER_CLASSES
        )

    csv_path = _save_csv(export_df, output_dir)

    annotated_path = _maybe_export_annotated_acmi(
        acmi_path=acmi_path,
        labels_csv_path=csv_path,
        output_dir=output_dir,
        annotate_acmi=annotate_acmi,
        annotated_acmi_path=annotated_acmi_path,
    )
    if annotated_path:
        results["annotated_acmi_path"] = annotated_path

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
    return str(csv_path)


def _default_model_filename(model_type: str) -> str:
    if model_type == "xgboost":
        return "maneuver_xgboost_model.joblib"
    return "maneuver_rf_model.joblib"


def _default_annotated_acmi_path(acmi_path: str, output_dir: str) -> str:
    source = Path(acmi_path)
    output = Path(output_dir)
    name = source.name

    if name.endswith(".zip.acmi"):
        return str(output / f"{name[:-9]}.maneuver.zip.acmi")
    if name.endswith(".acmi"):
        return str(output / f"{source.stem}.maneuver.acmi")
    return str(output / f"{name}.maneuver.acmi")


def _maybe_export_annotated_acmi(
    acmi_path: str,
    labels_csv_path: str,
    output_dir: str,
    annotate_acmi: bool,
    annotated_acmi_path: str | None = None,
) -> str | None:
    if not annotate_acmi:
        return None

    output_path = annotated_acmi_path or _default_annotated_acmi_path(
        acmi_path,
        output_dir,
    )

    try:
        result = annotate_acmi_file(
            acmi_path=acmi_path,
            labels_csv_path=labels_csv_path,
            output_path=output_path,
        )
    except Exception as exc:
        print(f"[ManeuverMain] ⚠ 注釈付き ACMI の出力に失敗: {exc}")
        return None

    print(f"[ManeuverMain] Annotated ACMI saved → {result['output_path']}")
    return str(result["output_path"])


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
        "--model-type",
        choices=["random_forest", "xgboost"],
        default="random_forest",
        help="学習に使う分類器 (default: random_forest)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="XGBoost 選択時に CUDA GPU を使用する",
    )
    parser.add_argument(
        "--list-aircraft", action="store_true",
        help="航空機一覧を表示して終了",
    )
    parser.add_argument(
        "--no-annotate-acmi", action="store_true",
        help="注釈付き ACMI の自動出力を無効化",
    )
    parser.add_argument(
        "--annotated-acmi-output", default=None,
        help="注釈付き ACMI の出力先 (default: output-dir 配下に自動決定)",
    )

    # ルールベースラベラーの閾値 (14クラス / G-Load対応版)
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
        "--g-load-high-g", type=float, default=4.0,
        help="High-G Turn 判定 g_load 閾値 (G, default: 4.0)",
    )
    th_group.add_argument(
        "--g-load-unloaded", type=float, default=0.5,
        help="Dive 判定 unloaded g_load 閾値 (G, default: 0.5)",
    )
    th_group.add_argument(
        "--g-load-zoom", type=float, default=2.0,
        help="Zoom Climb 判定 g_load 閾値 (G, default: 2.0)",
    )
    th_group.add_argument(
        "--g-std-jinking", type=float, default=1.0,
        help="Jinking 判定 g_load_std 閾値 (G, default: 1.0)",
    )
    th_group.add_argument(
        "--roll-std-jinking", type=float, default=15.0,
        help="Jinking 判定 roll_std 閾値 (deg, default: 15.0)",
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
        "g_load_high_g": args.g_load_high_g,
        "g_load_unloaded": args.g_load_unloaded,
        "g_load_zoom": args.g_load_zoom,
        "g_load_std_jinking": args.g_std_jinking,
        "roll_std_jinking": args.roll_std_jinking,
    }

    run_pipeline(
        acmi_path=args.acmi_file,
        window_sec=args.window,
        step_sec=args.step,
        output_dir=args.output_dir,
        aircraft_filter=args.aircraft_id,
        thresholds=thresholds,
        annotate_acmi=not args.no_annotate_acmi,
        annotated_acmi_path=args.annotated_acmi_output,
        model_type=args.model_type,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
