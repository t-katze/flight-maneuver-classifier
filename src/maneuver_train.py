"""
Train a maneuver classification model from one or more ACMI files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maneuver_acmi_export import annotate_acmi_file
from maneuver_classifier import save_model, train_and_evaluate
from maneuver_feature_engine import FEATURE_COLUMNS
from maneuver_labeler import add_threshold_arguments, thresholds_from_args
from maneuver_pipeline import (
    acmi_output_stem,
    add_model_predictions,
    build_combined_labeled_dataset,
    default_annotated_acmi_path,
    expand_acmi_inputs,
    prepare_training_dataframe,
    save_labeled_dataframe,
)


def train_from_acmi_files(
    acmi_paths: list[str],
    window_sec: float = 5.0,
    step_sec: float = 2.5,
    output_dir: str = "results/train",
    aircraft_filter: str | None = None,
    thresholds: dict | None = None,
    recursive: bool = False,
    annotate_inputs: bool = True,
    prediction_output_dir: str | None = None,
    model_type: str = "random_forest",
    use_gpu: bool = False,
) -> dict:
    acmi_paths = expand_acmi_inputs(acmi_paths, recursive=recursive)
    print(f"[ManeuverTrain] Inputs: {len(acmi_paths)} ACMI files")
    export_df = build_combined_labeled_dataset(
        acmi_paths=acmi_paths,
        window_sec=window_sec,
        step_sec=step_sec,
        aircraft_filter=aircraft_filter,
        thresholds=thresholds,
        verbose=True,
    )

    if export_df.empty:
        print("[ManeuverTrain] ⚠ 抽出可能な特徴量がありません")
        return {}

    training_df, prep = prepare_training_dataframe(export_df)
    csv_path = save_labeled_dataframe(
        export_df,
        str(Path(output_dir) / "maneuver_features_labeled.csv"),
    )
    print(f"[ManeuverTrain] CSV saved → {csv_path}")

    if prep["removed_names"]:
        print(f"[ManeuverTrain] ⚠ サンプル不足で除外: {prep['removed_names']}")

    if training_df.empty:
        print("[ManeuverTrain] ⚠ 学習に必要なクラス数が不足しています")
        return {}

    X = training_df[FEATURE_COLUMNS].values.astype(np.float64)
    y = training_df["label"].values.astype(int)
    results = train_and_evaluate(
        X,
        y,
        feature_names=FEATURE_COLUMNS,
        output_dir=output_dir,
        model_type=model_type,
        use_gpu=use_gpu,
    )

    model_path = str(Path(output_dir) / _default_model_filename(model_type))
    save_model(
        results["model"],
        FEATURE_COLUMNS,
        model_path,
        metadata={"model_type": model_type, "use_gpu": use_gpu},
    )

    export_with_pred = add_model_predictions(export_df, results["model"], FEATURE_COLUMNS)
    csv_path = save_labeled_dataframe(
        export_with_pred,
        str(Path(output_dir) / "maneuver_features_labeled.csv"),
    )
    print(f"[ManeuverTrain] CSV updated → {csv_path}")
    print(f"[ManeuverTrain] Model saved → {model_path}")

    auto_outputs = _export_predictions_for_inputs(
        export_with_pred=export_with_pred,
        annotate_inputs=annotate_inputs,
        prediction_output_dir=prediction_output_dir or str(Path(output_dir) / "predictions"),
    )
    if auto_outputs:
        print(f"[ManeuverTrain] Auto prediction outputs → {prediction_output_dir or str(Path(output_dir) / 'predictions')}")

    results["model_path"] = model_path
    results["dataset_csv"] = csv_path
    results["prediction_outputs"] = auto_outputs
    return results


def _export_predictions_for_inputs(
    export_with_pred,
    annotate_inputs: bool,
    prediction_output_dir: str,
) -> list[dict]:
    if not annotate_inputs or export_with_pred.empty or "source_acmi" not in export_with_pred.columns:
        return []

    outputs = []
    output_root = Path(prediction_output_dir)

    for acmi_path, acmi_df in export_with_pred.groupby("source_acmi", sort=True):
        per_file_output_dir = output_root / acmi_output_stem(acmi_path)
        csv_path = save_labeled_dataframe(
            acmi_df,
            str(per_file_output_dir / "maneuver_features_labeled.csv"),
        )
        annotated_path = default_annotated_acmi_path(acmi_path, str(per_file_output_dir))
        annotate_acmi_file(
            acmi_path=acmi_path,
            labels_csv_path=csv_path,
            output_path=annotated_path,
        )
        print(f"[ManeuverTrain] Annotated ACMI saved → {annotated_path}")
        outputs.append(
            {
                "acmi_path": acmi_path,
                "csv_path": csv_path,
                "annotated_acmi_path": annotated_path,
            }
        )

    return outputs


def _default_model_filename(model_type: str) -> str:
    if model_type == "xgboost":
        return "maneuver_xgboost_model.joblib"
    return "maneuver_rf_model.joblib"


def main():
    parser = argparse.ArgumentParser(
        description="Train maneuver model from one or more Tacview ACMI files or directories"
    )
    parser.add_argument(
        "acmi_files",
        nargs="+",
        help="Tacview ACMI file paths or directories containing .acmi files",
    )
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--step", type=float, default=2.5)
    parser.add_argument("--output-dir", default="results/train")
    parser.add_argument("--aircraft-id", default=None)
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
        "--recursive",
        action="store_true",
        help="ディレクトリ入力時にサブディレクトリも再帰的に探索する",
    )
    parser.add_argument(
        "--no-annotate-inputs",
        action="store_true",
        help="学習完了後の入力 ACMI への自動書き戻しを無効化する",
    )
    parser.add_argument(
        "--prediction-output-dir",
        default=None,
        help="自動書き戻し結果の出力先 (default: <output-dir>/predictions)",
    )

    add_threshold_arguments(parser)

    args = parser.parse_args()
    thresholds = thresholds_from_args(args)

    train_from_acmi_files(
        acmi_paths=args.acmi_files,
        window_sec=args.window,
        step_sec=args.step,
        output_dir=args.output_dir,
        aircraft_filter=args.aircraft_id,
        thresholds=thresholds,
        recursive=args.recursive,
        annotate_inputs=not args.no_annotate_inputs,
        prediction_output_dir=args.prediction_output_dir,
        model_type=args.model_type,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
