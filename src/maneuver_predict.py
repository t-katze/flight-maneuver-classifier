"""
Apply a trained maneuver model to one or more ACMI files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maneuver_acmi_export import annotate_acmi_file
from maneuver_classifier import load_model
from maneuver_labeler import add_threshold_arguments, thresholds_from_args
from maneuver_pipeline import (
    acmi_output_stem,
    add_model_predictions,
    default_annotated_acmi_path,
    expand_acmi_inputs,
    extract_labeled_features_from_acmi,
    save_labeled_dataframe,
)


def predict_for_acmi_files(
    acmi_paths: list[str],
    model_path: str,
    window_sec: float = 5.0,
    step_sec: float = 2.5,
    output_dir: str = "results/predict",
    aircraft_filter: str | None = None,
    thresholds: dict | None = None,
    recursive: bool = False,
) -> list[dict]:
    acmi_paths = expand_acmi_inputs(acmi_paths, recursive=recursive)
    model, feature_columns = load_model(model_path)
    results = []

    for acmi_path in acmi_paths:
        stem = acmi_output_stem(acmi_path)
        per_file_output_dir = Path(output_dir) / stem
        print(f"[ManeuverPredict] Predicting: {acmi_path}")

        labeled_df, meta = extract_labeled_features_from_acmi(
            acmi_path=acmi_path,
            window_sec=window_sec,
            step_sec=step_sec,
            aircraft_filter=aircraft_filter,
            thresholds=thresholds,
            verbose=True,
        )
        if labeled_df.empty:
            print("[ManeuverPredict] ⚠ 抽出可能な特徴量がありません")
            continue

        predicted_df = add_model_predictions(labeled_df, model, feature_columns)
        csv_path = save_labeled_dataframe(
            predicted_df,
            str(per_file_output_dir / "maneuver_features_labeled.csv"),
        )
        annotated_path = default_annotated_acmi_path(acmi_path, str(per_file_output_dir))
        annotate_acmi_file(
            acmi_path=acmi_path,
            labels_csv_path=csv_path,
            output_path=annotated_path,
        )

        print(f"[ManeuverPredict] CSV saved → {csv_path}")
        print(f"[ManeuverPredict] Annotated ACMI saved → {annotated_path}")
        results.append(
            {
                "acmi_path": acmi_path,
                "csv_path": csv_path,
                "annotated_acmi_path": annotated_path,
                "window_count": len(predicted_df),
                "title": meta["title"],
            }
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Apply a trained maneuver model to one or more Tacview ACMI files or directories"
    )
    parser.add_argument(
        "acmi_files",
        nargs="+",
        help="Tacview ACMI file paths or directories containing .acmi files",
    )
    parser.add_argument("--model", required=True, help="Trained maneuver model (.joblib)")
    parser.add_argument("--window", type=float, default=5.0)
    parser.add_argument("--step", type=float, default=2.5)
    parser.add_argument("--output-dir", default="results/predict")
    parser.add_argument("--aircraft-id", default=None)
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="ディレクトリ入力時にサブディレクトリも再帰的に探索する",
    )

    add_threshold_arguments(parser)

    args = parser.parse_args()
    thresholds = thresholds_from_args(args)

    predict_for_acmi_files(
        acmi_paths=args.acmi_files,
        model_path=args.model,
        window_sec=args.window,
        step_sec=args.step,
        output_dir=args.output_dir,
        aircraft_filter=args.aircraft_id,
        thresholds=thresholds,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()
