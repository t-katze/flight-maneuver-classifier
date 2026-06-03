"""
Retrain a maneuver classifier from an existing labeled CSV plus optional manual Tacview edits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from maneuver_classifier import save_model, train_and_evaluate
from maneuver_feature_engine import FEATURE_COLUMNS
from maneuver_manual_labels import (
    DEFAULT_MATCH_TOLERANCE_SEC,
    load_and_apply_manual_label_edits,
    save_manual_label_report,
)
from maneuver_pipeline import prepare_training_dataframe, save_labeled_dataframe


def retrain_from_labeled_csv(
    labeled_csv_path: str,
    manual_edits_csv_path: str | None = None,
    output_dir: str = "results/retrain",
    match_tolerance_sec: float = DEFAULT_MATCH_TOLERANCE_SEC,
    model_type: str = "random_forest",
    use_gpu: bool = False,
) -> dict:
    merged_df, report = load_and_apply_manual_label_edits(
        labeled_csv_path=labeled_csv_path,
        manual_edits_csv_path=manual_edits_csv_path,
        match_tolerance_sec=match_tolerance_sec,
    )

    merged_df["label"] = merged_df["training_label"].astype(int)
    merged_df["label_name"] = merged_df["training_label_name"].astype(str)

    training_df, prep = prepare_training_dataframe(merged_df)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    merged_csv_path = save_labeled_dataframe(
        merged_df,
        str(output_root / "maneuver_features_with_manual_labels.csv"),
    )
    unmatched_csv_path = save_manual_label_report(
        report,
        str(output_root / "manual_label_unmatched.csv"),
    )

    if prep["removed_names"]:
        print(f"[ManeuverRetrain] ⚠ サンプル不足で除外: {prep['removed_names']}")

    if training_df.empty:
        print("[ManeuverRetrain] ⚠ 学習に必要なクラス数が不足しています")
        return {
            "dataset_csv": merged_csv_path,
            "unmatched_csv": unmatched_csv_path,
            "manual_label_report": report,
        }

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

    model_filename = (
        "maneuver_xgboost_model.joblib"
        if model_type == "xgboost"
        else "maneuver_rf_model.joblib"
    )
    model_path = str(output_root / model_filename)
    save_model(
        results["model"],
        FEATURE_COLUMNS,
        model_path,
        metadata={
            "model_type": model_type,
            "use_gpu": use_gpu,
            "manual_edits_csv": manual_edits_csv_path,
            "manual_label_report": report,
        },
    )

    results["dataset_csv"] = merged_csv_path
    results["unmatched_csv"] = unmatched_csv_path
    results["manual_label_report"] = report
    results["model_path"] = model_path
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Retrain maneuver classifier from an existing labeled CSV and optional Tacview manual edits"
    )
    parser.add_argument("labeled_csv", help="Base maneuver_features_labeled.csv")
    parser.add_argument(
        "--manual-edits-csv",
        default=None,
        help="CSV exported by the Tacview manual label editor add-on",
    )
    parser.add_argument("--output-dir", default="results/retrain")
    parser.add_argument(
        "--match-tolerance-sec",
        type=float,
        default=DEFAULT_MATCH_TOLERANCE_SEC,
        help=f"Maximum allowed distance when matching an edit to a window (default: {DEFAULT_MATCH_TOLERANCE_SEC})",
    )
    parser.add_argument(
        "--model-type",
        choices=["random_forest", "xgboost"],
        default="random_forest",
    )
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    retrain_from_labeled_csv(
        labeled_csv_path=args.labeled_csv,
        manual_edits_csv_path=args.manual_edits_csv,
        output_dir=args.output_dir,
        match_tolerance_sec=args.match_tolerance_sec,
        model_type=args.model_type,
        use_gpu=args.use_gpu,
    )


if __name__ == "__main__":
    main()
