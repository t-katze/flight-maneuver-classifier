"""
Utilities for applying manual Tacview label edits to labeled maneuver datasets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from maneuver_labeler import CLASS_NAME_TO_ID, MANEUVER_CLASSES


DEFAULT_MATCH_TOLERANCE_SEC = 2.5


def _normalize_aircraft_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _normalize_manual_label_name(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return text


def load_manual_label_edits(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "edit_index",
                "edit_action",
                "aircraft_id",
                "aircraft_name",
                "sample_time",
                "window_start_hint",
                "window_end_hint",
                "manual_label_name",
                "manual_label_id",
            ]
        )

    work = df.copy()
    if "manual_label" in work.columns and "manual_label_name" not in work.columns:
        work["manual_label_name"] = work["manual_label"]
    if "label_name" in work.columns and "manual_label_name" not in work.columns:
        work["manual_label_name"] = work["label_name"]
    if "time" in work.columns and "sample_time" not in work.columns:
        work["sample_time"] = work["time"]

    required = {"aircraft_id", "sample_time", "manual_label_name"}
    missing = required - set(work.columns)
    if missing:
        raise ValueError(
            f"Manual label edits CSV is missing required columns: {sorted(missing)}"
        )

    work["edit_index"] = range(len(work))
    if "edit_action" not in work.columns:
        work["edit_action"] = "set"
    work["aircraft_id"] = work["aircraft_id"].map(_normalize_aircraft_id)
    if "aircraft_name" not in work.columns:
        work["aircraft_name"] = ""
    work["sample_time"] = pd.to_numeric(work["sample_time"], errors="coerce")
    if "window_start_hint" not in work.columns:
        work["window_start_hint"] = pd.NA
    if "window_end_hint" not in work.columns:
        work["window_end_hint"] = pd.NA
    work["manual_label_name"] = work["manual_label_name"].map(_normalize_manual_label_name)
    work["manual_label_id"] = work["manual_label_name"].map(
        lambda name: CLASS_NAME_TO_ID.get(name) if name else pd.NA
    )
    return work


def initialize_training_labels(labeled_df: pd.DataFrame) -> pd.DataFrame:
    result = labeled_df.copy()
    if "training_label" not in result.columns:
        result["training_label"] = result["label"]
    if "training_label_name" not in result.columns:
        result["training_label_name"] = result["label_name"]
    if "training_label_origin" not in result.columns:
        result["training_label_origin"] = "rule"
    if "manual_label_name" not in result.columns:
        result["manual_label_name"] = pd.NA
    if "manual_label_id" not in result.columns:
        result["manual_label_id"] = pd.NA
    if "manual_label_time" not in result.columns:
        result["manual_label_time"] = pd.NA
    if "manual_label_distance_sec" not in result.columns:
        result["manual_label_distance_sec"] = pd.NA
    if "manual_label_action" not in result.columns:
        result["manual_label_action"] = pd.NA
    if "manual_label_matched" not in result.columns:
        result["manual_label_matched"] = False
    return result


def apply_manual_label_edits(
    labeled_df: pd.DataFrame,
    edits_df: pd.DataFrame,
    match_tolerance_sec: float = DEFAULT_MATCH_TOLERANCE_SEC,
) -> tuple[pd.DataFrame, dict]:
    result = initialize_training_labels(labeled_df)
    if result.empty or edits_df.empty:
        return result, {
            "edit_count": int(len(edits_df)),
            "matched_count": 0,
            "unmatched_count": int(len(edits_df)),
            "manual_training_count": int(result["training_label_origin"].eq("manual").sum()),
            "unmatched_edits": [],
        }

    work_edits = edits_df.copy()
    result["aircraft_id"] = result["aircraft_id"].map(_normalize_aircraft_id)
    result["window_center_time"] = (
        pd.to_numeric(result["window_start"], errors="coerce")
        + pd.to_numeric(result["window_end"], errors="coerce")
    ) / 2.0

    unmatched = []
    matched_count = 0

    for edit in work_edits.sort_values("edit_index", kind="stable").itertuples(index=False):
        if pd.isna(edit.sample_time):
            unmatched.append(
                {
                    "edit_index": int(edit.edit_index),
                    "reason": "invalid_sample_time",
                    "aircraft_id": edit.aircraft_id,
                }
            )
            continue

        mask = result["aircraft_id"].eq(edit.aircraft_id)
        if "source_acmi" in result.columns and hasattr(edit, "source_acmi"):
            source_acmi = getattr(edit, "source_acmi")
            if source_acmi is not None and not pd.isna(source_acmi) and str(source_acmi).strip():
                mask &= result["source_acmi"].eq(source_acmi)

        candidates = result.loc[mask]
        if candidates.empty:
            unmatched.append(
                {
                    "edit_index": int(edit.edit_index),
                    "reason": "aircraft_not_found",
                    "aircraft_id": edit.aircraft_id,
                }
            )
            continue

        if not pd.isna(edit.window_start_hint) and not pd.isna(edit.window_end_hint):
            hinted = candidates.loc[
                (pd.to_numeric(candidates["window_start"], errors="coerce") - float(edit.window_start_hint)).abs() <= 1e-6
            ]
            hinted = hinted.loc[
                (pd.to_numeric(hinted["window_end"], errors="coerce") - float(edit.window_end_hint)).abs() <= 1e-6
            ]
            if not hinted.empty:
                best_index = hinted.index[-1]
                best_distance = abs(float(candidates.loc[best_index, "window_center_time"]) - float(edit.sample_time))
            else:
                distances = (candidates["window_center_time"] - float(edit.sample_time)).abs()
                best_index = distances.idxmin()
                best_distance = float(distances.loc[best_index])
        else:
            distances = (candidates["window_center_time"] - float(edit.sample_time)).abs()
            best_index = distances.idxmin()
            best_distance = float(distances.loc[best_index])
        if best_distance > match_tolerance_sec:
            unmatched.append(
                {
                    "edit_index": int(edit.edit_index),
                    "reason": "no_window_within_tolerance",
                    "aircraft_id": edit.aircraft_id,
                    "distance_sec": best_distance,
                }
            )
            continue

        matched_count += 1
        result.at[best_index, "manual_label_name"] = edit.manual_label_name
        result.at[best_index, "manual_label_id"] = edit.manual_label_id
        result.at[best_index, "manual_label_time"] = float(edit.sample_time)
        result.at[best_index, "manual_label_distance_sec"] = best_distance
        result.at[best_index, "manual_label_action"] = edit.edit_action
        result.at[best_index, "manual_label_matched"] = True

        if str(edit.edit_action).strip().lower() == "clear" or not edit.manual_label_name:
            result.at[best_index, "training_label"] = result.at[best_index, "label"]
            result.at[best_index, "training_label_name"] = result.at[best_index, "label_name"]
            result.at[best_index, "training_label_origin"] = "rule"
        else:
            label_id = CLASS_NAME_TO_ID.get(edit.manual_label_name)
            if label_id is None:
                unmatched.append(
                    {
                        "edit_index": int(edit.edit_index),
                        "reason": "unknown_manual_label",
                        "manual_label_name": edit.manual_label_name,
                    }
                )
                continue
            result.at[best_index, "training_label"] = int(label_id)
            result.at[best_index, "training_label_name"] = edit.manual_label_name
            result.at[best_index, "training_label_origin"] = "manual"

    report = {
        "edit_count": int(len(work_edits)),
        "matched_count": int(matched_count),
        "unmatched_count": int(len(unmatched)),
        "manual_training_count": int(result["training_label_origin"].eq("manual").sum()),
        "unmatched_edits": unmatched,
    }
    return result.drop(columns=["window_center_time"]), report


def load_and_apply_manual_label_edits(
    labeled_csv_path: str,
    manual_edits_csv_path: str | None,
    match_tolerance_sec: float = DEFAULT_MATCH_TOLERANCE_SEC,
) -> tuple[pd.DataFrame, dict]:
    labeled_df = pd.read_csv(labeled_csv_path)
    if not manual_edits_csv_path:
        result = initialize_training_labels(labeled_df)
        return result, {
            "edit_count": 0,
            "matched_count": 0,
            "unmatched_count": 0,
            "manual_training_count": 0,
            "unmatched_edits": [],
        }

    edits_df = load_manual_label_edits(manual_edits_csv_path)
    return apply_manual_label_edits(
        labeled_df=labeled_df,
        edits_df=edits_df,
        match_tolerance_sec=match_tolerance_sec,
    )


def save_manual_label_report(report: dict, output_path: str) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report["unmatched_edits"]).to_csv(out, index=False)
    return str(out)


def label_name_from_id(label_id: int) -> str:
    return MANEUVER_CLASSES[int(label_id)]
