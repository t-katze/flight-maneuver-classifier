"""
Shared pipeline helpers for maneuver training and prediction workflows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from acmi_parser import ACMIParser
from maneuver_feature_engine import FEATURE_COLUMNS, build_feature_matrix, get_all_aircraft_ids
from maneuver_labeler import MANEUVER_CLASSES, ManeuverLabeler


def expand_acmi_inputs(paths: list[str], recursive: bool = False) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()

    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            iterator = path.rglob("*.acmi") if recursive else path.glob("*.acmi")
            matches = sorted(str(match) for match in iterator if match.is_file())
            if not matches:
                raise FileNotFoundError(f"No ACMI files found in directory: {path}")
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    resolved.append(match)
            continue

        if not path.exists():
            raise FileNotFoundError(f"ACMI path not found: {path}")

        normalized = str(path)
        if normalized not in seen:
            seen.add(normalized)
            resolved.append(normalized)

    return resolved


def acmi_output_stem(acmi_path: str) -> str:
    path = Path(acmi_path)
    name = path.name
    if name.endswith(".zip.acmi"):
        return name[:-9]
    if name.endswith(".acmi"):
        return path.stem
    return name


def default_annotated_acmi_path(acmi_path: str, output_dir: str) -> str:
    source = Path(acmi_path)
    output = Path(output_dir)
    name = source.name

    if name.endswith(".zip.acmi"):
        return str(output / f"{name[:-9]}.maneuver.zip.acmi")
    if name.endswith(".acmi"):
        return str(output / f"{source.stem}.maneuver.acmi")
    return str(output / f"{name}.maneuver.acmi")


def save_labeled_dataframe(df: pd.DataFrame, path: str) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return str(output_path)


def extract_labeled_features_from_acmi(
    acmi_path: str,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    aircraft_filter: str | None = None,
    thresholds: dict | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    if verbose:
        print(f"[ManeuverPipeline] Parsing: {acmi_path}")

    parser = ACMIParser()
    frames, _ = parser.parse_file(acmi_path)
    aircraft = get_all_aircraft_ids(frames)

    if aircraft_filter:
        aircraft = {oid: name for oid, name in aircraft.items() if oid == aircraft_filter}

    meta = {
        "acmi_path": acmi_path,
        "title": parser.title,
        "frame_count": len(frames),
        "aircraft_count": len(aircraft),
    }

    if verbose:
        print(
            "[ManeuverPipeline] "
            f"Title={parser.title or '-'} Frames={len(frames)} Aircraft={len(aircraft)}"
        )

    if not aircraft:
        return pd.DataFrame(), meta

    all_features = []
    for oid, name in aircraft.items():
        if verbose:
            print(f"[ManeuverPipeline] 特徴量抽出: {name} (ID={oid})")

        feat_df = build_feature_matrix(frames, oid, window_sec, step_sec)
        if feat_df.empty:
            continue

        feat_df["aircraft_name"] = name
        feat_df["source_acmi"] = str(Path(acmi_path))
        feat_df["source_title"] = parser.title
        all_features.append(feat_df)

    if not all_features:
        return pd.DataFrame(), meta

    features_df = pd.concat(all_features, ignore_index=True)
    labeler = ManeuverLabeler(thresholds=thresholds)
    labeled_df = labeler.label_dataframe(features_df)

    if verbose:
        print(
            "[ManeuverPipeline] "
            f"Extracted {len(labeled_df)} labeled windows from {acmi_path}"
        )

    return labeled_df, meta


def build_combined_labeled_dataset(
    acmi_paths: list[str],
    window_sec: float = 5.0,
    step_sec: float = 1.0,
    aircraft_filter: str | None = None,
    thresholds: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    datasets = []
    for acmi_path in acmi_paths:
        labeled_df, _ = extract_labeled_features_from_acmi(
            acmi_path=acmi_path,
            window_sec=window_sec,
            step_sec=step_sec,
            aircraft_filter=aircraft_filter,
            thresholds=thresholds,
            verbose=verbose,
        )
        if not labeled_df.empty:
            datasets.append(labeled_df)

    if not datasets:
        return pd.DataFrame()

    return pd.concat(datasets, ignore_index=True)


def prepare_training_dataframe(labeled_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if labeled_df.empty:
        return labeled_df.copy(), {
            "reason": "empty",
            "removed_names": [],
            "valid_labels": [],
        }

    unique_labels = labeled_df["label"].nunique()
    if unique_labels < 2:
        return labeled_df.iloc[0:0].copy(), {
            "reason": "too_few_unique_labels",
            "removed_names": [],
            "valid_labels": sorted(labeled_df["label"].unique().tolist()),
        }

    label_counts = labeled_df["label"].value_counts()
    valid_labels = label_counts[label_counts >= 2].index.tolist()
    if len(valid_labels) < 2:
        return labeled_df.iloc[0:0].copy(), {
            "reason": "too_few_valid_labels",
            "removed_names": [],
            "valid_labels": valid_labels,
        }

    removed = set(labeled_df["label"].unique()) - set(valid_labels)
    removed_names = [MANEUVER_CLASSES.get(label, str(label)) for label in sorted(removed)]

    training_df = labeled_df[labeled_df["label"].isin(valid_labels)].copy()
    return training_df, {
        "reason": None,
        "removed_names": removed_names,
        "valid_labels": valid_labels,
    }


def add_model_predictions(
    labeled_df: pd.DataFrame,
    model,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    if labeled_df.empty:
        result = labeled_df.copy()
        result["predicted_label"] = pd.Series(dtype=int)
        result["predicted_label_name"] = pd.Series(dtype=str)
        return result

    feature_columns = feature_columns or FEATURE_COLUMNS
    result = labeled_df.copy()
    X = result[feature_columns].values.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    result["predicted_label"] = model.predict(X).astype(int)
    result["predicted_label_name"] = result["predicted_label"].map(MANEUVER_CLASSES)
    return result
