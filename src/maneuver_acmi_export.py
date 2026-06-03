"""
Export maneuver classification results back into Tacview ACMI telemetry.

The generated ACMI keeps the original telemetry and injects custom object
properties such as ``ManeuverLabel`` and ``ManeuverLabelId``. Tacview preserves
unsupported properties and shows them in the Raw Telemetry window.
"""

from __future__ import annotations

import argparse
import bisect
import zipfile
from pathlib import Path

import pandas as pd

from maneuver_labeler import MANEUVER_CLASSES


DEFAULT_LABELS_CSV = "results/maneuver_features_labeled.csv"
DEFAULT_PROPERTY_PREFIX = "Maneuver"
TIME_ANCHORS = ("start", "center", "end")
REQUIRED_LABEL_COLUMNS = {"window_start", "window_end", "aircraft_id"}


def _escape_acmi_text(value: str) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace(",", "\\,")
    )


def _format_number(value: float) -> str:
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _normalize_object_id(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _snake_to_property_suffix(value: str) -> str:
    parts = [part for part in str(value).strip("_").split("_") if part]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _property_name(prefix: str, order: int, name: str) -> str:
    return f"{prefix}{order:03d}{name}"


def _is_truthy(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, (bool, int)):
        return bool(value)
    if isinstance(value, float):
        return value != 0.0
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y"}


def _format_boolish(value) -> str:
    return "1" if _is_truthy(value) else "0"


def _safe_getattr(row, name: str, default=None):
    return getattr(row, name, default)


def _append_text_property(properties: list[str], key: str, value):
    if value is None or pd.isna(value):
        return
    text = str(value).strip()
    if not text:
        return
    properties.append(f"{key}={_escape_acmi_text(text)}")


def _append_number_property(properties: list[str], key: str, value):
    if value is None or pd.isna(value):
        return
    properties.append(f"{key}={_format_number(value)}")


def read_acmi_text(path: str) -> tuple[str, str | None]:
    acmi_path = Path(path)
    if acmi_path.suffix == ".zip" or str(acmi_path).endswith(".zip.acmi"):
        with zipfile.ZipFile(acmi_path, "r") as archive:
            member_name = archive.namelist()[0]
            raw = archive.read(member_name)
        return raw.decode("utf-8-sig"), member_name

    return acmi_path.read_text(encoding="utf-8-sig"), None


def _default_zip_member_name(output_path: Path) -> str:
    name = output_path.name
    if name.endswith(".zip.acmi"):
        return f"{name[:-9]}.txt.acmi"
    if name.endswith(".zip"):
        return f"{name[:-4]}.txt.acmi"
    return f"{output_path.stem}.txt.acmi"


def write_acmi_text(path: str, text: str, member_name: str | None = None):
    output_path = Path(path)
    if output_path.suffix == ".zip" or str(output_path).endswith(".zip.acmi"):
        inner_name = member_name or _default_zip_member_name(output_path)
        with zipfile.ZipFile(
            output_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(inner_name, text.encode("utf-8-sig"))
        return

    output_path.write_text(text, encoding="utf-8-sig")


def extract_frame_markers(acmi_text: str) -> list[tuple[str, float]]:
    markers: list[tuple[str, float]] = []
    for raw_line in acmi_text.splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if not line.startswith("#"):
            continue
        time_text = line[1:].strip()
        try:
            time_value = float(time_text)
        except ValueError:
            continue
        markers.append((time_text, time_value))
    return markers


def extract_object_first_frames(acmi_text: str) -> dict[str, int]:
    first_frames: dict[str, int] = {}
    current_frame_index = -1

    for raw_line in acmi_text.splitlines():
        line = raw_line.lstrip("\ufeff").strip()
        if not line:
            continue
        if line.startswith("#"):
            current_frame_index += 1
            continue
        if (
            current_frame_index < 0
            or line.startswith(("FileType=", "FileVersion=", "0,", "-"))
            or "," not in line
        ):
            continue

        object_id = line.split(",", 1)[0].strip()
        if object_id and object_id not in first_frames:
            first_frames[object_id] = current_frame_index

    return first_frames


def load_labeled_windows(csv_path: str) -> pd.DataFrame:
    labels_df = pd.read_csv(csv_path)
    missing = REQUIRED_LABEL_COLUMNS - set(labels_df.columns)
    if missing:
        raise ValueError(
            f"Missing required label columns: {', '.join(sorted(missing))}"
        )

    labels_df = labels_df.copy()
    if "label_name" not in labels_df.columns:
        if "label" not in labels_df.columns:
            raise ValueError("Labels CSV must contain either 'label_name' or 'label'.")
        labels_df["label_name"] = labels_df["label"].map(MANEUVER_CLASSES).fillna("Unknown")

    if "rule_based_label" not in labels_df.columns and "label" in labels_df.columns:
        labels_df["rule_based_label"] = labels_df["label"]
    if "rule_based_label_name" not in labels_df.columns and "label_name" in labels_df.columns:
        labels_df["rule_based_label_name"] = labels_df["label_name"]

    if "predicted_label_name" not in labels_df.columns and "predicted_label" in labels_df.columns:
        labels_df["predicted_label_name"] = labels_df["predicted_label"].map(
            MANEUVER_CLASSES
        ).fillna("Unknown")

    labels_df["aircraft_id"] = labels_df["aircraft_id"].map(_normalize_object_id)
    return labels_df


def _resolve_anchor_time(window_start: float, window_end: float, anchor: str) -> float:
    if anchor == "start":
        return float(window_start)
    if anchor == "end":
        return float(window_end)
    return (float(window_start) + float(window_end)) / 2.0


def _nearest_frame_index(frame_times: list[float], target_time: float) -> int:
    if not frame_times:
        raise ValueError("No frame times available.")

    index = bisect.bisect_left(frame_times, target_time)
    if index <= 0:
        return 0
    if index >= len(frame_times):
        return len(frame_times) - 1

    previous_time = frame_times[index - 1]
    next_time = frame_times[index]
    if abs(target_time - previous_time) <= abs(next_time - target_time):
        return index - 1
    return index


def build_annotation_schedule(
    frame_markers: list[tuple[str, float]],
    labels_df: pd.DataFrame,
    time_anchor: str = "center",
    property_prefix: str = DEFAULT_PROPERTY_PREFIX,
    first_frame_by_object: dict[str, int] | None = None,
) -> dict[str, list[str]]:
    if time_anchor not in TIME_ANCHORS:
        raise ValueError(f"time_anchor must be one of {TIME_ANCHORS}")

    if not frame_markers:
        raise ValueError("ACMI does not contain any frame markers.")

    frame_texts = [time_text for time_text, _ in frame_markers]
    frame_times = [time_value for _, time_value in frame_markers]
    scheduled_updates: dict[str, dict[str, str]] = {}
    flag_columns = [c for c in labels_df.columns if c.startswith("flag_")]
    score_columns = [c for c in labels_df.columns if c.startswith("score_")]
    attr_columns = [c for c in labels_df.columns if c.startswith("attr_")]

    for row in labels_df.sort_values(
        by=["window_start", "window_end", "aircraft_id"],
        kind="stable",
    ).itertuples(index=False):
        aircraft_id = _normalize_object_id(row.aircraft_id)
        if not aircraft_id:
            continue
        if first_frame_by_object is not None and aircraft_id not in first_frame_by_object:
            continue

        anchor_time = _resolve_anchor_time(row.window_start, row.window_end, time_anchor)
        frame_index = _nearest_frame_index(frame_times, anchor_time)
        if first_frame_by_object is not None:
            frame_index = max(frame_index, first_frame_by_object[aircraft_id])
        frame_key = frame_texts[frame_index]

        primary_label_name = getattr(row, "predicted_label_name", None) or row.label_name
        primary_label = getattr(row, "predicted_label", None)
        if primary_label is None or pd.isna(primary_label):
            primary_label = getattr(row, "label", None)

        properties = [
            f"{_property_name(property_prefix, 10, 'PredLabel')}={_escape_acmi_text(primary_label_name)}",
            f"{_property_name(property_prefix, 70, 'WindowStart')}={_format_number(row.window_start)}",
            f"{_property_name(property_prefix, 71, 'WindowEnd')}={_format_number(row.window_end)}",
            f"{_property_name(property_prefix, 72, 'SampleTime')}={_format_number(anchor_time)}",
        ]

        if primary_label is not None and not pd.isna(primary_label):
            properties.insert(
                1,
                f"{_property_name(property_prefix, 11, 'PredLabelId')}={int(primary_label)}",
            )

        rule_based_label_name = getattr(row, "rule_based_label_name", None)
        if not rule_based_label_name:
            rule_based_label_name = getattr(row, "label_name", None)
        if rule_based_label_name:
            properties.append(
                f"{_property_name(property_prefix, 20, 'RuleLabel')}={_escape_acmi_text(rule_based_label_name)}"
            )

        rule_based_label = getattr(row, "rule_based_label", None)
        if rule_based_label is None or pd.isna(rule_based_label):
            rule_based_label = getattr(row, "label", None)
        if rule_based_label is not None and not pd.isna(rule_based_label):
            properties.append(
                f"{_property_name(property_prefix, 21, 'RuleLabelId')}={int(rule_based_label)}"
            )

        main_label_name = _safe_getattr(row, "main_label", None)
        if not main_label_name:
            main_label_name = _safe_getattr(row, "label_name", None)
        _append_text_property(
            properties,
            _property_name(property_prefix, 30, "MainLabel"),
            main_label_name,
        )

        main_label_id = _safe_getattr(row, "main_label_id", None)
        if main_label_id is None or pd.isna(main_label_id):
            main_label_id = _safe_getattr(row, "label", None)
        if main_label_id is not None and not pd.isna(main_label_id):
            properties.append(
                f"{_property_name(property_prefix, 31, 'MainLabelId')}={int(main_label_id)}"
            )

        _append_text_property(
            properties,
            _property_name(property_prefix, 40, "Attributes"),
            _safe_getattr(row, "attributes", None),
        )
        attribute_count = _safe_getattr(row, "attribute_count", None)
        if attribute_count is not None and not pd.isna(attribute_count):
            properties.append(
                f"{_property_name(property_prefix, 41, 'AttributeCount')}={int(attribute_count)}"
            )
        _append_text_property(
            properties,
            _property_name(property_prefix, 50, "UncertainReason"),
            _safe_getattr(row, "uncertain_reason", None),
        )
        _append_text_property(
            properties,
            _property_name(property_prefix, 51, "TransitionReason"),
            _safe_getattr(row, "transition_reason", None),
        )

        active_flags = []
        for column in flag_columns:
            value = _safe_getattr(row, column, None)
            suffix = _snake_to_property_suffix(column.removeprefix("flag_"))
            properties.append(
                f"{_property_name(property_prefix, 200, f'Flag{suffix}')}={_format_boolish(value)}"
            )
            if _is_truthy(value):
                active_flags.append(column.removeprefix("flag_"))
        if active_flags:
            properties.append(
                f"{_property_name(property_prefix, 190, 'ActiveFlags')}={_escape_acmi_text(';'.join(active_flags))}"
            )

        active_attributes = []
        for column in attr_columns:
            value = _safe_getattr(row, column, None)
            suffix = _snake_to_property_suffix(column.removeprefix("attr_"))
            properties.append(
                f"{_property_name(property_prefix, 300, f'Attr{suffix}')}={_format_boolish(value)}"
            )
            if _is_truthy(value):
                active_attributes.append(column.removeprefix("attr_"))
        if active_attributes:
            properties.append(
                f"{_property_name(property_prefix, 290, 'ActiveAttributes')}={_escape_acmi_text(';'.join(active_attributes))}"
            )

        score_pairs: list[tuple[str, float]] = []
        for column in score_columns:
            raw_value = _safe_getattr(row, column, None)
            if raw_value is None or pd.isna(raw_value):
                continue
            suffix = _snake_to_property_suffix(column.removeprefix("score_"))
            properties.append(
                f"{_property_name(property_prefix, 400, f'Score{suffix}')}={_format_number(raw_value)}"
            )
            score_pairs.append((column.removeprefix("score_"), float(raw_value)))

        if score_pairs:
            ranked_scores = sorted(score_pairs, key=lambda item: item[1], reverse=True)
            top_label, top_score = ranked_scores[0]
            properties.append(
                f"{_property_name(property_prefix, 410, 'TopScoreLabel')}={_escape_acmi_text(top_label)}"
            )
            properties.append(
                f"{_property_name(property_prefix, 411, 'TopScore')}={_format_number(top_score)}"
            )
            if len(ranked_scores) >= 2:
                second_label, second_score = ranked_scores[1]
                properties.append(
                    f"{_property_name(property_prefix, 412, 'SecondScoreLabel')}={_escape_acmi_text(second_label)}"
                )
                properties.append(
                    f"{_property_name(property_prefix, 413, 'SecondScore')}={_format_number(second_score)}"
                )

        for numeric_column in (
            "g_mean",
            "g_std",
            "turn_rate_mean",
            "turn_rate_peak",
            "roll_rate_peak",
            "vertical_speed_mean",
            "heading_delta",
            "speed_slope",
        ):
            value = _safe_getattr(row, numeric_column, None)
            if value is None or pd.isna(value):
                continue
            suffix = _snake_to_property_suffix(numeric_column)
            properties.append(
                f"{_property_name(property_prefix, 500, suffix)}={_format_number(value)}"
            )

        g_mean_value = _safe_getattr(row, "g_mean", None)
        if g_mean_value is not None and not pd.isna(g_mean_value):
            properties.append(
                f"{_property_name(property_prefix, 500, 'ComputedGMean')}={_format_number(g_mean_value)}"
            )
        g_std_value = _safe_getattr(row, "g_std", None)
        if g_std_value is not None and not pd.isna(g_std_value):
            properties.append(
                f"{_property_name(property_prefix, 500, 'ComputedGStd')}={_format_number(g_std_value)}"
            )
        properties.append(
            f"{_property_name(property_prefix, 52, 'ComputedGSource')}="
            f"{_escape_acmi_text('trajectory_curvature_estimate')}"
        )

        scheduled_updates.setdefault(frame_key, {})[aircraft_id] = (
            f"{aircraft_id}," + ",".join(properties)
        )

    return {
        frame_key: [updates[obj_id] for obj_id in sorted(updates)]
        for frame_key, updates in scheduled_updates.items()
    }


def inject_annotation_lines(
    acmi_text: str,
    schedule: dict[str, list[str]],
) -> str:
    output_lines: list[str] = []
    for raw_line in acmi_text.splitlines(keepends=True):
        output_lines.append(raw_line)

        line = raw_line.lstrip("\ufeff").strip()
        if not line.startswith("#"):
            continue

        frame_key = line[1:].strip()
        updates = schedule.get(frame_key)
        if not updates:
            continue

        for update in updates:
            output_lines.append(f"{update}\n")

    return "".join(output_lines)


def annotate_acmi_file(
    acmi_path: str,
    labels_csv_path: str,
    output_path: str,
    time_anchor: str = "center",
    property_prefix: str = DEFAULT_PROPERTY_PREFIX,
) -> dict[str, int | str]:
    acmi_text, member_name = read_acmi_text(acmi_path)
    frame_markers = extract_frame_markers(acmi_text)
    first_frame_by_object = extract_object_first_frames(acmi_text)
    labels_df = load_labeled_windows(labels_csv_path)
    schedule = build_annotation_schedule(
        frame_markers=frame_markers,
        labels_df=labels_df,
        time_anchor=time_anchor,
        property_prefix=property_prefix,
        first_frame_by_object=first_frame_by_object,
    )
    annotated_text = inject_annotation_lines(acmi_text, schedule)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_acmi_text(str(output), annotated_text, member_name=member_name)

    return {
        "output_path": str(output),
        "frame_count": len(frame_markers),
        "annotation_frames": len(schedule),
        "annotation_rows": int(sum(len(rows) for rows in schedule.values())),
    }


def _default_output_path(acmi_path: str) -> str:
    path = Path(acmi_path)
    name = path.name
    if name.endswith(".zip.acmi"):
        return str(path.with_name(f"{name[:-9]}.maneuver.zip.acmi"))
    if name.endswith(".acmi"):
        return str(path.with_name(f"{path.stem}.maneuver.acmi"))
    return str(path.with_name(f"{name}.maneuver.acmi"))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export labeled maneuver windows back into Tacview ACMI as custom "
            "Raw Telemetry properties."
        )
    )
    parser.add_argument("acmi_file", help="Source Tacview .acmi or .zip.acmi file")
    parser.add_argument(
        "--labels-csv",
        default=DEFAULT_LABELS_CSV,
        help=f"Labeled maneuver CSV (default: {DEFAULT_LABELS_CSV})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output annotated .acmi / .zip.acmi path",
    )
    parser.add_argument(
        "--time-anchor",
        choices=TIME_ANCHORS,
        default="center",
        help="How to align a maneuver window to an ACMI frame (default: center)",
    )
    parser.add_argument(
        "--property-prefix",
        default=DEFAULT_PROPERTY_PREFIX,
        help=f"Custom telemetry property prefix (default: {DEFAULT_PROPERTY_PREFIX})",
    )

    args = parser.parse_args()
    output_path = args.output or _default_output_path(args.acmi_file)

    result = annotate_acmi_file(
        acmi_path=args.acmi_file,
        labels_csv_path=args.labels_csv,
        output_path=output_path,
        time_anchor=args.time_anchor,
        property_prefix=args.property_prefix,
    )

    print(f"[ManeuverACMIExport] Source ACMI: {args.acmi_file}")
    print(f"[ManeuverACMIExport] Labels CSV: {args.labels_csv}")
    print(f"[ManeuverACMIExport] Output ACMI: {result['output_path']}")
    print(
        "[ManeuverACMIExport] "
        f"Frames={result['frame_count']} "
        f"AnnotatedFrames={result['annotation_frames']} "
        f"AnnotationRows={result['annotation_rows']}"
    )


if __name__ == "__main__":
    main()
