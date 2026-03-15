"""
Rule-based window labeler for single-aircraft maneuver states.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MANEUVER_CLASSES = {
    0: "Straight_Level",
    1: "Straight_Climb",
    2: "Straight_Descent",
    3: "Level_Turn",
    4: "Climbing_Turn",
    5: "Descending_Turn",
    6: "Dive",
    7: "Zoom_Climb",
    8: "Large_Heading_Change",
    9: "Oscillatory_Maneuver",
    10: "Transition",
    11: "Uncertain",
}

CLASS_NAME_TO_ID = {name: idx for idx, name in MANEUVER_CLASSES.items()}
CLASS_NAMES = list(MANEUVER_CLASSES.values())
NUM_CLASSES = len(MANEUVER_CLASSES)

ATTRIBUTE_LABELS = [
    "High_G",
    "Unloaded",
    "Accelerating",
    "Decelerating",
    "High_Roll_Rate",
    "High_Turn_Rate",
    "Rapid_Altitude_Change",
    "Heading_Reversal_Like",
    "Jinking_Like",
]

CONCRETE_MAIN_LABELS = [
    "Straight_Level",
    "Straight_Climb",
    "Straight_Descent",
    "Level_Turn",
    "Climbing_Turn",
    "Descending_Turn",
    "Dive",
    "Zoom_Climb",
    "Large_Heading_Change",
    "Oscillatory_Maneuver",
]

PRIORITY_ORDER = [
    "Uncertain",
    "Transition",
    "Oscillatory_Maneuver",
    "Large_Heading_Change",
    "Dive",
    "Zoom_Climb",
    "Climbing_Turn",
    "Descending_Turn",
    "Level_Turn",
    "Straight_Climb",
    "Straight_Descent",
    "Straight_Level",
]

DEFAULT_THRESHOLDS = {
    "turn_small": 3.0,
    "turn_large": 10.0,
    "turn_high": 18.0,
    "vs_small": 5.0,
    "vs_up": 10.0,
    "vs_down": 10.0,
    "vs_rapid": 30.0,
    "vs_rapid_up": 30.0,
    "pitch_small": 5.0,
    "pitch_dive": 15.0,
    "pitch_zoom": 15.0,
    "g_unloaded_upper": 0.7,
    "g_high": 5.5,
    "g_pull": 2.0,
    "g_std_high": 0.8,
    "roll_sign_changes": 2,
    "turn_sign_changes": 2,
    "roll_rate_high": 60.0,
    "heading_large": 110.0,
    "acc": 1.0,
    "dec": 1.0,
    "uncertain_missing_ratio": 0.35,
    "uncertain_long_gap_ratio": 0.25,
    "transition_score_gap": 0.35,
    "large_heading_dominance_margin": 0.25,
}

_THRESHOLD_ARGUMENTS = [
    ("turn_small", "--turn-small-th", float, "直進判定 turn_rate_mean 上限"),
    ("turn_large", "--turn-large-th", float, "旋回判定 turn_rate_mean 下限"),
    ("turn_high", "--turn-high-th", float, "High_Turn_Rate 属性閾値"),
    ("vs_small", "--vs-small-th", float, "Level 判定 vertical_speed 許容幅"),
    ("vs_up", "--vs-up-th", float, "上昇判定 vertical_speed 閾値"),
    ("vs_down", "--vs-down-th", float, "降下判定 vertical_speed 閾値"),
    ("vs_rapid", "--vs-rapid-th", float, "急降下判定 vertical_speed 閾値"),
    ("vs_rapid_up", "--vs-rapid-up-th", float, "急上昇判定 vertical_speed 閾値"),
    ("pitch_small", "--pitch-small-th", float, "Straight_Level 判定 pitch 許容幅"),
    ("pitch_dive", "--pitch-dive-th", float, "Dive 判定 pitch 閾値"),
    ("pitch_zoom", "--pitch-zoom-th", float, "Zoom_Climb 判定 pitch 閾値"),
    ("g_unloaded_upper", "--g-unloaded-th", float, "Unloaded 属性閾値"),
    ("g_high", "--g-high-th", float, "High_G 属性閾値"),
    ("g_pull", "--g-pull-th", float, "Zoom 補助閾値"),
    ("g_std_high", "--g-std-high-th", float, "Oscillatory 判定 g_std 閾値"),
    ("roll_sign_changes", "--roll-sign-changes-th", int, "Oscillatory/Jinking sign change 閾値"),
    ("turn_sign_changes", "--turn-sign-changes-th", int, "Oscillatory/Jinking sign change 閾値"),
    ("roll_rate_high", "--roll-rate-high-th", float, "High_Roll_Rate / Oscillatory 判定 roll_rate_peak 閾値"),
    ("heading_large", "--heading-large-th", float, "Large_Heading_Change 判定 heading_delta 閾値"),
    ("acc", "--acc-th", float, "Accelerating 属性 speed_slope 閾値"),
    ("dec", "--dec-th", float, "Decelerating 属性 speed_slope 閾値"),
    ("uncertain_missing_ratio", "--uncertain-missing-ratio-th", float, "Uncertain 判定 missing_ratio 閾値"),
    ("uncertain_long_gap_ratio", "--uncertain-long-gap-ratio-th", float, "Uncertain 判定 long_gap_ratio 閾値"),
    ("transition_score_gap", "--transition-score-gap-th", float, "Transition 判定 1位2位スコア差閾値"),
]


def add_threshold_arguments(parser) -> None:
    group = parser.add_argument_group("ラベラー閾値")
    for key, flag, arg_type, description in _THRESHOLD_ARGUMENTS:
        group.add_argument(
            flag,
            dest=f"threshold_{key}",
            type=arg_type,
            default=DEFAULT_THRESHOLDS[key],
            help=f"{description} (default: {DEFAULT_THRESHOLDS[key]})",
        )


def thresholds_from_args(args) -> dict:
    thresholds = {}
    for key, _, _, _ in _THRESHOLD_ARGUMENTS:
        thresholds[key] = getattr(args, f"threshold_{key}")
    return thresholds


def _flag_column(label_name: str) -> str:
    return f"flag_{label_name.lower()}"


def _score_column(label_name: str) -> str:
    return f"score_{label_name.lower()}"


def _attribute_column(label_name: str) -> str:
    return f"attr_{label_name.lower()}"


def _clip_score(value: float) -> float:
    return float(max(0.0, value))


class ManeuverLabeler:
    """Attach one exclusive main label and multiple non-exclusive attribute labels."""

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = {**DEFAULT_THRESHOLDS}
        if thresholds:
            self.thresholds.update(thresholds)

    @staticmethod
    def _oscillatory_inputs(row: dict | pd.Series) -> dict[str, float]:
        return {
            "turn_sign_changes": max(
                float(row.get("turn_sign_changes", 0.0)),
                float(row.get("short_turn_sign_changes_max", 0.0)),
            ),
            "roll_sign_changes": max(
                float(row.get("roll_sign_changes", 0.0)),
                float(row.get("short_roll_sign_changes_max", 0.0)),
            ),
            "pitch_sign_changes": max(
                float(row.get("pitch_sign_changes", 0.0)),
                float(row.get("short_pitch_sign_changes_max", 0.0)),
            ),
            "turn_rate_peak": max(
                abs(float(row.get("turn_rate_peak", 0.0))),
                abs(float(row.get("short_turn_rate_peak_max", 0.0))),
            ),
            "roll_rate_peak": max(
                abs(float(row.get("roll_rate_peak", 0.0))),
                abs(float(row.get("short_roll_rate_peak_max", 0.0))),
            ),
            "g_std": max(
                float(row.get("g_std", 0.0)),
                float(row.get("short_g_std_max", 0.0)),
            ),
        }

    def _compute_scores(self, row: dict | pd.Series) -> dict[str, float]:
        th = self.thresholds
        turn_rate_mean = abs(float(row.get("turn_rate_mean", 0.0)))
        vertical_speed_mean = float(row.get("vertical_speed_mean", 0.0))
        pitch_mean = float(row.get("pitch_mean", 0.0))
        heading_delta = abs(float(row.get("heading_delta", 0.0)))
        osc = self._oscillatory_inputs(row)
        roll_sign_changes = osc["roll_sign_changes"]
        turn_sign_changes = osc["turn_sign_changes"]
        roll_rate_peak = osc["roll_rate_peak"]
        g_std = osc["g_std"]
        g_mean = float(row.get("g_mean", 0.0))

        scores = {
            "Straight_Level": _clip_score(
                (1.0 - turn_rate_mean / th["turn_small"])
                + (1.0 - abs(vertical_speed_mean) / th["vs_small"])
                + (1.0 - abs(pitch_mean) / th["pitch_small"])
            ),
            "Straight_Climb": _clip_score(
                (1.0 - turn_rate_mean / th["turn_small"])
                + (vertical_speed_mean / th["vs_up"])
            ),
            "Straight_Descent": _clip_score(
                (1.0 - turn_rate_mean / th["turn_small"])
                + (-vertical_speed_mean / th["vs_down"])
            ),
            "Level_Turn": _clip_score(
                (turn_rate_mean / th["turn_large"])
                + (1.0 - abs(vertical_speed_mean) / th["vs_small"])
            ),
            "Climbing_Turn": _clip_score(
                (turn_rate_mean / th["turn_large"])
                + (vertical_speed_mean / th["vs_up"])
            ),
            "Descending_Turn": _clip_score(
                (turn_rate_mean / th["turn_large"])
                + (-vertical_speed_mean / th["vs_down"])
            ),
            "Dive": _clip_score(
                (-pitch_mean / th["pitch_dive"])
                + (-vertical_speed_mean / th["vs_rapid"])
                + max(0.0, 1.0 - g_mean)
            ),
            "Zoom_Climb": _clip_score(
                (pitch_mean / th["pitch_zoom"])
                + (vertical_speed_mean / th["vs_rapid_up"])
                + max(0.0, g_mean - 1.0)
            ),
            "Large_Heading_Change": _clip_score(heading_delta / th["heading_large"]),
            "Oscillatory_Maneuver": _clip_score(
                max(
                    roll_sign_changes / max(1, th["roll_sign_changes"]),
                    turn_sign_changes / max(1, th["turn_sign_changes"]),
                )
                + max(
                    roll_rate_peak / th["roll_rate_high"],
                    g_std / max(th["g_std_high"], 1e-6),
                )
            ),
        }
        return scores

    def _main_flags(self, row: dict | pd.Series) -> dict[str, bool]:
        th = self.thresholds
        turn_rate_mean = abs(float(row.get("turn_rate_mean", 0.0)))
        vertical_speed_mean = float(row.get("vertical_speed_mean", 0.0))
        pitch_mean = float(row.get("pitch_mean", 0.0))
        heading_delta = abs(float(row.get("heading_delta", 0.0)))
        osc = self._oscillatory_inputs(row)
        g_std = osc["g_std"]
        roll_sign_changes = osc["roll_sign_changes"]
        turn_sign_changes = osc["turn_sign_changes"]
        roll_rate_peak = osc["roll_rate_peak"]

        flags = {
            "Straight_Level": (
                turn_rate_mean < th["turn_small"]
                and abs(vertical_speed_mean) < th["vs_small"]
                and abs(pitch_mean) < th["pitch_small"]
            ),
            "Straight_Climb": (
                turn_rate_mean < th["turn_small"]
                and vertical_speed_mean > th["vs_up"]
            ),
            "Straight_Descent": (
                turn_rate_mean < th["turn_small"]
                and vertical_speed_mean < -th["vs_down"]
            ),
            "Level_Turn": (
                turn_rate_mean >= th["turn_large"]
                and abs(vertical_speed_mean) < th["vs_small"]
            ),
            "Climbing_Turn": (
                turn_rate_mean >= th["turn_large"]
                and vertical_speed_mean > th["vs_up"]
            ),
            "Descending_Turn": (
                turn_rate_mean >= th["turn_large"]
                and vertical_speed_mean < -th["vs_down"]
            ),
            "Dive": (
                pitch_mean < -th["pitch_dive"]
                and vertical_speed_mean < -th["vs_rapid"]
            ),
            "Zoom_Climb": (
                pitch_mean > th["pitch_zoom"]
                and vertical_speed_mean > th["vs_rapid_up"]
            ),
            "Large_Heading_Change": heading_delta >= th["heading_large"],
            "Oscillatory_Maneuver": (
                (
                    roll_sign_changes >= th["roll_sign_changes"]
                    or turn_sign_changes >= th["turn_sign_changes"]
                )
                and (
                    roll_rate_peak >= th["roll_rate_high"]
                    or g_std >= th["g_std_high"]
                )
            ),
        }
        return flags

    def _compute_attributes(self, row: dict | pd.Series) -> dict[str, bool]:
        th = self.thresholds
        g_mean = float(row.get("g_mean", 0.0))
        speed_slope = float(row.get("speed_slope", 0.0))
        vertical_speed_mean = float(row.get("vertical_speed_mean", 0.0))
        heading_delta = abs(float(row.get("heading_delta", 0.0)))
        osc = self._oscillatory_inputs(row)
        roll_rate_peak = osc["roll_rate_peak"]
        turn_rate_peak = osc["turn_rate_peak"]
        roll_sign_changes = osc["roll_sign_changes"]
        turn_sign_changes = osc["turn_sign_changes"]

        return {
            "High_G": g_mean >= th["g_high"],
            "Unloaded": g_mean <= th["g_unloaded_upper"],
            "Accelerating": speed_slope > th["acc"],
            "Decelerating": speed_slope < -th["dec"],
            "High_Roll_Rate": roll_rate_peak >= th["roll_rate_high"],
            "High_Turn_Rate": turn_rate_peak >= th["turn_high"],
            "Rapid_Altitude_Change": abs(vertical_speed_mean) >= th["vs_rapid"],
            "Heading_Reversal_Like": heading_delta >= th["heading_large"],
            "Jinking_Like": (
                roll_sign_changes >= th["roll_sign_changes"]
                or turn_sign_changes >= th["turn_sign_changes"]
            ),
        }

    def _coarse_half_label(
        self,
        turn_rate_mean: float,
        vertical_speed_mean: float,
        pitch_mean: float,
        heading_delta: float,
    ) -> str | None:
        th = self.thresholds
        turn_rate_mean = abs(turn_rate_mean)
        heading_delta = abs(heading_delta)

        if pitch_mean < -th["pitch_dive"] and vertical_speed_mean < -th["vs_rapid"]:
            return "Dive"
        if pitch_mean > th["pitch_zoom"] and vertical_speed_mean > th["vs_rapid_up"]:
            return "Zoom_Climb"
        if turn_rate_mean >= th["turn_large"] or heading_delta >= th["heading_large"] / 2.0:
            if vertical_speed_mean > th["vs_up"]:
                return "Climbing_Turn"
            if vertical_speed_mean < -th["vs_down"]:
                return "Descending_Turn"
            if abs(vertical_speed_mean) < th["vs_small"]:
                return "Level_Turn"
            return None
        if turn_rate_mean < th["turn_small"]:
            if vertical_speed_mean > th["vs_up"]:
                return "Straight_Climb"
            if vertical_speed_mean < -th["vs_down"]:
                return "Straight_Descent"
            if abs(vertical_speed_mean) < th["vs_small"] and abs(pitch_mean) < th["pitch_small"]:
                return "Straight_Level"
        return None

    def _is_uncertain(self, row: dict | pd.Series) -> tuple[bool, str]:
        reasons = []
        if float(row.get("missing_ratio", 0.0)) >= self.thresholds["uncertain_missing_ratio"]:
            reasons.append("missing_ratio")
        if float(row.get("long_gap_ratio", 0.0)) >= self.thresholds["uncertain_long_gap_ratio"]:
            reasons.append("long_gap_ratio")
        if bool(row.get("sensor_conflict", 0)):
            reasons.append("sensor_conflict")
        if bool(row.get("feature_out_of_range", 0)):
            reasons.append("feature_out_of_range")

        numeric_keys = [
            "turn_rate_mean",
            "vertical_speed_mean",
            "pitch_mean",
            "heading_delta",
            "speed_slope",
            "g_mean",
        ]
        if any(not np.isfinite(float(row.get(key, 0.0))) for key in numeric_keys):
            reasons.append("non_finite_feature")

        return bool(reasons), ";".join(reasons)

    def _is_transition(
        self,
        row: dict | pd.Series,
        flags: dict[str, bool],
        scores: dict[str, float],
    ) -> tuple[bool, str]:
        active = [label for label in CONCRETE_MAIN_LABELS if flags.get(label)]
        if not active:
            return True, "no_rule_matched"

        sorted_active = sorted(active, key=lambda name: scores.get(name, 0.0), reverse=True)
        if len(sorted_active) >= 2:
            gap = scores[sorted_active[0]] - scores[sorted_active[1]]
            if gap < self.thresholds["transition_score_gap"]:
                return True, "multiple_flags_no_dominant"

        first_label = self._coarse_half_label(
            turn_rate_mean=float(row.get("first_half_turn_rate_mean", 0.0)),
            vertical_speed_mean=float(row.get("first_half_vertical_speed_mean", 0.0)),
            pitch_mean=float(row.get("first_half_pitch_mean", 0.0)),
            heading_delta=float(row.get("first_half_heading_delta", 0.0)),
        )
        second_label = self._coarse_half_label(
            turn_rate_mean=float(row.get("second_half_turn_rate_mean", 0.0)),
            vertical_speed_mean=float(row.get("second_half_vertical_speed_mean", 0.0)),
            pitch_mean=float(row.get("second_half_pitch_mean", 0.0)),
            heading_delta=float(row.get("second_half_heading_delta", 0.0)),
        )
        if first_label and second_label and first_label != second_label:
            return True, f"window_halves_disagree:{first_label}->{second_label}"

        return False, ""

    def _evaluate_row(self, row: dict | pd.Series) -> dict:
        uncertain, uncertain_reason = self._is_uncertain(row)
        flags = self._main_flags(row)
        scores = self._compute_scores(row)

        if flags["Large_Heading_Change"]:
            if flags["Oscillatory_Maneuver"]:
                flags["Large_Heading_Change"] = False
            else:
                max_dive_zoom = max(scores["Dive"], scores["Zoom_Climb"])
                if max_dive_zoom > scores["Large_Heading_Change"] + self.thresholds["large_heading_dominance_margin"]:
                    flags["Large_Heading_Change"] = False

        transition, transition_reason = (False, "")
        if not uncertain:
            transition, transition_reason = self._is_transition(row, flags, scores)

        if uncertain:
            main_label = "Uncertain"
        elif transition:
            main_label = "Transition"
        else:
            main_label = "Transition"
            for candidate in PRIORITY_ORDER[2:]:
                if flags.get(candidate):
                    main_label = candidate
                    break

        attributes = self._compute_attributes(row)
        active_attributes = [name for name in ATTRIBUTE_LABELS if attributes[name]]

        result = {
            "label": CLASS_NAME_TO_ID[main_label],
            "main_label_id": CLASS_NAME_TO_ID[main_label],
            "label_name": main_label,
            "main_label": main_label,
            "attributes": ";".join(active_attributes),
            "attribute_count": len(active_attributes),
            "uncertain_reason": uncertain_reason,
            "transition_reason": transition_reason,
        }
        for label_name in CONCRETE_MAIN_LABELS:
            result[_flag_column(label_name)] = bool(flags[label_name])
            result[_score_column(label_name)] = float(scores[label_name])
        for attr_name in ATTRIBUTE_LABELS:
            result[_attribute_column(attr_name)] = bool(attributes[attr_name])
        return result

    def label_single(self, row: dict | pd.Series) -> int:
        """Return the numeric main-label ID for one feature window."""
        return int(self._evaluate_row(row)["label"])

    def label_dataframe(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a copy of features_df with main label, attributes, flags, and scores.
        """
        if features_df.empty:
            df = features_df.copy()
            df["label"] = pd.Series(dtype=int)
            df["main_label_id"] = pd.Series(dtype=int)
            df["label_name"] = pd.Series(dtype=str)
            df["main_label"] = pd.Series(dtype=str)
            df["attributes"] = pd.Series(dtype=str)
            df["attribute_count"] = pd.Series(dtype=int)
            df["uncertain_reason"] = pd.Series(dtype=str)
            df["transition_reason"] = pd.Series(dtype=str)
            for label_name in CONCRETE_MAIN_LABELS:
                df[_flag_column(label_name)] = pd.Series(dtype=bool)
                df[_score_column(label_name)] = pd.Series(dtype=float)
            for attr_name in ATTRIBUTE_LABELS:
                df[_attribute_column(attr_name)] = pd.Series(dtype=bool)
            return df

        df = features_df.copy()
        evaluated = pd.DataFrame([self._evaluate_row(row) for _, row in df.iterrows()])
        return pd.concat([df.reset_index(drop=True), evaluated], axis=1)

    def print_distribution(self, df: pd.DataFrame):
        """Print the main label distribution."""
        if "label" not in df.columns:
            print("[ManeuverLabeler] ラベル列がありません")
            return

        total = len(df)
        print(f"\n[ManeuverLabeler] === ラベル分布 (N={total}) ===")
        for cls_id, cls_name in MANEUVER_CLASSES.items():
            count = int((df["label"] == cls_id).sum())
            pct = count / total * 100 if total > 0 else 0.0
            print(f"  {cls_id:2d}: {cls_name:24s} {count:6d}  ({pct:5.1f}%)")
