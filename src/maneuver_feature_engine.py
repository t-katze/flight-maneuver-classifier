"""
Flight Maneuver Classifier — windowed feature extraction for single-aircraft kinematics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


ANALYSIS_AIRCRAFT_TYPE = "air+fixedwing"

REQUIRED_SIGNALS = ["altitude", "speed", "heading", "pitch", "roll", "g_load"]
WINDOW_COLUMNS = [
    "window_start",
    "window_end",
    "window_start_time",
    "window_end_time",
    "window_duration",
    "sample_count",
]
SHORT_WINDOW_SEC = 2.0

FEATURE_COLUMNS: list[str] = [
    "speed_mean",
    "speed_delta",
    "speed_slope",
    "alt_mean",
    "alt_delta",
    "alt_slope",
    "pitch_mean",
    "roll_mean",
    "g_mean",
    "g_std",
    "turn_rate_mean",
    "turn_rate_peak",
    "roll_rate_mean",
    "roll_rate_peak",
    "pitch_rate_mean",
    "vertical_speed_mean",
    "vertical_speed_peak",
    "heading_delta",
    "turn_sign_changes",
    "roll_sign_changes",
    "pitch_sign_changes",
    "alt_residual_std",
    "dive_score_raw",
    "zoom_score_raw",
    "straightness_score",
    "turn_dominance",
    "first_half_turn_rate_mean",
    "second_half_turn_rate_mean",
    "first_half_vertical_speed_mean",
    "second_half_vertical_speed_mean",
    "first_half_pitch_mean",
    "second_half_pitch_mean",
    "first_half_heading_delta",
    "second_half_heading_delta",
    "short_window_count",
    "short_turn_sign_changes_max",
    "short_roll_sign_changes_max",
    "short_pitch_sign_changes_max",
    "short_turn_rate_peak_max",
    "short_roll_rate_peak_max",
    "short_g_std_max",
    "missing_ratio",
    "long_gap_ratio",
    "sensor_conflict",
    "feature_out_of_range",
]


def is_analysis_target_aircraft(obj) -> bool:
    """Return True when the object is a fixed-wing aircraft tracked by this tool."""
    return (
        obj is not None
        and str(getattr(obj, "obj_type", "")).strip().lower() == ANALYSIS_AIRCRAFT_TYPE
    )


def frames_to_aircraft_df(frames, aircraft_id: str) -> pd.DataFrame:
    """
    Build a per-aircraft time series DataFrame from ACMI frames.
    """
    rows = []
    for frame in frames:
        obj = frame.objects.get(aircraft_id)
        if obj is None:
            continue
        rows.append(
            {
                "time": frame.time,
                "altitude": obj.altitude,
                "speed": obj.speed,
                "heading": obj.yaw,
                "pitch": obj.pitch,
                "roll": obj.roll,
                "g_load": obj.g_load,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["time"] + REQUIRED_SIGNALS)

    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def get_all_aircraft_ids(frames) -> dict[str, str]:
    """
    Return {object_id: object_name} for every fixed-wing aircraft found in frames.
    """
    aircraft = {}
    for frame in frames:
        for oid, obj in frame.objects.items():
            if oid not in aircraft and is_analysis_target_aircraft(obj) and obj.name:
                aircraft[oid] = obj.name
    return aircraft


def _unwrap_degrees(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    if valid.sum() < 2:
        return numeric

    unwrapped = numeric.copy()
    rad = np.deg2rad(numeric.loc[valid].to_numpy(dtype=float))
    unwrapped.loc[valid] = np.rad2deg(np.unwrap(rad))
    return unwrapped


def _resample_numeric_series(
    source_times: np.ndarray,
    source_values: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(source_values)
    if valid.sum() == 0:
        return np.full_like(target_times, np.nan, dtype=float)
    if valid.sum() == 1:
        return np.full_like(target_times, source_values[valid][0], dtype=float)
    return np.interp(target_times, source_times[valid], source_values[valid])


def _build_long_gap_mask(
    source_times: np.ndarray,
    target_times: np.ndarray,
    long_gap_sec: float,
) -> np.ndarray:
    if len(source_times) < 2:
        return np.zeros_like(target_times, dtype=float)

    mask = np.zeros_like(target_times, dtype=float)
    for left, right in zip(source_times[:-1], source_times[1:]):
        if (right - left) <= long_gap_sec:
            continue
        inside_gap = (target_times > left) & (target_times < right)
        mask[inside_gap] = 1.0
    return mask


def _rolling_mean(series: pd.Series, window_size: int) -> pd.Series:
    if window_size <= 1:
        return series
    return series.rolling(window=window_size, center=True, min_periods=1).mean()


def _safe_gradient(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros_like(values, dtype=float)
    try:
        return np.gradient(values, times)
    except ValueError:
        return np.zeros_like(values, dtype=float)


def preprocess_timeseries(
    df: pd.DataFrame,
    resample_hz: float = 5.0,
    smoothing_window_sec: float = 0.6,
    long_gap_sec: float = 1.0,
) -> pd.DataFrame:
    """
    Preprocess a single-aircraft time series.

    Steps:
        1. sort by time
        2. unwrap heading / roll
        3. resample to a fixed rate (default 5 Hz)
        4. fill short gaps by interpolation
        5. smooth the kinematic signals with a moving average
        6. derive first-order rates used by the labeler
    """
    if df.empty:
        empty_columns = ["time"] + REQUIRED_SIGNALS + [
            "raw_missing_any",
            "long_gap_flag",
            "heading_rate",
            "roll_rate",
            "pitch_rate",
            "vertical_speed",
        ]
        return pd.DataFrame(columns=empty_columns)

    work = df.copy()
    work["time"] = pd.to_numeric(work["time"], errors="coerce")
    work = work.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["time"] + REQUIRED_SIGNALS)

    if "g_load" not in work.columns:
        work["g_load"] = 1.0

    for col in REQUIRED_SIGNALS:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work["raw_missing_any"] = 1.0 - work[REQUIRED_SIGNALS].notna().all(axis=1).astype(float)
    work["heading"] = _unwrap_degrees(work["heading"])
    work["roll"] = _unwrap_degrees(work["roll"])

    pitch_numeric = pd.to_numeric(work["pitch"], errors="coerce")
    if pitch_numeric.notna().sum() >= 2 and np.nanmax(np.abs(np.diff(pitch_numeric))) > 150.0:
        work["pitch"] = _unwrap_degrees(work["pitch"])

    source_times = work["time"].to_numpy(dtype=float)
    if len(source_times) < 2 or resample_hz <= 0:
        resampled = work[["time"] + REQUIRED_SIGNALS + ["raw_missing_any"]].copy()
        resampled["long_gap_flag"] = 0.0
    else:
        sample_period = 1.0 / resample_hz
        target_times = np.arange(
            source_times[0],
            source_times[-1] + sample_period * 0.5,
            sample_period,
            dtype=float,
        )
        resampled = pd.DataFrame({"time": target_times})
        for col in REQUIRED_SIGNALS:
            resampled[col] = _resample_numeric_series(
                source_times,
                work[col].to_numpy(dtype=float),
                target_times,
            )
        resampled["raw_missing_any"] = _resample_numeric_series(
            source_times,
            work["raw_missing_any"].to_numpy(dtype=float),
            target_times,
        )
        resampled["long_gap_flag"] = _build_long_gap_mask(
            source_times,
            target_times,
            long_gap_sec=long_gap_sec,
        )

    for col in REQUIRED_SIGNALS:
        resampled[col] = resampled[col].interpolate(method="linear").ffill().bfill()

    window_size = max(1, int(round(smoothing_window_sec * resample_hz)))
    for col in REQUIRED_SIGNALS:
        resampled[col] = _rolling_mean(resampled[col], window_size)

    times = resampled["time"].to_numpy(dtype=float)
    resampled["heading_rate"] = _safe_gradient(
        resampled["heading"].to_numpy(dtype=float),
        times,
    )
    resampled["roll_rate"] = _safe_gradient(
        resampled["roll"].to_numpy(dtype=float),
        times,
    )
    resampled["pitch_rate"] = _safe_gradient(
        resampled["pitch"].to_numpy(dtype=float),
        times,
    )
    resampled["vertical_speed"] = _safe_gradient(
        resampled["altitude"].to_numpy(dtype=float),
        times,
    )

    resampled["raw_missing_any"] = resampled["raw_missing_any"].clip(0.0, 1.0)
    return resampled.reset_index(drop=True)


def _linear_slope(times: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    rel_time = times - times[0]
    try:
        slope, _ = np.polyfit(rel_time, values, 1)
    except (np.linalg.LinAlgError, ValueError):
        slope = 0.0
    return float(slope)


def _linear_residual_std(times: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 3:
        return 0.0
    rel_time = times - times[0]
    try:
        slope, intercept = np.polyfit(rel_time, values, 1)
    except (np.linalg.LinAlgError, ValueError):
        return 0.0
    fitted = slope * rel_time + intercept
    return float(np.std(values - fitted, ddof=0))


def _count_sign_changes(values: np.ndarray, deadband: float = 1e-3) -> int:
    if len(values) < 2:
        return 0
    signs = np.sign(values)
    signs[np.abs(values) <= deadband] = 0.0
    filtered = signs[signs != 0.0]
    if len(filtered) < 2:
        return 0
    return int(np.sum(filtered[1:] != filtered[:-1]))


def _window_mean(df: pd.DataFrame, column: str) -> float:
    return float(df[column].mean()) if not df.empty else 0.0


def _window_delta(df: pd.DataFrame, column: str) -> float:
    if df.empty:
        return 0.0
    values = df[column].to_numpy(dtype=float)
    return float(values[-1] - values[0])


def _window_sensor_conflict(window_df: pd.DataFrame) -> bool:
    speed = window_df["speed"].to_numpy(dtype=float)
    pitch = window_df["pitch"].to_numpy(dtype=float)
    g_load = window_df["g_load"].to_numpy(dtype=float)
    heading_rate = np.abs(window_df["heading_rate"].to_numpy(dtype=float))

    return bool(
        np.any(speed < -1e-6)
        or np.any(np.abs(pitch) > 100.0)
        or np.any((g_load < -3.0) | (g_load > 12.0))
        or np.any(heading_rate > 200.0)
    )


def _window_feature_out_of_range(window_df: pd.DataFrame) -> bool:
    altitude = window_df["altitude"].to_numpy(dtype=float)
    vertical_speed = np.abs(window_df["vertical_speed"].to_numpy(dtype=float))
    roll_rate = np.abs(window_df["roll_rate"].to_numpy(dtype=float))
    pitch_rate = np.abs(window_df["pitch_rate"].to_numpy(dtype=float))

    return bool(
        np.any(~np.isfinite(window_df[REQUIRED_SIGNALS + ["heading_rate", "roll_rate", "pitch_rate", "vertical_speed"]].to_numpy(dtype=float)))
        or np.any(altitude < -1000.0)
        or np.any(vertical_speed > 300.0)
        or np.any(roll_rate > 400.0)
        or np.any(pitch_rate > 200.0)
    )


def _compute_half_window_features(window_df: pd.DataFrame) -> dict[str, float]:
    if len(window_df) < 2:
        return {
            "first_half_turn_rate_mean": 0.0,
            "second_half_turn_rate_mean": 0.0,
            "first_half_vertical_speed_mean": 0.0,
            "second_half_vertical_speed_mean": 0.0,
            "first_half_pitch_mean": 0.0,
            "second_half_pitch_mean": 0.0,
            "first_half_heading_delta": 0.0,
            "second_half_heading_delta": 0.0,
        }

    midpoint = (window_df["time"].iloc[0] + window_df["time"].iloc[-1]) / 2.0
    first_half = window_df[window_df["time"] <= midpoint]
    second_half = window_df[window_df["time"] >= midpoint]
    if len(first_half) < 2:
        first_half = window_df.iloc[: max(2, len(window_df) // 2)].copy()
    if len(second_half) < 2:
        second_half = window_df.iloc[-max(2, len(window_df) // 2) :].copy()

    return {
        "first_half_turn_rate_mean": float(np.mean(np.abs(first_half["heading_rate"].to_numpy(dtype=float)))),
        "second_half_turn_rate_mean": float(np.mean(np.abs(second_half["heading_rate"].to_numpy(dtype=float)))),
        "first_half_vertical_speed_mean": _window_mean(first_half, "vertical_speed"),
        "second_half_vertical_speed_mean": _window_mean(second_half, "vertical_speed"),
        "first_half_pitch_mean": _window_mean(first_half, "pitch"),
        "second_half_pitch_mean": _window_mean(second_half, "pitch"),
        "first_half_heading_delta": abs(_window_delta(first_half, "heading")),
        "second_half_heading_delta": abs(_window_delta(second_half, "heading")),
    }


def _compute_short_window_features(
    window_df: pd.DataFrame,
    short_window_sec: float = SHORT_WINDOW_SEC,
) -> dict[str, float]:
    if len(window_df) < 2:
        return {
            "short_window_count": 0,
            "short_turn_sign_changes_max": 0.0,
            "short_roll_sign_changes_max": 0.0,
            "short_pitch_sign_changes_max": 0.0,
            "short_turn_rate_peak_max": 0.0,
            "short_roll_rate_peak_max": 0.0,
            "short_g_std_max": 0.0,
        }

    times = window_df["time"].to_numpy(dtype=float)
    dt = float(np.median(np.diff(times))) if len(times) >= 2 else short_window_sec
    if not np.isfinite(dt) or dt <= 0:
        dt = short_window_sec

    if (times[-1] - times[0]) <= short_window_sec + 1e-9:
        segments = [window_df]
    else:
        starts = np.arange(times[0], times[-1] - short_window_sec + 1e-9, dt, dtype=float)
        segments = []
        for start in starts:
            end = start + short_window_sec
            segment = window_df.loc[
                (window_df["time"] >= start - 1e-9) & (window_df["time"] <= end + 1e-9)
            ]
            if len(segment) >= 2:
                segments.append(segment)

    if not segments:
        return {
            "short_window_count": 0,
            "short_turn_sign_changes_max": 0.0,
            "short_roll_sign_changes_max": 0.0,
            "short_pitch_sign_changes_max": 0.0,
            "short_turn_rate_peak_max": 0.0,
            "short_roll_rate_peak_max": 0.0,
            "short_g_std_max": 0.0,
        }

    return {
        "short_window_count": int(len(segments)),
        "short_turn_sign_changes_max": float(
            max(
                _count_sign_changes(
                    segment["heading_rate"].to_numpy(dtype=float),
                    deadband=0.2,
                )
                for segment in segments
            )
        ),
        "short_roll_sign_changes_max": float(
            max(
                _count_sign_changes(
                    segment["roll_rate"].to_numpy(dtype=float),
                    deadband=1.0,
                )
                for segment in segments
            )
        ),
        "short_pitch_sign_changes_max": float(
            max(
                _count_sign_changes(
                    segment["pitch_rate"].to_numpy(dtype=float),
                    deadband=0.5,
                )
                for segment in segments
            )
        ),
        "short_turn_rate_peak_max": float(
            max(
                np.max(np.abs(segment["heading_rate"].to_numpy(dtype=float)))
                for segment in segments
            )
        ),
        "short_roll_rate_peak_max": float(
            max(
                np.max(np.abs(segment["roll_rate"].to_numpy(dtype=float)))
                for segment in segments
            )
        ),
        "short_g_std_max": float(
            max(
                np.std(segment["g_load"].to_numpy(dtype=float), ddof=0)
                for segment in segments
            )
        ),
    }


def extract_window_features(
    df: pd.DataFrame,
    window_sec: float = 5.0,
    step_sec: float = 2.5,
) -> pd.DataFrame:
    """
    Extract the rule-based labeling feature set from a preprocessed time series.
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=WINDOW_COLUMNS + FEATURE_COLUMNS)

    times = df["time"].to_numpy(dtype=float)
    t_min, t_max = times[0], times[-1]
    if t_max - t_min < window_sec:
        return pd.DataFrame(columns=WINDOW_COLUMNS + FEATURE_COLUMNS)

    results = []
    t_start = t_min

    while t_start + window_sec <= t_max + 1e-9:
        t_end = t_start + window_sec
        window_df = df.loc[(times >= t_start - 1e-9) & (times <= t_end + 1e-9)].copy()
        if len(window_df) < 2:
            t_start += step_sec
            continue

        window_times = window_df["time"].to_numpy(dtype=float)
        altitude = window_df["altitude"].to_numpy(dtype=float)
        speed = window_df["speed"].to_numpy(dtype=float)
        pitch = window_df["pitch"].to_numpy(dtype=float)
        roll = window_df["roll"].to_numpy(dtype=float)
        g_load = window_df["g_load"].to_numpy(dtype=float)
        heading = window_df["heading"].to_numpy(dtype=float)
        heading_rate = window_df["heading_rate"].to_numpy(dtype=float)
        roll_rate = window_df["roll_rate"].to_numpy(dtype=float)
        pitch_rate = window_df["pitch_rate"].to_numpy(dtype=float)
        vertical_speed = window_df["vertical_speed"].to_numpy(dtype=float)

        speed_mean = float(np.mean(speed))
        speed_delta = float(speed[-1] - speed[0])
        speed_slope = _linear_slope(window_times, speed)
        alt_mean = float(np.mean(altitude))
        alt_delta = float(altitude[-1] - altitude[0])
        alt_slope = _linear_slope(window_times, altitude)
        pitch_mean = float(np.mean(pitch))
        roll_mean = float(np.mean(roll))
        g_mean = float(np.mean(g_load))
        g_std = float(np.std(g_load, ddof=0))
        turn_rate_mean = float(np.mean(np.abs(heading_rate)))
        turn_rate_peak = float(np.max(np.abs(heading_rate)))
        roll_rate_mean = float(np.mean(np.abs(roll_rate)))
        roll_rate_peak = float(np.max(np.abs(roll_rate)))
        pitch_rate_mean = float(np.mean(np.abs(pitch_rate)))
        vertical_speed_mean = float(np.mean(vertical_speed))
        vertical_speed_peak = float(np.max(np.abs(vertical_speed)))
        heading_delta = abs(float(heading[-1] - heading[0]))
        turn_sign_changes = _count_sign_changes(heading_rate, deadband=0.2)
        roll_sign_changes = _count_sign_changes(roll_rate, deadband=1.0)
        pitch_sign_changes = _count_sign_changes(pitch_rate, deadband=0.5)
        alt_residual_std = _linear_residual_std(window_times, altitude)
        dive_score_raw = max(0.0, -pitch_mean) + max(0.0, -vertical_speed_mean)
        zoom_score_raw = max(0.0, pitch_mean) + max(0.0, vertical_speed_mean)
        straightness_score = 1.0 / (1.0 + turn_rate_mean)
        turn_dominance = turn_rate_mean / (abs(vertical_speed_mean) + 1e-3)

        row = {
            "window_start": float(t_start),
            "window_end": float(t_end),
            "window_start_time": float(t_start),
            "window_end_time": float(t_end),
            "window_duration": float(window_sec),
            "sample_count": int(len(window_df)),
            "speed_mean": speed_mean,
            "speed_delta": speed_delta,
            "speed_slope": speed_slope,
            "alt_mean": alt_mean,
            "alt_delta": alt_delta,
            "alt_slope": alt_slope,
            "pitch_mean": pitch_mean,
            "roll_mean": roll_mean,
            "g_mean": g_mean,
            "g_std": g_std,
            "turn_rate_mean": turn_rate_mean,
            "turn_rate_peak": turn_rate_peak,
            "roll_rate_mean": roll_rate_mean,
            "roll_rate_peak": roll_rate_peak,
            "pitch_rate_mean": pitch_rate_mean,
            "vertical_speed_mean": vertical_speed_mean,
            "vertical_speed_peak": vertical_speed_peak,
            "heading_delta": heading_delta,
            "turn_sign_changes": turn_sign_changes,
            "roll_sign_changes": roll_sign_changes,
            "pitch_sign_changes": pitch_sign_changes,
            "alt_residual_std": alt_residual_std,
            "dive_score_raw": dive_score_raw,
            "zoom_score_raw": zoom_score_raw,
            "straightness_score": straightness_score,
            "turn_dominance": turn_dominance,
            "missing_ratio": float(window_df["raw_missing_any"].mean()),
            "long_gap_ratio": float(window_df["long_gap_flag"].mean()),
            "sensor_conflict": int(_window_sensor_conflict(window_df)),
            "feature_out_of_range": int(_window_feature_out_of_range(window_df)),
        }
        row.update(_compute_half_window_features(window_df))
        row.update(_compute_short_window_features(window_df))
        results.append(row)
        t_start += step_sec

    if not results:
        return pd.DataFrame(columns=WINDOW_COLUMNS + FEATURE_COLUMNS)

    result_df = pd.DataFrame(results)
    for col in ["sensor_conflict", "feature_out_of_range"]:
        result_df[col] = result_df[col].astype(int)
    return result_df


def build_feature_matrix(
    frames,
    aircraft_id: str,
    window_sec: float = 5.0,
    step_sec: float = 2.5,
    resample_hz: float = 5.0,
    smoothing_window_sec: float = 0.6,
    long_gap_sec: float = 1.0,
) -> pd.DataFrame:
    """
    Parse frames for one aircraft and return one row per analysis window.
    """
    df = frames_to_aircraft_df(frames, aircraft_id)
    if df.empty:
        return pd.DataFrame(columns=WINDOW_COLUMNS + FEATURE_COLUMNS)

    processed = preprocess_timeseries(
        df,
        resample_hz=resample_hz,
        smoothing_window_sec=smoothing_window_sec,
        long_gap_sec=long_gap_sec,
    )
    features = extract_window_features(processed, window_sec=window_sec, step_sec=step_sec)
    features["aircraft_id"] = aircraft_id
    return features
