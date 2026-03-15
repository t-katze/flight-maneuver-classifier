"""
Flight Maneuver Classifier — スライディングウィンドウ特徴量エンジン

ACMI フレームデータから航空機ごとの時系列を構築し、
短時間窓ベースの特徴量を抽出する。

特徴量 (各窓あたり):
    - 平均 (mean): altitude, speed, heading, pitch, roll, g_load
    - 標準偏差 (std): altitude, speed, heading, pitch, roll, g_load
    - 始端終端差 (delta): altitude, speed, heading, pitch, roll, g_load
    - 線形傾き (slope): altitude, speed, heading, pitch, roll, g_load
    - 派生特徴量: altitude_rate, heading_rate, roll_abs_mean
"""

import numpy as np
import pandas as pd


ANALYSIS_AIRCRAFT_TYPE = "air+fixedwing"


# ============================================================
# ACMI フレーム → 航空機時系列 DataFrame
# ============================================================

def is_analysis_target_aircraft(obj) -> bool:
    """機動分類の解析対象かどうかを判定する。"""
    return (
        obj is not None
        and str(getattr(obj, "obj_type", "")).strip().lower() == ANALYSIS_AIRCRAFT_TYPE
    )


def frames_to_aircraft_df(frames, aircraft_id: str) -> pd.DataFrame:
    """
    ACMI フレーム列から特定航空機の時系列 DataFrame を構築する。

    Args:
        frames: ACMIParser.parse_file() の戻り値 (list[ACMIFrame])
        aircraft_id: 抽出する航空機のオブジェクトID

    Returns:
        DataFrame (columns: time, altitude, speed, heading, pitch, roll, g_load)
        該当機が存在しないフレームはスキップされる。
    """
    rows = []
    for frame in frames:
        obj = frame.objects.get(aircraft_id)
        if obj is None:
            continue
        rows.append({
            "time": frame.time,
            "altitude": obj.altitude,       # m (ASL)
            "speed": obj.speed,             # m/s
            "heading": obj.yaw,             # deg
            "pitch": obj.pitch,             # deg
            "roll": obj.roll,               # deg
            "g_load": obj.g_load,           # G
        })

    if not rows:
        return pd.DataFrame(columns=["time", "altitude", "speed",
                                      "heading", "pitch", "roll", "g_load"])

    df = pd.DataFrame(rows)
    df = df.sort_values("time").reset_index(drop=True)
    return df


def get_all_aircraft_ids(frames) -> dict[str, str]:
    """
    全フレームを走査し、航空機のオブジェクトID → 名前のマッピングを返す。

    Returns:
        {obj_id: name} の辞書 (航空機のみ)
    """
    aircraft = {}
    for frame in frames:
        for oid, obj in frame.objects.items():
            if oid not in aircraft and is_analysis_target_aircraft(obj) and obj.name:
                aircraft[oid] = obj.name
    return aircraft


# ============================================================
# 前処理
# ============================================================

def preprocess_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    時系列データの前処理を行う。

    処理内容:
        1. 時刻でソート
        2. 角度の unwrap (heading, roll) — 0°/360° 不連続を除去
        3. 欠損値の線形補間
        4. 残る NaN を前方/後方埋め

    Args:
        df: frames_to_aircraft_df() の出力

    Returns:
        前処理済み DataFrame (元データのコピー)
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    # 角度 unwrap: deg → rad → unwrap → deg
    for col in ["heading", "roll"]:
        if col in df.columns:
            rad = np.deg2rad(df[col].values.astype(float))
            unwrapped = np.unwrap(rad)
            df[col] = np.rad2deg(unwrapped)

    # Pitch は通常 -90°〜+90° なので unwrap 不要だが、念のため
    # 180° ジャンプがあり得る場合のみ unwrap
    if "pitch" in df.columns:
        pitch_diff = np.abs(np.diff(df["pitch"].values.astype(float)))
        if np.any(pitch_diff > 150):
            rad = np.deg2rad(df["pitch"].values.astype(float))
            df["pitch"] = np.rad2deg(np.unwrap(rad))

    # 欠損補間
    if "g_load" not in df.columns:
        df["g_load"] = 1.0

    numeric_cols = ["altitude", "speed", "heading", "pitch", "roll", "g_load"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].interpolate(method="linear")
            df[col] = df[col].ffill().bfill()

    return df


# ============================================================
# スライディングウィンドウ特徴量抽出
# ============================================================

# 特徴量カラムの基底チャネル
_BASE_CHANNELS = ["altitude", "speed", "heading", "pitch", "roll", "g_load"]

# 特徴量サフィックス (基底チャネル × サフィックス)
_SUFFIXES = ["mean", "std", "delta", "slope"]

# 全特徴量名リスト
FEATURE_COLUMNS: list[str] = []
for _ch in _BASE_CHANNELS:
    for _sf in _SUFFIXES:
        FEATURE_COLUMNS.append(f"{_ch}_{_sf}")
# 派生特徴量
FEATURE_COLUMNS.extend(["altitude_rate", "heading_rate", "roll_abs_mean"])


def extract_window_features(
    df: pd.DataFrame,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> pd.DataFrame:
    """
    スライディングウィンドウで特徴量を抽出する。

    Args:
        df: 前処理済み時系列 DataFrame
        window_sec: 窓幅 (秒)
        step_sec: ステップ (秒)

    Returns:
        特徴量 DataFrame (各行 = 1窓)。
        列: window_start, window_end, + FEATURE_COLUMNS
    """
    if df.empty or len(df) < 2:
        return pd.DataFrame(columns=["window_start", "window_end"] + FEATURE_COLUMNS)

    df = df.copy()
    if "g_load" not in df.columns:
        df["g_load"] = 1.0

    times = df["time"].values.astype(float)
    t_min, t_max = times[0], times[-1]

    # 窓幅がデータ全体より長い場合はデータ全体を1窓とする
    if t_max - t_min < window_sec:
        window_sec = t_max - t_min
        if window_sec <= 0:
            return pd.DataFrame(
                columns=["window_start", "window_end"] + FEATURE_COLUMNS
            )

    results = []
    t_start = t_min

    while t_start + window_sec <= t_max + 1e-9:
        t_end = t_start + window_sec
        mask = (times >= t_start - 1e-9) & (times <= t_end + 1e-9)
        window_df = df.loc[mask]

        if len(window_df) < 2:
            t_start += step_sec
            continue

        row = {"window_start": t_start, "window_end": t_end}
        w_times = window_df["time"].values.astype(float)
        dt = w_times[-1] - w_times[0]
        if dt <= 0:
            dt = 1e-6  # ゼロ除算防止

        for ch in _BASE_CHANNELS:
            vals = window_df[ch].values.astype(float)

            # 平均
            row[f"{ch}_mean"] = np.mean(vals)
            # 標準偏差
            row[f"{ch}_std"] = np.std(vals, ddof=0)
            # 始端終端差
            row[f"{ch}_delta"] = vals[-1] - vals[0]
            # 線形傾き (最小二乗)
            if len(vals) >= 2:
                t_rel = w_times - w_times[0]
                try:
                    slope, _ = np.polyfit(t_rel, vals, 1)
                except (np.linalg.LinAlgError, ValueError):
                    slope = 0.0
                row[f"{ch}_slope"] = slope
            else:
                row[f"{ch}_slope"] = 0.0

        # 派生特徴量
        row["altitude_rate"] = row["altitude_delta"] / dt  # m/s
        row["heading_rate"] = row["heading_delta"] / dt     # deg/s
        row["roll_abs_mean"] = np.mean(
            np.abs(window_df["roll"].values.astype(float))
        )

        results.append(row)
        t_start += step_sec

    if not results:
        return pd.DataFrame(columns=["window_start", "window_end"] + FEATURE_COLUMNS)

    result_df = pd.DataFrame(results)
    return result_df


def build_feature_matrix(
    frames,
    aircraft_id: str,
    window_sec: float = 5.0,
    step_sec: float = 1.0,
) -> pd.DataFrame:
    """
    ACMI フレーム → 航空機時系列 → 前処理 → 特徴量抽出 の一括実行。

    Args:
        frames: ACMIParser.parse_file() の戻り値
        aircraft_id: 対象航空機ID
        window_sec: 窓幅 (秒)
        step_sec: ステップ (秒)

    Returns:
        特徴量 DataFrame
    """
    df = frames_to_aircraft_df(frames, aircraft_id)
    if df.empty:
        return pd.DataFrame(columns=["window_start", "window_end"] + FEATURE_COLUMNS)

    df = preprocess_timeseries(df)
    features = extract_window_features(df, window_sec, step_sec)
    features["aircraft_id"] = aircraft_id
    return features
