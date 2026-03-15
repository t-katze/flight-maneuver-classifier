"""
Flight Maneuver Classifier — maneuver_feature_engine のテスト
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acmi_parser import ACMIFrame, ACMIObject
from maneuver_feature_engine import (
    FEATURE_COLUMNS,
    extract_window_features,
    frames_to_aircraft_df,
    get_all_aircraft_ids,
    preprocess_timeseries,
)


# ============================================================
# ヘルパー
# ============================================================

def _make_simple_df(n=50, dt=0.1):
    """テスト用の簡単な時系列を生成"""
    times = np.arange(0, n * dt, dt)
    return pd.DataFrame({
        "time": times,
        "altitude": 5000 + np.arange(n) * 2.0,       # 上昇
        "speed": np.full(n, 250.0),
        "heading": np.linspace(0, 30, n),             # 右旋回
        "pitch": np.full(n, 5.0),
        "roll": np.full(n, 15.0),
        "g_load": np.full(n, 1.2),
    })


class TestAircraftFiltering:
    def test_get_all_aircraft_ids_only_accepts_air_fixedwing(self):
        frames = [
            ACMIFrame(
                time=0.0,
                objects={
                    "1": ACMIObject(
                        obj_id="1",
                        name="F-16C",
                        obj_type="Air+FixedWing",
                    ),
                    "2": ACMIObject(
                        obj_id="2",
                        name="KC-135",
                        obj_type="Air+Refueling",
                    ),
                    "3": ACMIObject(
                        obj_id="3",
                        name="CVN-73",
                        obj_type="Sea+Watercraft+AircraftCarrier",
                    ),
                    "4": ACMIObject(
                        obj_id="4",
                        name="Generic Aircraft",
                        obj_type="Aircraft",
                    ),
                },
            )
        ]

        aircraft = get_all_aircraft_ids(frames)

        assert aircraft == {"1": "F-16C"}


# ============================================================
# preprocess_timeseries
# ============================================================

class TestPreprocessTimeseries:
    def test_empty_df(self):
        df = pd.DataFrame(columns=["time", "altitude", "speed", "heading", "pitch", "roll", "g_load"])
        result = preprocess_timeseries(df)
        assert result.empty

    def test_unwrap_heading_360_to_0(self):
        """heading が 350° → 10° と変化する場合、unwrap で連続化"""
        df = pd.DataFrame({
            "time": [0.0, 1.0, 2.0, 3.0],
            "altitude": [5000] * 4,
            "speed": [250] * 4,
            "heading": [350.0, 355.0, 0.0, 5.0],    # 360→0 跨ぎ
            "pitch": [0] * 4,
            "roll": [0] * 4,
            "g_load": [1.0] * 4,
        })
        result = preprocess_timeseries(df)
        # unwrap 後は単調増加になるはず
        headings = result["heading"].values
        diffs = np.diff(headings)
        assert all(d > 0 for d in diffs), f"Not monotonic: {headings}"
        # 最終値は 365 付近のはず
        assert headings[-1] > 360

    def test_interpolation(self):
        """NaN が補間される"""
        df = pd.DataFrame({
            "time": [0.0, 1.0, 2.0, 3.0],
            "altitude": [5000, np.nan, np.nan, 5300],
            "speed": [250, 260, np.nan, 280],
            "heading": [90] * 4,
            "pitch": [0] * 4,
            "roll": [0] * 4,
            "g_load": [1.0, np.nan, 2.0, 2.5],
        })
        result = preprocess_timeseries(df)
        assert not result["altitude"].isna().any()
        assert not result["speed"].isna().any()
        assert not result["g_load"].isna().any()
        # 中間値は線形補間
        assert abs(result.iloc[1]["altitude"] - 5100) < 1

    def test_sort_by_time(self):
        """時刻でソートされる"""
        df = pd.DataFrame({
            "time": [2.0, 0.0, 1.0],
            "altitude": [5200, 5000, 5100],
            "speed": [250] * 3,
            "heading": [90] * 3,
            "pitch": [0] * 3,
            "roll": [0] * 3,
            "g_load": [1.0] * 3,
        })
        result = preprocess_timeseries(df)
        assert list(result["time"]) == [0.0, 1.0, 2.0]
        assert result.iloc[0]["altitude"] == 5000


# ============================================================
# extract_window_features
# ============================================================

class TestExtractWindowFeatures:
    def test_basic_window_count(self):
        """窓数が正しい"""
        df = _make_simple_df(n=100, dt=0.1)  # 10秒間
        result = extract_window_features(df, window_sec=5.0, step_sec=1.0)
        # 10秒 - 5秒窓 = 最大6窓 (0-5, 1-6, 2-7, 3-8, 4-9, 5-10)
        assert len(result) >= 5

    def test_feature_columns_present(self):
        """全特徴量カラムが存在する"""
        df = _make_simple_df(n=100, dt=0.1)
        result = extract_window_features(df, window_sec=5.0, step_sec=1.0)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_g_load_feature_columns_present(self):
        df = _make_simple_df(n=100, dt=0.1)
        result = extract_window_features(df, window_sec=5.0, step_sec=1.0)

        for col in ["g_load_mean", "g_load_std", "g_load_delta", "g_load_slope"]:
            assert col in FEATURE_COLUMNS
            assert col in result.columns

    def test_window_start_end(self):
        """window_start / window_end が正しい"""
        df = _make_simple_df(n=100, dt=0.1)
        result = extract_window_features(df, window_sec=5.0, step_sec=1.0)
        for _, row in result.iterrows():
            assert row["window_end"] - row["window_start"] == pytest.approx(5.0, abs=0.01)

    def test_altitude_slope_positive(self):
        """高度が上昇しているデータで altitude_slope > 0"""
        df = _make_simple_df(n=100, dt=0.1)
        result = extract_window_features(df, window_sec=5.0, step_sec=1.0)
        assert all(result["altitude_slope"] > 0)

    def test_altitude_delta_matches_slope(self):
        """altitude_delta と altitude_slope の符号が一致"""
        df = _make_simple_df(n=100, dt=0.1)
        result = extract_window_features(df, window_sec=5.0, step_sec=1.0)
        for _, row in result.iterrows():
            assert np.sign(row["altitude_delta"]) == np.sign(row["altitude_slope"])

    def test_empty_df(self):
        """空の DataFrame は空の結果"""
        df = pd.DataFrame(columns=["time", "altitude", "speed", "heading", "pitch", "roll", "g_load"])
        result = extract_window_features(df, window_sec=5.0, step_sec=1.0)
        assert result.empty

    def test_insufficient_data(self):
        """データ点が1つだけの場合"""
        df = pd.DataFrame({
            "time": [0.0],
            "altitude": [5000],
            "speed": [250],
            "heading": [90],
            "pitch": [0],
            "roll": [0],
            "g_load": [1.0],
        })
        result = extract_window_features(df, window_sec=5.0, step_sec=1.0)
        assert result.empty

    def test_short_data_single_window(self):
        """データが窓幅より短い場合、全体を1窓とする"""
        df = pd.DataFrame({
            "time": [0.0, 1.0, 2.0],
            "altitude": [5000, 5050, 5100],
            "speed": [250, 255, 260],
            "heading": [90, 91, 92],
            "pitch": [5, 5, 5],
            "roll": [0, 0, 0],
            "g_load": [1.0, 1.1, 1.2],
        })
        result = extract_window_features(df, window_sec=5.0, step_sec=1.0)
        assert len(result) >= 1

    def test_constant_heading_zero_std(self):
        """一定の heading → heading_std ≈ 0"""
        n = 60
        df = pd.DataFrame({
            "time": np.arange(n) * 0.1,
            "altitude": np.full(n, 5000.0),
            "speed": np.full(n, 250.0),
            "heading": np.full(n, 90.0),
            "pitch": np.full(n, 0.0),
            "roll": np.full(n, 0.0),
            "g_load": np.full(n, 1.0),
        })
        result = extract_window_features(df, window_sec=5.0, step_sec=1.0)
        assert all(result["heading_std"] < 0.01)
