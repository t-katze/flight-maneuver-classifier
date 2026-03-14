"""
Flight Maneuver Classifier — maneuver_labeler のテスト
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maneuver_labeler import ManeuverLabeler, MANEUVER_CLASSES


class TestManeuverLabeler:
    """ManeuverLabeler のルール判定テスト"""

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_level_flight(self):
        """全指標が閾値以下 → Level Flight"""
        row = {
            "roll_std": 2.0, "roll_delta": 1.0,
            "heading_delta": 1.0, "altitude_slope": 0.5,
        }
        assert self.labeler.label_single(row) == 0

    def test_climb(self):
        """高度上昇かつ旋回なし → Climb"""
        row = {
            "roll_std": 2.0, "roll_delta": 1.0,
            "heading_delta": 2.0, "altitude_slope": 5.0,
        }
        assert self.labeler.label_single(row) == 1

    def test_descent(self):
        """高度降下かつ旋回なし → Descent"""
        row = {
            "roll_std": 2.0, "roll_delta": 1.0,
            "heading_delta": 2.0, "altitude_slope": -5.0,
        }
        assert self.labeler.label_single(row) == 2

    def test_left_turn(self):
        """heading_delta < 0 → Left Turn"""
        row = {
            "roll_std": 5.0, "roll_delta": 3.0,
            "heading_delta": -10.0, "altitude_slope": 0.0,
        }
        assert self.labeler.label_single(row) == 3

    def test_right_turn(self):
        """heading_delta > 0 → Right Turn"""
        row = {
            "roll_std": 5.0, "roll_delta": 3.0,
            "heading_delta": 10.0, "altitude_slope": 0.0,
        }
        assert self.labeler.label_single(row) == 4

    def test_roll_maneuver_by_std(self):
        """roll_std が大きい → Roll Maneuver"""
        row = {
            "roll_std": 20.0, "roll_delta": 5.0,
            "heading_delta": 10.0, "altitude_slope": 5.0,
        }
        assert self.labeler.label_single(row) == 5

    def test_roll_maneuver_by_delta(self):
        """roll_delta が大きい → Roll Maneuver"""
        row = {
            "roll_std": 5.0, "roll_delta": 40.0,
            "heading_delta": 10.0, "altitude_slope": 5.0,
        }
        assert self.labeler.label_single(row) == 5

    def test_roll_priority_over_turn(self):
        """Roll Maneuver が Turn より優先される"""
        row = {
            "roll_std": 25.0, "roll_delta": 50.0,
            "heading_delta": -20.0, "altitude_slope": 0.0,
        }
        # heading_delta で Left Turn にもなり得るが、Roll 優先
        assert self.labeler.label_single(row) == 5

    def test_turn_priority_over_climb(self):
        """Turn が Climb/Descent より優先される"""
        row = {
            "roll_std": 3.0, "roll_delta": 2.0,
            "heading_delta": 15.0, "altitude_slope": 5.0,
        }
        # Climb でもあるが、Turn 優先
        assert self.labeler.label_single(row) == 4

    def test_custom_thresholds(self):
        """閾値を変更可能"""
        labeler = ManeuverLabeler(thresholds={
            "heading_delta_threshold": 20.0,  # 閾値を大きくする
        })
        row = {
            "roll_std": 2.0, "roll_delta": 1.0,
            "heading_delta": 10.0, "altitude_slope": 0.0,  # 10° は閾値以下
        }
        # heading_delta=10 < threshold=20 → Level Flight
        assert labeler.label_single(row) == 0


class TestLabelDataFrame:
    """label_dataframe のテスト"""

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_labels_added(self):
        """DataFrame に label / label_name 列が追加される"""
        df = pd.DataFrame({
            "roll_std": [2.0, 20.0],
            "roll_delta": [1.0, 5.0],
            "heading_delta": [1.0, 0.0],
            "altitude_slope": [0.0, 0.0],
        })
        result = self.labeler.label_dataframe(df)
        assert "label" in result.columns
        assert "label_name" in result.columns
        assert result.iloc[0]["label"] == 0  # Level Flight
        assert result.iloc[1]["label"] == 5  # Roll Maneuver

    def test_label_names_correct(self):
        """label_name が MANEUVER_CLASSES に対応"""
        df = pd.DataFrame({
            "roll_std": [2.0],
            "roll_delta": [1.0],
            "heading_delta": [-10.0],
            "altitude_slope": [0.0],
        })
        result = self.labeler.label_dataframe(df)
        assert result.iloc[0]["label_name"] == "Left Turn"

    def test_empty_dataframe(self):
        """空 DataFrame"""
        df = pd.DataFrame()
        result = self.labeler.label_dataframe(df)
        assert "label" in result.columns

    def test_all_classes_possible(self):
        """6クラス全てが出力可能"""
        df = pd.DataFrame({
            "roll_std":       [2.0,  2.0,  2.0,  5.0, 5.0, 20.0],
            "roll_delta":     [1.0,  1.0,  1.0,  3.0, 3.0, 40.0],
            "heading_delta":  [1.0,  1.0,  1.0, -10., 10., 0.0],
            "altitude_slope": [0.5,  5.0, -5.0,  0.0, 0.0, 0.0],
        })
        result = self.labeler.label_dataframe(df)
        labels = set(result["label"].tolist())
        assert labels == {0, 1, 2, 3, 4, 5}
