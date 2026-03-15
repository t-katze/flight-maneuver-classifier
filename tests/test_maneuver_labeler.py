"""
Flight Maneuver Classifier — maneuver_labeler のテスト (14クラス版)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maneuver_labeler import ManeuverLabeler, MANEUVER_CLASSES, NUM_CLASSES


# ============================================================
# 基本飛行クラス (0–4)
# ============================================================

class TestBasicFlightClasses:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_straight_and_level(self):
        """全指標が閾値以下 → Straight & Level (0)"""
        row = {
            "heading_delta": 1.0, "heading_std": 1.0,
            "altitude_slope": 0.5, "speed_delta": 2.0,
            "roll_std": 2.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 0

    def test_acceleration(self):
        """speed 増加 + 直線 + level → Acceleration (1)"""
        row = {
            "heading_delta": 1.0, "heading_std": 1.0,
            "altitude_slope": 0.0, "speed_delta": 10.0,
            "roll_std": 2.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 1

    def test_deceleration(self):
        """speed 減少 + 直線 + level → Deceleration (2)"""
        row = {
            "heading_delta": 1.0, "heading_std": 1.0,
            "altitude_slope": 0.0, "speed_delta": -10.0,
            "roll_std": 2.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 2

    def test_steady_climb(self):
        """altitude 上昇 + 旋回なし → Steady Climb (3)"""
        row = {
            "heading_delta": 2.0, "heading_std": 1.0,
            "altitude_slope": 5.0, "speed_delta": 0.0,
            "roll_std": 2.0, "pitch_mean": 5.0,
        }
        assert self.labeler.label_single(row) == 3

    def test_steady_descent(self):
        """altitude 降下 + 旋回なし → Steady Descent (4)"""
        row = {
            "heading_delta": 2.0, "heading_std": 1.0,
            "altitude_slope": -5.0, "speed_delta": 0.0,
            "roll_std": 2.0, "pitch_mean": -3.0,
        }
        assert self.labeler.label_single(row) == 4


# ============================================================
# 旋回系クラス (5–8)
# ============================================================

class TestTurnClasses:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_level_turn(self):
        """heading 変化 + altitude flat → Level Turn (5)"""
        row = {
            "heading_delta": 15.0, "heading_std": 5.0,
            "altitude_slope": 0.0, "speed_delta": -3.0,
            "roll_std": 5.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 5

    def test_level_turn_left(self):
        """heading 負方向 → Level Turn (5)"""
        row = {
            "heading_delta": -15.0, "heading_std": 5.0,
            "altitude_slope": 0.0, "speed_delta": -3.0,
            "roll_std": 5.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 5

    def test_climbing_turn(self):
        """heading 変化 + altitude 上昇 → Climbing Turn (6)"""
        row = {
            "heading_delta": 20.0, "heading_std": 5.0,
            "altitude_slope": 5.0, "speed_delta": -5.0,
            "roll_std": 5.0, "pitch_mean": 5.0,
        }
        assert self.labeler.label_single(row) == 6

    def test_descending_turn(self):
        """heading 変化 + altitude 降下 → Descending Turn (7)"""
        row = {
            "heading_delta": -20.0, "heading_std": 5.0,
            "altitude_slope": -5.0, "speed_delta": -3.0,
            "roll_std": 5.0, "pitch_mean": -3.0,
        }
        assert self.labeler.label_single(row) == 7

    def test_high_g_turn(self):
        """heading 変化 + speed 大幅減少 → High-G Turn (8)"""
        row = {
            "heading_delta": 30.0, "heading_std": 8.0,
            "altitude_slope": 0.0, "speed_delta": -25.0,
            "roll_std": 10.0, "pitch_mean": 2.0,
        }
        assert self.labeler.label_single(row) == 8


# ============================================================
# 戦術機動クラス (9–13)
# ============================================================

class TestTacticalClasses:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_dive(self):
        """pitch 大きく負 + altitude 急降下 + speed 増加 → Dive (9)"""
        row = {
            "heading_delta": 2.0, "heading_std": 2.0,
            "altitude_slope": -20.0, "speed_delta": 15.0,
            "roll_std": 3.0, "pitch_mean": -25.0,
        }
        assert self.labeler.label_single(row) == 9

    def test_zoom_climb(self):
        """pitch 大きく正 + altitude 急上昇 + speed 減少 → Zoom Climb (10)"""
        row = {
            "heading_delta": 2.0, "heading_std": 2.0,
            "altitude_slope": 20.0, "speed_delta": -15.0,
            "roll_std": 3.0, "pitch_mean": 25.0,
        }
        assert self.labeler.label_single(row) == 10

    def test_reversal(self):
        """heading 大幅変化 (≥120°) → Reversal (11)"""
        row = {
            "heading_delta": 150.0, "heading_std": 30.0,
            "altitude_slope": -5.0, "speed_delta": -10.0,
            "roll_std": 15.0, "pitch_mean": -10.0,
        }
        assert self.labeler.label_single(row) == 11

    def test_reversal_negative(self):
        """heading 大幅変化 (負方向) → Reversal (11)"""
        row = {
            "heading_delta": -130.0, "heading_std": 30.0,
            "altitude_slope": 5.0, "speed_delta": -10.0,
            "roll_std": 15.0, "pitch_mean": 10.0,
        }
        assert self.labeler.label_single(row) == 11

    def test_jinking(self):
        """roll_std + heading_std が大きい → Jinking (12)"""
        row = {
            "heading_delta": 5.0, "heading_std": 15.0,
            "altitude_slope": 0.0, "speed_delta": 0.0,
            "roll_std": 25.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 12

    def test_extension(self):
        """speed 増加 + 直線 + 微降下 → Extension (13)"""
        row = {
            "heading_delta": 2.0, "heading_std": 2.0,
            "altitude_slope": -1.0, "speed_delta": 10.0,
            "roll_std": 3.0, "pitch_mean": -1.0,
        }
        assert self.labeler.label_single(row) == 13


# ============================================================
# 優先度テスト
# ============================================================

class TestPriorityRules:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_jinking_over_turn(self):
        """Jinking が Turn より優先"""
        row = {
            "heading_delta": 20.0, "heading_std": 15.0,
            "altitude_slope": 0.0, "speed_delta": 0.0,
            "roll_std": 25.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 12

    def test_reversal_over_high_g(self):
        """Reversal が High-G Turn より優先"""
        row = {
            "heading_delta": 150.0, "heading_std": 20.0,
            "altitude_slope": 0.0, "speed_delta": -30.0,
            "roll_std": 10.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 11

    def test_high_g_over_level_turn(self):
        """High-G Turn が Level Turn より優先"""
        row = {
            "heading_delta": 30.0, "heading_std": 8.0,
            "altitude_slope": 0.0, "speed_delta": -20.0,
            "roll_std": 10.0, "pitch_mean": 0.0,
        }
        assert self.labeler.label_single(row) == 8

    def test_climb_over_acceleration(self):
        """Steady Climb が Acceleration より優先"""
        row = {
            "heading_delta": 1.0, "heading_std": 1.0,
            "altitude_slope": 5.0, "speed_delta": 10.0,
            "roll_std": 2.0, "pitch_mean": 5.0,
        }
        assert self.labeler.label_single(row) == 3

    def test_dive_over_descent(self):
        """Dive が Steady Descent より優先"""
        row = {
            "heading_delta": 2.0, "heading_std": 2.0,
            "altitude_slope": -15.0, "speed_delta": 10.0,
            "roll_std": 3.0, "pitch_mean": -20.0,
        }
        assert self.labeler.label_single(row) == 9


# ============================================================
# DataFrame / 設定テスト
# ============================================================

class TestLabelDataFrame:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_labels_added(self):
        """DataFrame に label / label_name 列が追加される"""
        df = pd.DataFrame({
            "heading_delta": [1.0, 150.0],
            "heading_std": [1.0, 30.0],
            "altitude_slope": [0.0, -5.0],
            "speed_delta": [0.0, -10.0],
            "roll_std": [2.0, 15.0],
            "pitch_mean": [0.0, -10.0],
        })
        result = self.labeler.label_dataframe(df)
        assert "label" in result.columns
        assert "label_name" in result.columns
        assert result.iloc[0]["label"] == 0   # Straight & Level
        assert result.iloc[1]["label"] == 11  # Reversal

    def test_empty_dataframe(self):
        """空 DataFrame"""
        df = pd.DataFrame()
        result = self.labeler.label_dataframe(df)
        assert "label" in result.columns

    def test_custom_thresholds(self):
        """閾値をカスタマイズ"""
        labeler = ManeuverLabeler(thresholds={
            "heading_delta_threshold": 30.0,
        })
        row = {
            "heading_delta": 20.0, "heading_std": 5.0,
            "altitude_slope": 0.0, "speed_delta": 0.0,
            "roll_std": 2.0, "pitch_mean": 0.0,
        }
        # 20° < threshold 30° → not a turn → Straight & Level
        assert labeler.label_single(row) == 0

    def test_num_classes(self):
        """14クラスが定義されている"""
        assert NUM_CLASSES == 14
        assert len(MANEUVER_CLASSES) == 14
