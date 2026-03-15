"""
Flight Maneuver Classifier — maneuver_labeler のテスト (14クラス / G-Load対応版)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maneuver_labeler import ManeuverLabeler, MANEUVER_CLASSES, NUM_CLASSES


# ============================================================
# ヘルパー: デフォルト行データ (全値明示)
# ============================================================

def _row(**overrides):
    """デフォルト値付きの特徴量行を生成 (g_load 込み)"""
    base = {
        "heading_delta": 0.0,
        "heading_std": 0.0,
        "altitude_slope": 0.0,
        "speed_delta": 0.0,
        "roll_std": 0.0,
        "pitch_mean": 0.0,
        "g_load_mean": 1.0,   # 1G = 水平直線飛行
        "g_load_std": 0.0,
    }
    base.update(overrides)
    return base


# ============================================================
# 基本飛行クラス (0–4)
# ============================================================

class TestBasicFlightClasses:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_straight_and_level(self):
        """全指標が閾値以下, g≈1 → Straight & Level (0)"""
        assert self.labeler.label_single(_row()) == 0

    def test_straight_and_level_with_minor_g(self):
        """g=1.1 but no heading/alt change → still Straight & Level"""
        assert self.labeler.label_single(_row(g_load_mean=1.1)) == 0

    def test_acceleration(self):
        """speed↑ + level + g≈1 → Acceleration (1)"""
        assert self.labeler.label_single(_row(speed_delta=10.0)) == 1

    def test_deceleration(self):
        """speed↓ + level + g≈1 → Deceleration (2)"""
        assert self.labeler.label_single(_row(speed_delta=-10.0)) == 2

    def test_steady_climb(self):
        """alt↑ + 旋回なし → Steady Climb (3)"""
        assert self.labeler.label_single(_row(altitude_slope=5.0, pitch_mean=5.0)) == 3

    def test_steady_descent(self):
        """alt↓ + 旋回なし → Steady Descent (4)"""
        assert self.labeler.label_single(_row(altitude_slope=-5.0, pitch_mean=-3.0)) == 4


# ============================================================
# 旋回系クラス (5–8)
# ============================================================

class TestTurnClasses:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_level_turn(self):
        """heading 変化 + alt flat + g=2 → Level Turn (5)"""
        assert self.labeler.label_single(
            _row(heading_delta=15.0, g_load_mean=2.0)
        ) == 5

    def test_level_turn_left(self):
        """heading 負方向 → Level Turn (5)"""
        assert self.labeler.label_single(
            _row(heading_delta=-15.0, g_load_mean=2.0)
        ) == 5

    def test_climbing_turn(self):
        """heading + alt↑ → Climbing Turn (6)"""
        assert self.labeler.label_single(
            _row(heading_delta=20.0, altitude_slope=5.0, g_load_mean=2.5)
        ) == 6

    def test_descending_turn(self):
        """heading + alt↓ → Descending Turn (7)"""
        assert self.labeler.label_single(
            _row(heading_delta=-20.0, altitude_slope=-5.0, g_load_mean=2.0)
        ) == 7

    def test_high_g_turn(self):
        """g≥4 + heading 変化 → High-G Turn (8)"""
        assert self.labeler.label_single(
            _row(heading_delta=30.0, g_load_mean=5.0)
        ) == 8

    def test_high_g_turn_threshold(self):
        """g=4.0 ちょうど → High-G Turn (8)"""
        assert self.labeler.label_single(
            _row(heading_delta=10.0, g_load_mean=4.0)
        ) == 8

    def test_moderate_g_not_high_g(self):
        """g=3.5 < 4 → Level Turn (5), not High-G"""
        assert self.labeler.label_single(
            _row(heading_delta=15.0, g_load_mean=3.5)
        ) == 5


# ============================================================
# 戦術機動クラス (9–13)
# ============================================================

class TestTacticalClasses:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_dive(self):
        """pitch負 + g<0.5 (unloaded) + alt急降下 → Dive (9)"""
        assert self.labeler.label_single(
            _row(pitch_mean=-25.0, g_load_mean=0.2, altitude_slope=-20.0, speed_delta=15.0)
        ) == 9

    def test_dive_requires_unloaded(self):
        """g=1.0 では Dive にならない (Steady Descent になる)"""
        result = self.labeler.label_single(
            _row(pitch_mean=-20.0, g_load_mean=1.0, altitude_slope=-15.0, speed_delta=10.0)
        )
        assert result == 4  # Steady Descent (not Dive)

    def test_zoom_climb(self):
        """pitch正 + g≥2 + alt急上昇 → Zoom Climb (10)"""
        assert self.labeler.label_single(
            _row(pitch_mean=25.0, g_load_mean=3.0, altitude_slope=20.0, speed_delta=-15.0)
        ) == 10

    def test_zoom_climb_requires_g_pull(self):
        """g=1.0 では Zoom Climb にならない (Steady Climb になる)"""
        result = self.labeler.label_single(
            _row(pitch_mean=20.0, g_load_mean=1.0, altitude_slope=15.0, speed_delta=-10.0)
        )
        assert result == 3  # Steady Climb

    def test_reversal(self):
        """heading大幅変化 → Reversal (11)"""
        assert self.labeler.label_single(
            _row(heading_delta=150.0, g_load_mean=3.0)
        ) == 11

    def test_reversal_negative(self):
        """heading負方向大幅変化 → Reversal (11)"""
        assert self.labeler.label_single(
            _row(heading_delta=-130.0, g_load_mean=3.0)
        ) == 11

    def test_jinking(self):
        """g_load_std 大 + roll_std 大 → Jinking (12)"""
        assert self.labeler.label_single(
            _row(g_load_std=1.5, roll_std=20.0, heading_delta=5.0)
        ) == 12

    def test_jinking_by_g_and_heading_std(self):
        """g_load_std 大 + heading_std 大 → Jinking (12)"""
        assert self.labeler.label_single(
            _row(g_load_std=1.5, heading_std=15.0, roll_std=5.0)
        ) == 12

    def test_low_g_std_not_jinking(self):
        """g_load_std=0.5 < 1.0 → Jinking にならない"""
        result = self.labeler.label_single(
            _row(g_load_std=0.5, roll_std=20.0, heading_std=15.0)
        )
        assert result != 12

    def test_extension(self):
        """speed↑ + 直線 + 微降下 + g≈1 → Extension (13)"""
        assert self.labeler.label_single(
            _row(speed_delta=10.0, altitude_slope=-1.0, g_load_mean=1.0)
        ) == 13


# ============================================================
# 優先度テスト
# ============================================================

class TestPriorityRules:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_jinking_over_high_g(self):
        """Jinking (g_std大) が High-G Turn より優先"""
        assert self.labeler.label_single(
            _row(g_load_mean=5.0, g_load_std=2.0, heading_delta=20.0, roll_std=25.0)
        ) == 12

    def test_reversal_over_high_g(self):
        """Reversal が High-G Turn より優先"""
        assert self.labeler.label_single(
            _row(heading_delta=150.0, g_load_mean=5.0)
        ) == 11

    def test_high_g_over_level_turn(self):
        """g≥4 + turn → High-G Turn (not Level Turn)"""
        assert self.labeler.label_single(
            _row(heading_delta=30.0, g_load_mean=5.0)
        ) == 8

    def test_climb_over_acceleration(self):
        """Steady Climb が Acceleration より優先"""
        assert self.labeler.label_single(
            _row(altitude_slope=5.0, speed_delta=10.0)
        ) == 3

    def test_dive_over_descent(self):
        """Dive (unloaded) が Steady Descent より優先"""
        assert self.labeler.label_single(
            _row(pitch_mean=-20.0, g_load_mean=0.2, altitude_slope=-15.0, speed_delta=10.0)
        ) == 9

    def test_zoom_over_climb(self):
        """Zoom Climb (g≥2, pitch正) が Steady Climb より優先"""
        assert self.labeler.label_single(
            _row(pitch_mean=20.0, g_load_mean=3.0, altitude_slope=15.0)
        ) == 10


# ============================================================
# DataFrame / 設定テスト
# ============================================================

class TestLabelDataFrame:

    def setup_method(self):
        self.labeler = ManeuverLabeler()

    def test_labels_added(self):
        """DataFrame に label / label_name 列が追加される"""
        df = pd.DataFrame([
            _row(),                                                  # Straight & Level
            _row(heading_delta=150.0, g_load_mean=3.0),              # Reversal
        ])
        result = self.labeler.label_dataframe(df)
        assert "label" in result.columns
        assert "label_name" in result.columns
        assert result.iloc[0]["label"] == 0
        assert result.iloc[1]["label"] == 11

    def test_empty_dataframe(self):
        """空 DataFrame"""
        df = pd.DataFrame()
        result = self.labeler.label_dataframe(df)
        assert "label" in result.columns

    def test_custom_thresholds(self):
        """閾値をカスタマイズ — g_load_high_g を上げる"""
        labeler = ManeuverLabeler(thresholds={"g_load_high_g": 6.0})
        # g=5 は閾値 6 未満 → High-G Turn にならず Level Turn
        assert labeler.label_single(
            _row(heading_delta=20.0, g_load_mean=5.0)
        ) == 5

    def test_num_classes(self):
        """14クラスが定義されている"""
        assert NUM_CLASSES == 14
        assert len(MANEUVER_CLASSES) == 14

    def test_g_load_defaults_to_1(self):
        """g_load_mean 未指定時は 1.0 (level) として扱われる"""
        row = {"heading_delta": 0.0, "altitude_slope": 0.0, "speed_delta": 0.0}
        assert ManeuverLabeler().label_single(row) == 0  # Straight & Level
