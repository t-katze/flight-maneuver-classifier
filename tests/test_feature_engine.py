"""
Dogfight Supporter — 特徴量エンジニアリングのテスト
"""

import math
import sys
from pathlib import Path

import pytest
import numpy as np

# プロジェクトの src をパスに追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from feature_engine import AircraftState, FeatureEngine, FeatureVector


class TestAircraftState:
    """AircraftState のテスト"""

    def test_default_values(self):
        ac = AircraftState()
        assert ac.x == 0.0
        assert ac.TAS == 0.0
        assert ac.Gy == 0.0

    def test_custom_values(self):
        ac = AircraftState(x=100, y=5000, z=200, TAS=250, heading=1.57)
        assert ac.x == 100
        assert ac.y == 5000
        assert ac.TAS == 250


class TestFeatureVector:
    """FeatureVector のテスト"""

    def test_to_dict(self):
        fv = FeatureVector(distance=1000, aspect_angle=90)
        d = fv.to_dict()
        assert d["distance"] == 1000
        assert d["aspect_angle"] == 90

    def test_to_array(self):
        fv = FeatureVector(distance=1000, aspect_angle=90)
        arr = fv.to_array()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float64

    def test_feature_names(self):
        names = FeatureVector.feature_names()
        assert "distance" in names
        assert "aspect_angle" in names
        assert "antenna_train_angle" in names
        assert len(names) == len(FeatureVector.__dataclass_fields__)


class TestFeatureEngine:
    """FeatureEngine のテスト"""

    def setup_method(self):
        """テストごとにエンジンをリセット"""
        self.engine = FeatureEngine()

    def test_distance_calculation(self):
        """二機間距離の計算"""
        self_ac = AircraftState(x=0, y=0, z=0)
        enemy_ac = AircraftState(x=3000, y=0, z=4000)
        fv = self.engine.compute(self_ac, enemy_ac, 0.0)
        assert abs(fv.distance - 5000.0) < 0.1  # 3-4-5 三角形

    def test_distance_with_altitude(self):
        """高度差を含む3D距離"""
        self_ac = AircraftState(x=0, y=0, z=0)
        enemy_ac = AircraftState(x=0, y=1000, z=0)
        fv = self.engine.compute(self_ac, enemy_ac, 0.0)
        assert abs(fv.distance - 1000.0) < 0.1

    def test_altitude_diff(self):
        """高度差の計算（正=自機が上）"""
        self_ac = AircraftState(alt_asl=5000)
        enemy_ac = AircraftState(alt_asl=3000)
        fv = self.engine.compute(self_ac, enemy_ac, 0.0)
        assert abs(fv.altitude_diff - 2000.0) < 0.1

    def test_specific_energy(self):
        """比エネルギーの計算 E_s = h + V²/(2g)"""
        self_ac = AircraftState(alt_asl=5000, TAS=250)
        fv = self.engine.compute(self_ac, AircraftState(), 0.0)
        expected = 5000 + (250 ** 2) / (2 * 9.80665)
        assert abs(fv.self_specific_energy - expected) < 1.0

    def test_heading_crossing_angle_same_direction(self):
        """同方向のHCA = 0°"""
        self_ac = AircraftState(heading=math.radians(90))
        enemy_ac = AircraftState(heading=math.radians(90))
        fv = self.engine.compute(self_ac, enemy_ac, 0.0)
        assert abs(fv.heading_crossing_angle) < 1.0

    def test_heading_crossing_angle_opposite(self):
        """対面のHCA = 180°"""
        self_ac = AircraftState(heading=0)
        enemy_ac = AircraftState(heading=math.pi)
        fv = self.engine.compute(self_ac, enemy_ac, 0.0)
        assert abs(fv.heading_crossing_angle - 180.0) < 1.0

    def test_turn_rate_from_g_load(self):
        """G荷重からのターンレート計算"""
        self_ac = AircraftState(TAS=200, Gy=4.0)
        fv = self.engine.compute(self_ac, AircraftState(), 0.0)
        # TR = g * sqrt(n²-1) / V
        expected = math.degrees(9.80665 * math.sqrt(16 - 1) / 200)
        assert abs(fv.self_turn_rate - expected) < 0.1

    def test_turn_rate_level_flight(self):
        """1G水平飛行のターンレート = 0"""
        self_ac = AircraftState(TAS=200, Gy=1.0)
        fv = self.engine.compute(self_ac, AircraftState(), 0.0)
        assert fv.self_turn_rate == 0.0

    def test_speed_rate_computation(self):
        """速度変化率の計算（2フレーム必要）"""
        # フレーム1
        self1 = AircraftState(x=0, y=5000, z=0, TAS=200, alt_asl=5000)
        enemy1 = AircraftState(x=3000, y=5000, z=0, alt_asl=5000)
        self.engine.compute(self1, enemy1, 0.0)

        # フレーム2 (dt=1.0s, 速度が210に)
        self2 = AircraftState(x=0, y=5000, z=0, TAS=210, alt_asl=5000)
        enemy2 = AircraftState(x=2900, y=5000, z=0, alt_asl=5000)
        fv = self.engine.compute(self2, enemy2, 1.0)

        assert abs(fv.self_speed_rate - 10.0) < 0.1  # 10 m/s²

    def test_distance_rate_computation(self):
        """距離変化率の計算"""
        # フレーム1: 距離3000m
        self1 = AircraftState(x=0, y=0, z=0)
        enemy1 = AircraftState(x=3000, y=0, z=0)
        self.engine.compute(self1, enemy1, 0.0)

        # フレーム2: 距離2800m (200m接近)
        self2 = AircraftState(x=0, y=0, z=0)
        enemy2 = AircraftState(x=2800, y=0, z=0)
        fv = self.engine.compute(self2, enemy2, 1.0)

        assert abs(fv.distance_rate - (-200.0)) < 1.0  # -200 m/s (接近)

    def test_reset(self):
        """リセット後は微分特徴量が0"""
        # 1フレーム目
        self.engine.compute(
            AircraftState(TAS=200), AircraftState(x=3000), 0.0
        )
        self.engine.reset()

        # リセット後の1フレーム目
        fv = self.engine.compute(
            AircraftState(TAS=250), AircraftState(x=2000), 1.0
        )
        assert fv.self_speed_rate == 0.0
        assert fv.distance_rate == 0.0


class TestFeatureEngineEdgeCases:
    """エッジケースのテスト"""

    def setup_method(self):
        self.engine = FeatureEngine()

    def test_zero_speed(self):
        """速度0の場合（地上等）"""
        ac = AircraftState(TAS=0, Gy=1.0)
        fv = self.engine.compute(ac, AircraftState(), 0.0)
        assert fv.self_turn_rate == 0.0

    def test_same_position(self):
        """二機が同じ位置にいる場合"""
        ac = AircraftState(x=100, y=100, z=100)
        fv = self.engine.compute(ac, ac, 0.0)
        assert fv.distance == 0.0

    def test_very_high_speed(self):
        """超音速でもクラッシュしない"""
        ac = AircraftState(TAS=600, mach=1.8, Gy=5.0, alt_asl=10000)
        fv = self.engine.compute(ac, AircraftState(x=5000), 0.0)
        assert fv.self_turn_rate > 0
        assert fv.self_specific_energy > 10000
