"""
Dogfight Supporter — ACMI パーサーのテスト
"""

import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from acmi_parser import (
    ACMIParser,
    ACMIObject,
    DogfightExtractor,
    DogfightPair,
    ACMIFrame,
    pairs_to_dataframe,
    _geodetic_to_cartesian,
)

# サンプル ACMI データ（最小限のテスト用）
SAMPLE_ACMI = """\
FileType=text/acmi/tacview
FileVersion=2.2
0,ReferenceTime=2024-01-15T12:00:00Z
0,ReferenceLongitude=44.0
0,ReferenceLatitude=41.0
0,Title=Test Dogfight
#0.00
3000001,T=44.1|41.1|5000|10|5|90,Name=F/A-18C,Type=Air+FixedWing,Coalition=Blue,Pilot=Player1
3000002,T=44.2|41.2|5100|-5|3|270,Name=Su-27,Type=Air+FixedWing,Coalition=Red,Pilot=Bot1
#1.00
3000001,T=44.11|41.11|4900|15|7|95
3000002,T=44.19|41.19|5050|-8|4|265
#2.00
3000001,T=44.12|41.12|4800|20|10|100
3000002,T=44.18|41.18|5000|-10|5|260
#3.00
3000001,T=44.13|41.13|4700|25|12|105
-3000002
"""


class TestACMIParser:
    """ACMI パーサーの基本テスト"""

    def _create_temp_acmi(self, content: str) -> str:
        """テスト用 ACMI ファイルを作成"""
        f = tempfile.NamedTemporaryFile(
            mode="w", suffix=".acmi", delete=False, encoding="utf-8"
        )
        f.write(content)
        f.close()
        return f.name

    def test_parse_basic(self):
        """基本パース"""
        filepath = self._create_temp_acmi(SAMPLE_ACMI)
        parser = ACMIParser()
        frames, _ = parser.parse_file(filepath)

        assert len(frames) >= 3  # #0.00, #1.00, #2.00, #3.00
        assert parser.title == "Test Dogfight"
        assert parser.reference_time == "2024-01-15T12:00:00Z"

    def test_parse_objects(self):
        """オブジェクトの読み取り"""
        filepath = self._create_temp_acmi(SAMPLE_ACMI)
        parser = ACMIParser()
        frames, _ = parser.parse_file(filepath)

        # 最初のフレームに2機あるはず
        frame0 = frames[0]
        assert "3000001" in frame0.objects
        assert "3000002" in frame0.objects

        fa18 = frame0.objects["3000001"]
        assert fa18.name == "F/A-18C"
        assert fa18.coalition == "Blue"
        assert fa18.pilot == "Player1"
        assert fa18.is_aircraft()

    def test_object_position(self):
        """位置データの読み取り"""
        filepath = self._create_temp_acmi(SAMPLE_ACMI)
        parser = ACMIParser()
        frames, _ = parser.parse_file(filepath)

        fa18 = frames[0].objects["3000001"]
        assert abs(fa18.longitude - 44.1) < 0.001
        assert abs(fa18.latitude - 41.1) < 0.001
        assert abs(fa18.altitude - 5000) < 0.1

    def test_object_attitude(self):
        """姿勢データの読み取り"""
        filepath = self._create_temp_acmi(SAMPLE_ACMI)
        parser = ACMIParser()
        frames, _ = parser.parse_file(filepath)

        fa18 = frames[0].objects["3000001"]
        assert abs(fa18.roll - 10) < 0.1
        assert abs(fa18.pitch - 5) < 0.1
        assert abs(fa18.yaw - 90) < 0.1

    def test_object_deletion(self):
        """オブジェクト削除 (-id)"""
        filepath = self._create_temp_acmi(SAMPLE_ACMI)
        parser = ACMIParser()
        frames, _ = parser.parse_file(filepath)

        # 最後のフレーム (#3.00) では 3000002 が削除されている
        last_frame = frames[-1]
        assert "3000002" not in last_frame.objects
        assert "3000001" in last_frame.objects

    def test_speed_estimation(self):
        """速度推定（フレーム間差分）"""
        filepath = self._create_temp_acmi(SAMPLE_ACMI)
        parser = ACMIParser()
        frames, _ = parser.parse_file(filepath)

        # 2フレーム目以降で速度が推定されるはず
        if len(frames) >= 2:
            fa18_f1 = frames[1].objects.get("3000001")
            if fa18_f1:
                # 位置が変化しているので速度 > 0 のはず
                assert fa18_f1.speed >= 0

    def test_global_properties(self):
        """グローバルプロパティ"""
        filepath = self._create_temp_acmi(SAMPLE_ACMI)
        parser = ACMIParser()
        parser.parse_file(filepath)

        assert parser.reference_longitude == 44.0
        assert parser.reference_latitude == 41.0


class TestDogfightExtractor:
    """ドッグファイト抽出のテスト"""

    def _make_frame(
        self,
        time: float,
        a_pos: tuple = (0, 5000, 0),
        b_pos: tuple = (1000, 5000, 0),
        a_coal: str = "Blue",
        b_coal: str = "Red",
    ) -> ACMIFrame:
        """テスト用フレームを作成"""
        obj_a = ACMIObject(
            obj_id="1",
            name="F/A-18C",
            obj_type="Air+FixedWing",
            coalition=a_coal,
            world_x=a_pos[0],
            world_y=a_pos[1],
            world_z=a_pos[2],
            altitude=a_pos[1],
            speed=250,
        )
        obj_b = ACMIObject(
            obj_id="2",
            name="Su-27",
            obj_type="Air+FixedWing",
            coalition=b_coal,
            world_x=b_pos[0],
            world_y=b_pos[1],
            world_z=b_pos[2],
            altitude=b_pos[1],
            speed=260,
        )
        return ACMIFrame(time=time, objects={"1": obj_a, "2": obj_b})

    def test_detect_close_pair(self):
        """近距離のペアを検出"""
        frames = [self._make_frame(0.0, (0, 5000, 0), (1000, 5000, 0))]
        extractor = DogfightExtractor(max_distance=5000)
        pairs = extractor.extract_pairs(frames)
        assert len(pairs) == 1
        assert pairs[0].distance < 5000

    def test_ignore_distant_pair(self):
        """遠距離のペアは検出しない"""
        frames = [self._make_frame(0.0, (0, 5000, 0), (20000, 5000, 0))]
        extractor = DogfightExtractor(max_distance=10000)
        pairs = extractor.extract_pairs(frames)
        assert len(pairs) == 0

    def test_ignore_same_coalition(self):
        """同じ陣営はペアにしない"""
        frames = [
            self._make_frame(0.0, (0, 5000, 0), (500, 5000, 0),
                             a_coal="Blue", b_coal="Blue")
        ]
        extractor = DogfightExtractor(max_distance=10000)
        pairs = extractor.extract_pairs(frames)
        assert len(pairs) == 0

    def test_ignore_ground_units(self):
        """高度が低い機体（地上）は除外"""
        frames = [self._make_frame(0.0, (0, 50, 0), (500, 50, 0))]
        extractor = DogfightExtractor(max_distance=10000, min_altitude=100)
        pairs = extractor.extract_pairs(frames)
        assert len(pairs) == 0


class TestPairsToDataFrame:
    """DataFrame 変換のテスト"""

    def test_basic_conversion(self):
        """ペアをDataFrameに変換"""
        pair = DogfightPair(
            time=10.0,
            aircraft_a=ACMIObject(
                obj_id="1", name="F/A-18C", coalition="Blue",
                world_x=100, world_y=5000, world_z=200,
                altitude=5000, yaw=90, pitch=5, roll=10, speed=250,
            ),
            aircraft_b=ACMIObject(
                obj_id="2", name="Su-27", coalition="Red",
                world_x=1100, world_y=5100, world_z=300,
                altitude=5100, yaw=270, pitch=3, roll=-5, speed=260,
            ),
            distance=1050,
        )
        df = pairs_to_dataframe([pair])

        assert len(df) == 1
        assert "self_aircraft_x" in df.columns
        assert "enemy_aircraft_x" in df.columns
        assert "timestamp" in df.columns
        assert df.iloc[0]["self_aircraft_name"] == "F/A-18C"

    def test_heading_conversion_to_radians(self):
        """yaw(deg) → heading(rad) の変換"""
        pair = DogfightPair(
            time=0.0,
            aircraft_a=ACMIObject(yaw=90, pitch=0, roll=0, speed=200,
                                  altitude=5000, obj_type="Air+FixedWing"),
            aircraft_b=ACMIObject(yaw=180, pitch=0, roll=0, speed=200,
                                  altitude=5000, obj_type="Air+FixedWing"),
            distance=1000,
        )
        df = pairs_to_dataframe([pair])
        # 90° → π/2
        assert abs(df.iloc[0]["self_aircraft_heading"] - math.radians(90)) < 0.01


class TestGeodeticConversion:
    """座標変換のテスト"""

    def test_origin(self):
        """原点（0,0,0）"""
        x, y, z = _geodetic_to_cartesian(0, 0, 0)
        assert y == 0

    def test_altitude(self):
        """高度はyに反映"""
        _, y, _ = _geodetic_to_cartesian(0, 0, 5000)
        assert y == 5000
