"""
Dogfight Supporter — TacView ACMI パーサー

TacView の ACMI ファイルをパースし、ドッグファイトのペア（二機組）を検出して
学習データとして抽出する。

ACMI 2.2 フォーマット:
    - UTF-8 テキスト (zip/7z 圧縮の場合もある)
    - "#" で始まる行 → タイムフレーム (秒)
    - "0," → グローバルプロパティ
    - "hex_id,T=..." → オブジェクトの位置・プロパティ
    - "-hex_id" → オブジェクト削除

Usage:
    # ACMI → CSV 変換
    python acmi_parser.py flight.acmi -o data/raw/flight_data.csv

    # ドッグファイトペア抽出付き
    python acmi_parser.py flight.acmi --extract-dogfights -o data/raw/dogfight_pairs.csv
"""

import csv
import math
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import pandas as pd

# 単位変換定数
KNOTS_TO_MS = 0.514444  # 1 knot = 0.514444 m/s


# ============================================================
# データ構造
# ============================================================

@dataclass
class ACMIObject:
    """ACMI 内の1オブジェクト（航空機等）の状態"""
    obj_id: str = ""
    name: str = ""
    pilot: str = ""
    obj_type: str = ""
    coalition: str = ""
    country: str = ""

    # 位置 (T= プロパティから)
    longitude: float = 0.0  # deg
    latitude: float = 0.0   # deg
    altitude: float = 0.0   # m (ASL)

    # 姿勢 (T= プロパティから, deg)
    roll: float = 0.0       # deg
    pitch: float = 0.0      # deg
    yaw: float = 0.0        # deg (heading)

    # 速度 — TacView プロパティから取得（knots → m/s 変換済）
    #         取得不可の場合はフレーム間位置差分から推定
    speed: float = 0.0      # m/s (TAS or 推定値)
    ias: float = 0.0        # m/s (IAS, TacView から。元は knots)
    tas: float = 0.0        # m/s (TAS, TacView から。元は knots)
    mach: float = 0.0       # マッハ数
    aoa: float = 0.0        # 迎角 (rad, TacView は deg で出力 → 変換)
    g_load: float = 1.0     # G荷重 (TacView の GLoad プロパティ)

    # TacView が速度プロパティを提供したかのフラグ
    _has_tas_property: bool = False

    # 直前のワールド座標 (速度推定用)
    _prev_x: float = 0.0
    _prev_y: float = 0.0
    _prev_z: float = 0.0
    _prev_time: float = -1.0

    # ワールド座標 (緯度経度から変換)
    world_x: float = 0.0
    world_y: float = 0.0  # = altitude
    world_z: float = 0.0

    def is_aircraft(self) -> bool:
        """航空機かどうかを判定"""
        t = self.obj_type.lower()
        return any(k in t for k in ["air+fixedwing", "aircraft", "fixedwing"])


@dataclass
class ACMIFrame:
    """1タイムフレームの全オブジェクト状態"""
    time: float
    objects: dict[str, ACMIObject] = field(default_factory=dict)


@dataclass
class DogfightPair:
    """ドッグファイトのペア（二機組）"""
    time: float
    aircraft_a: ACMIObject
    aircraft_b: ACMIObject
    distance: float  # m


@dataclass
class KillEvent:
    """撃墜イベント"""
    time: float
    victim_id: str
    victim_name: str
    killer_id: str       # 推定 — ATAが最小の機体
    killer_name: str
    victim_coalition: str
    killer_coalition: str


# ============================================================
# 座標変換
# ============================================================

# WGS84
_A = 6378137.0          # 赤道半径 (m)
_F = 1.0 / 298.257223563  # 扁平率


def _geodetic_to_cartesian(lat: float, lon: float, alt: float) -> tuple[float, float, float]:
    """
    緯度・経度・高度からローカル直交座標 (m) に変換。
    簡易変換: 基準点からの相対距離を計算。
    """
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)

    # メートル変換 (球面近似)
    x = _A * lon_r * math.cos(lat_r)
    z = _A * lat_r
    y = alt

    return x, y, z


# ============================================================
# ACMI パーサー
# ============================================================

class ACMIParser:
    """
    TacView ACMI 2.2 ファイルのパーサー。

    全てのオブジェクトの時系列データを読み取り、
    ドッグファイトのペアを検出して学習データとして出力する。
    """

    def __init__(self):
        self.reference_time: str = ""
        self.reference_longitude: float = 0.0
        self.reference_latitude: float = 0.0
        self.recording_time: str = ""
        self.title: str = ""
        self.author: str = ""

        # オブジェクト状態 (obj_id → ACMIObject)
        self._objects: dict[str, ACMIObject] = {}

        # 全フレーム履歴
        self._frames: list[ACMIFrame] = []
        self._current_time: float = 0.0

        # 撃墜イベント
        self.kill_events: list[KillEvent] = []

    def parse_file(self, filepath: str) -> tuple[list[ACMIFrame], list[KillEvent]]:
        """
        ACMI ファイルをパースする。

        .acmi / .txt.acmi / .zip.acmi をサポート。

        Returns:
            (frames, kill_events)
        """
        path = Path(filepath)

        if path.suffix == ".zip" or str(path).endswith(".zip.acmi"):
            return self._parse_zip(filepath)
        else:
            return self._parse_text(filepath)

    def _parse_zip(self, filepath: str) -> list[ACMIFrame]:
        """ZIP 圧縮された ACMI をパース"""
        with zipfile.ZipFile(filepath, "r") as zf:
            # 最初のファイルを読む
            name = zf.namelist()[0]
            with zf.open(name) as f:
                text_io = TextIOWrapper(f, encoding="utf-8-sig")
                return self._parse_lines(text_io)

    def _parse_text(self, filepath: str) -> list[ACMIFrame]:
        """テキスト ACMI をパース"""
        with open(filepath, "r", encoding="utf-8-sig") as f:
            return self._parse_lines(f)

    def _parse_lines(self, lines) -> list[ACMIFrame]:
        """行単位でパース"""
        self._pending_time: float | None = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if line.startswith("FileType=") or line.startswith("FileVersion="):
                # ヘッダー行 (スキップ)
                continue

            if line.startswith("#"):
                # 新しいタイムフレーム → 前フレームのスナップショットを保存
                self._flush_frame()
                try:
                    self._current_time = float(line[1:])
                    self._pending_time = self._current_time
                except ValueError:
                    pass
            elif line.startswith("-"):
                # オブジェクト削除 — 航空機なら撃墜イベントの可能性
                obj_id = line[1:]
                if obj_id in self._objects:
                    removed = self._objects[obj_id]
                    if removed.is_aircraft():
                        self._record_kill(obj_id, removed)
                self._objects.pop(obj_id, None)
            elif line.startswith("0,"):
                # グローバルプロパティ（イベント含む）
                self._handle_global_properties(line[2:])
            else:
                # オブジェクト更新
                self._handle_object_update(line)

        # 最後のフレームをフラッシュ
        self._flush_frame()

        print(f"[ACMIParser] Parsed {len(self._frames)} frames, "
              f"{len(self._objects)} objects, "
              f"{len(self.kill_events)} kill events")
        return self._frames, self.kill_events

    def _flush_frame(self):
        """現在のオブジェクト状態をフレームとしてスナップショット保存"""
        if self._pending_time is None:
            return

        frame = ACMIFrame(
            time=self._pending_time,
            objects={
                oid: ACMIObject(
                    obj_id=obj.obj_id,
                    name=obj.name,
                    pilot=obj.pilot,
                    obj_type=obj.obj_type,
                    coalition=obj.coalition,
                    country=obj.country,
                    longitude=obj.longitude,
                    latitude=obj.latitude,
                    altitude=obj.altitude,
                    roll=obj.roll,
                    pitch=obj.pitch,
                    yaw=obj.yaw,
                    speed=obj.speed,
                    ias=obj.ias,
                    tas=obj.tas,
                    mach=obj.mach,
                    aoa=obj.aoa,
                    g_load=obj.g_load,
                    world_x=obj.world_x,
                    world_y=obj.world_y,
                    world_z=obj.world_z,
                )
                for oid, obj in self._objects.items()
            },
        )
        self._frames.append(frame)
        self._pending_time = None

    def _handle_global_properties(self, props_str: str):
        """グローバルプロパティを処理"""
        for prop in self._split_properties(props_str):
            key, _, value = prop.partition("=")
            key = key.strip()
            value = value.strip()

            if key == "ReferenceTime":
                self.reference_time = value
            elif key == "ReferenceLongitude":
                self.reference_longitude = float(value)
            elif key == "ReferenceLatitude":
                self.reference_latitude = float(value)
            elif key == "RecordingTime":
                self.recording_time = value
            elif key == "Title":
                self.title = value
            elif key == "Author":
                self.author = value
            elif key == "Event":
                # イベント処理: "Destroyed|objid" or "HasFired|..."
                self._handle_event(value)

    def _handle_event(self, event_str: str):
        """ACMI イベントを処理"""
        parts = event_str.split("|")
        if len(parts) < 2:
            return

        event_type = parts[0].strip()
        obj_id = parts[1].strip()

        if "Destroyed" in event_type or "LeftArea" in event_type:
            if obj_id in self._objects:
                victim = self._objects[obj_id]
                if victim.is_aircraft():
                    self._record_kill(obj_id, victim)

    def _record_kill(self, victim_id: str, victim: ACMIObject):
        """
        撃墜イベントを記録。

        キラーの推定: 被害機に最も近い敵機のうち、
        ATA（ノーズから被害機への角度）が最小の機体。
        """
        best_killer_id = ""
        best_killer_name = ""
        best_killer_coalition = ""
        best_score = float("inf")

        for oid, obj in self._objects.items():
            if oid == victim_id:
                continue
            if not obj.is_aircraft():
                continue
            # 異なる陣営を優先
            if obj.coalition and victim.coalition and obj.coalition == victim.coalition:
                continue

            # 距離
            dx = obj.world_x - victim.world_x
            dy = obj.world_y - victim.world_y
            dz = obj.world_z - victim.world_z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            # ATA 推定（簡易: 方位角とヘディングの差）
            bearing_to_victim = math.degrees(math.atan2(
                victim.world_x - obj.world_x,
                victim.world_z - obj.world_z,
            ))
            ata = abs(bearing_to_victim - obj.yaw) % 360
            if ata > 180:
                ata = 360 - ata

            # スコア = ATA * 距離重み（低いほうがキラーらしい）
            score = ata + dist / 100.0
            if score < best_score:
                best_score = score
                best_killer_id = oid
                best_killer_name = obj.name
                best_killer_coalition = obj.coalition

        if best_killer_id:
            event = KillEvent(
                time=self._current_time,
                victim_id=victim_id,
                victim_name=victim.name,
                killer_id=best_killer_id,
                killer_name=best_killer_name,
                victim_coalition=victim.coalition or "",
                killer_coalition=best_killer_coalition or "",
            )
            self.kill_events.append(event)
            print(f"[ACMIParser] Kill: {event.killer_name} → {event.victim_name} "
                  f"at t={event.time:.1f}s")

    def _handle_object_update(self, line: str):
        """オブジェクト更新行を処理"""
        # "hex_id,prop1=val1,prop2=val2,T=x|y|z|roll|pitch|yaw"
        parts = line.split(",", 1)
        if len(parts) < 2:
            return

        obj_id = parts[0].strip()
        props_str = parts[1]

        # オブジェクトがなければ作成
        if obj_id not in self._objects:
            self._objects[obj_id] = ACMIObject(obj_id=obj_id)

        obj = self._objects[obj_id]

        for prop in self._split_properties(props_str):
            key, _, value = prop.partition("=")
            key = key.strip()
            value = value.strip()

            if key == "T":
                self._parse_transform(obj, value)
            elif key == "Name":
                obj.name = value
            elif key == "Pilot":
                obj.pilot = value
            elif key == "Type":
                obj.obj_type = value
            elif key == "Coalition":
                obj.coalition = value
            elif key == "Country":
                obj.country = value
            # ---- TacView 追加プロパティ (単位変換あり) ----
            elif key == "IAS":
                # TacView: IAS は knots → m/s に変換
                obj.ias = float(value) * KNOTS_TO_MS
            elif key == "TAS":
                # TacView: TAS は knots → m/s に変換
                obj.tas = float(value) * KNOTS_TO_MS
                obj.speed = obj.tas  # TAS が最も正確
                obj._has_tas_property = True
            elif key == "Mach":
                obj.mach = float(value)
            elif key == "AOA":
                # TacView: AOA は degrees → radians に変換
                obj.aoa = math.radians(float(value))
            elif key == "GLoad":
                obj.g_load = float(value)

    def _parse_transform(self, obj: ACMIObject, value: str):
        """
        T= プロパティをパース。

        形式: longitude|latitude|altitude|roll|pitch|yaw
         or:  longitude|latitude|altitude
        座標が空の場合は前回値を維持。
        """
        parts = value.split("|")

        if len(parts) >= 3:
            if parts[0]:
                obj.longitude = float(parts[0])
            if parts[1]:
                obj.latitude = float(parts[1])
            if parts[2]:
                obj.altitude = float(parts[2])

        if len(parts) >= 6:
            if parts[3]:
                obj.roll = float(parts[3])
            if parts[4]:
                obj.pitch = float(parts[4])
            if parts[5]:
                obj.yaw = float(parts[5])

        # ワールド座標に変換
        obj.world_x, obj.world_y, obj.world_z = _geodetic_to_cartesian(
            obj.latitude, obj.longitude, obj.altitude
        )

        # 速度: TacView の TAS プロパティがあればそれを優先
        # なければフレーム間位置差分から推定（対地速度に近い）
        if not obj._has_tas_property:
            if obj._prev_time >= 0 and self._current_time > obj._prev_time:
                dt = self._current_time - obj._prev_time
                dx = obj.world_x - obj._prev_x
                dy = obj.world_y - obj._prev_y
                dz = obj.world_z - obj._prev_z
                obj.speed = math.sqrt(dx * dx + dy * dy + dz * dz) / dt

        obj._prev_x = obj.world_x
        obj._prev_y = obj.world_y
        obj._prev_z = obj.world_z
        obj._prev_time = self._current_time

    @staticmethod
    def _split_properties(props_str: str) -> list[str]:
        """
        プロパティ文字列を分割する。

        ACMI フォーマット: "T=lon|lat|alt|roll|pitch|yaw,Name=F/A-18C,Type=..."
        T= の値は "|" 区切りだが、他のプロパティとの区切りは "," 。
        """
        result = []
        current = ""
        in_transform = False

        for ch in props_str:
            if ch == ",":
                if in_transform:
                    # T= 値の終端: カンマが来たら transform は終了
                    in_transform = False
                if current.strip():
                    result.append(current.strip())
                current = ""
            else:
                current += ch
                # "T=" の直後に入ったら transform モードに入る
                if current.endswith("T=") and not in_transform:
                    in_transform = True

        if current.strip():
            result.append(current.strip())

        return result


# ============================================================
# ドッグファイト検出
# ============================================================

class DogfightExtractor:
    """
    ACMI フレームデータからドッグファイトのペア（交戦中の二機）を抽出する。
    """

    def __init__(
        self,
        max_distance: float = 10000.0,   # ドッグファイト判定距離 (m)
        min_speed: float = 50.0,          # 最低速度 (m/s, 地上機排除)
        min_altitude: float = 100.0,      # 最低高度 (m, 地上機排除)
    ):
        self.max_distance = max_distance
        self.min_speed = min_speed
        self.min_altitude = min_altitude

    def extract_pairs(
        self, frames: list[ACMIFrame]
    ) -> list[DogfightPair]:
        """
        全フレームからドッグファイトペアを抽出する。

        異なる陣営の航空機同士で、指定距離内にいるペアを返す。
        """
        pairs: list[DogfightPair] = []

        for frame in frames:
            # 航空機のみフィルタ
            aircraft = [
                obj
                for obj in frame.objects.values()
                if obj.is_aircraft()
                and obj.altitude >= self.min_altitude
                and obj.speed >= self.min_speed
            ]

            if len(aircraft) < 2:
                continue

            # 全ペアをチェック
            for i in range(len(aircraft)):
                for j in range(i + 1, len(aircraft)):
                    a = aircraft[i]
                    b = aircraft[j]

                    # 異なる陣営かチェック
                    if a.coalition and b.coalition and a.coalition == b.coalition:
                        continue

                    # 距離計算
                    dist = math.sqrt(
                        (a.world_x - b.world_x) ** 2
                        + (a.world_y - b.world_y) ** 2
                        + (a.world_z - b.world_z) ** 2
                    )

                    if dist <= self.max_distance:
                        pairs.append(DogfightPair(
                            time=frame.time,
                            aircraft_a=a,
                            aircraft_b=b,
                            distance=dist,
                        ))

        print(f"[DogfightExtractor] Found {len(pairs)} dogfight pair frames")
        return pairs


# ============================================================
# CSV 変換
# ============================================================

def _acmi_obj_to_aircraft_state_dict(
    obj: ACMIObject, prefix: str
) -> dict:
    """
    ACMIObject を feature_engine 互換の辞書に変換。

    単位変換:
        - yaw/pitch/roll: deg → rad (feature_engine が rad を期待)
        - speed: すでに m/s (knots→m/s 変換はパース時に実施済)
        - mach: TacView プロパティがあればそれを使用、なければ速度から近似
        - aoa: すでに rad (deg→rad 変換はパース時に実施済)
    """
    return {
        f"{prefix}_x": obj.world_x,
        f"{prefix}_y": obj.world_y,
        f"{prefix}_z": obj.world_z,
        f"{prefix}_heading": math.radians(obj.yaw),    # deg → rad
        f"{prefix}_pitch": math.radians(obj.pitch),    # deg → rad
        f"{prefix}_bank": math.radians(obj.roll),      # deg → rad
        f"{prefix}_TAS": obj.speed,                     # m/s (変換済)
        f"{prefix}_IAS": obj.ias,                       # m/s (変換済)
        f"{prefix}_mach": obj.mach if obj.mach > 0 else obj.speed / 340.0,
        f"{prefix}_alt_asl": obj.altitude,              # m (MSL)
        f"{prefix}_AoA": obj.aoa,                       # rad (変換済)
        f"{prefix}_Gy": obj.g_load,                     # G
        f"{prefix}_name": obj.name,
        f"{prefix}_pilot": obj.pilot,
        f"{prefix}_coalition": obj.coalition,
    }


def pairs_to_dataframe(pairs: list[DogfightPair]) -> pd.DataFrame:
    """ドッグファイトペアを DataFrame に変換"""
    rows = []
    for pair in pairs:
        row = {"timestamp": pair.time, "distance_raw": pair.distance}
        row.update(_acmi_obj_to_aircraft_state_dict(pair.aircraft_a, "self_aircraft"))
        row.update(_acmi_obj_to_aircraft_state_dict(pair.aircraft_b, "enemy_aircraft"))
        rows.append(row)

    return pd.DataFrame(rows)


def acmi_to_training_csv(
    acmi_path: str,
    output_path: str,
    max_distance: float = 10000.0,
) -> pd.DataFrame:
    """
    ACMI ファイルをパースし、ドッグファイトペアを抽出して CSV に保存。

    これが学習データ生成のメインエントリポイント。
    
    Args:
        acmi_path: TacView ACMI ファイルパス
        output_path: 出力 CSV パス
        max_distance: ドッグファイト判定距離 (m)

    Returns:
        ドッグファイトペアの DataFrame
    """
    # パース
    parser = ACMIParser()
    frames, kill_events = parser.parse_file(acmi_path)

    print(f"[ACMI] Title: {parser.title}")
    print(f"[ACMI] Reference Time: {parser.reference_time}")

    # ドッグファイト抽出
    extractor = DogfightExtractor(max_distance=max_distance)
    pairs = extractor.extract_pairs(frames)

    if not pairs:
        print("[ACMI] No dogfight pairs found")
        return pd.DataFrame()

    # DataFrame 変換
    df = pairs_to_dataframe(pairs)

    # 保存
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[ACMI] Saved {len(df)} rows to {output_path}")

    return df


# ============================================================
# CLI
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Dogfight Supporter — TacView ACMI パーサー"
    )
    parser.add_argument("acmi_file", help="TacView ACMI ファイルパス")
    parser.add_argument(
        "-o", "--output", default=None, help="出力 CSV パス"
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=10000,
        help="ドッグファイト判定距離 (m, default: 10000)",
    )
    parser.add_argument(
        "--list-objects",
        action="store_true",
        help="ファイル内のオブジェクト一覧を表示して終了",
    )
    args = parser.parse_args()

    if args.list_objects:
        # オブジェクト一覧モード
        acmi_parser = ACMIParser()
        frames, _ = acmi_parser.parse_file(args.acmi_file)
        seen = {}
        for frame in frames:
            for oid, obj in frame.objects.items():
                if oid not in seen and obj.name:
                    seen[oid] = obj
        print(f"\n{'ID':>10} {'Type':30s} {'Coalition':12s} {'Name'}")
        print("-" * 80)
        for oid, obj in sorted(seen.items()):
            if obj.is_aircraft():
                print(f"{oid:>10} {obj.obj_type:30s} {obj.coalition:12s} {obj.name}")
        return

    # CSV 出力
    if args.output is None:
        p = Path(args.acmi_file)
        args.output = str(p.parent / f"{p.stem}_dogfights.csv")

    acmi_to_training_csv(args.acmi_file, args.output, args.max_distance)


if __name__ == "__main__":
    main()
