"""
Dogfight Supporter — TacView リアルタイムテレメトリクライアント

TacView のリアルタイムテレメトリポート (TCP 42674) に接続し、
ACMI データをストリーム受信してリアルタイム推論を実行する。

これにより、マルチプレイでもサーバー変更なしで
全機の座標データを取得し [FULL] モードで推論できる。

プロトコル:
    1. TCP 接続
    2. ハンドシェイク: クライアントヘッダ送信 → サーバーヘッダ受信
    3. パスワード認証 (設定されている場合): CRC64 ハッシュ送信
    4. ACMI データのストリーム受信（テキスト行単位）

Usage:
    python tacview_client.py --host SERVER_IP --model models/xgboost_model.joblib
    python tacview_client.py --host SERVER_IP --port 42674 --password SECRET
"""

import argparse
import math
import signal
import socket
import struct
import sys
import json
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acmi_parser import ACMIObject, _geodetic_to_cartesian, KNOTS_TO_MS
from feature_engine import AircraftState, FeatureEngine
from inference import AdvantagePredictor


# ============================================================
# TacView プロトコル定数
# ============================================================
TACVIEW_DEFAULT_PORT = 42674
TACVIEW_CLIENT_HANDSHAKE = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nClient Dogfight-Supporter\n\0"
TACVIEW_HEADER_TERMINATOR = "\0"

# CRC64 テーブル (ECMA-182)
_CRC64_TABLE = None


def _init_crc64_table():
    """CRC64 テーブルを初期化（ECMA-182 多項式）"""
    global _CRC64_TABLE
    if _CRC64_TABLE is not None:
        return

    poly = 0x42F0E1EBA9EA3693
    _CRC64_TABLE = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
        _CRC64_TABLE.append(crc)


def crc64(data: bytes) -> int:
    """CRC64 ハッシュを計算"""
    _init_crc64_table()
    crc = 0xFFFFFFFFFFFFFFFF
    for byte in data:
        crc = _CRC64_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFFFFFFFFFF


def password_to_hash(password: str) -> str:
    """
    パスワードを TacView 認証用の CRC64 ハッシュに変換。
    パスワードは UTF-16LE でエンコードしてからハッシュ化する。
    """
    pw_bytes = password.encode("utf-16-le")
    hash_val = crc64(pw_bytes)
    return f"0x{hash_val:016x}"


# ============================================================
# TacView リアルタイムクライアント
# ============================================================

class TacViewRealtimeClient:
    """
    TacView のリアルタイムテレメトリに接続し、
    全機のデータを受信してドッグファイト推論を実行する。
    """

    def __init__(
        self,
        host: str,
        port: int = TACVIEW_DEFAULT_PORT,
        password: str = "",
        model_path: str = "",
        player_name: str = "",
        result_port: int = 9090,
    ):
        self.host = host
        self.port = port
        self.password = password
        self.player_name = player_name
        self.result_port = result_port

        # TCP ソケット
        self.sock: socket.socket | None = None
        self.buffer = ""

        # ACMI 状態
        self._objects: dict[str, ACMIObject] = {}
        self._current_time: float = 0.0
        self._reference_time: str = ""
        self._self_id: str = ""  # 自機のオブジェクト ID

        # 推論
        self.feature_engine = FeatureEngine()
        self.predictor: AdvantagePredictor | None = None
        if model_path:
            self.predictor = AdvantagePredictor(model_path)

        # 結果送信 (DCS Overlay へ)
        self.send_sock: socket.socket | None = None

        self.running = False
        self.frame_count = 0
        self.start_time = None

    # ============================================================
    # 接続 & ハンドシェイク
    # ============================================================

    def connect(self) -> bool:
        """TacView サーバーに接続しハンドシェイクを実行"""
        print(f"[TacView] Connecting to {self.host}:{self.port}...")

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10.0)
            self.sock.connect((self.host, self.port))
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            print(f"[TacView] Connection failed: {e}")
            return False

        # ---- クライアントハンドシェイク送信 ----
        self.sock.sendall(TACVIEW_CLIENT_HANDSHAKE.encode("utf-8"))
        print("[TacView] Client handshake sent")

        # ---- サーバーハンドシェイク受信 ----
        server_header = self._recv_until_null()
        if not server_header:
            print("[TacView] No server handshake received")
            return False

        print(f"[TacView] Server: {server_header.strip()}")

        # ---- パスワード認証 ----
        if self.password:
            pw_hash = password_to_hash(self.password)
            pw_packet = f"{pw_hash}\n\0"
            self.sock.sendall(pw_packet.encode("utf-8"))
            print(f"[TacView] Password hash sent")
        else:
            # パスワードなしでも空のハッシュを送る必要がある場合がある
            empty_packet = "0\n\0"
            self.sock.sendall(empty_packet.encode("utf-8"))

        # 接続成功
        self.sock.settimeout(1.0)  # ストリーミング用にタイムアウト設定

        # 結果送信ソケット
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print("[TacView] Connected successfully!")
        return True

    def _recv_until_null(self) -> str:
        """NUL文字までデータを受信"""
        data = ""
        while True:
            try:
                chunk = self.sock.recv(4096).decode("utf-8")
            except socket.timeout:
                break
            if not chunk:
                break
            data += chunk
            if "\0" in data:
                data = data.split("\0")[0]
                break
        return data

    # ============================================================
    # ACMI ストリームパース
    # ============================================================

    def _process_line(self, line: str):
        """ACMI 1行を処理"""
        line = line.strip()
        if not line:
            return

        if line.startswith("FileType=") or line.startswith("FileVersion="):
            return

        if line.startswith("#"):
            # タイムフレーム → 推論実行
            try:
                self._current_time = float(line[1:])
            except ValueError:
                return
            self._on_frame()

        elif line.startswith("-"):
            obj_id = line[1:]
            self._objects.pop(obj_id, None)

        elif line.startswith("0,"):
            self._handle_global(line[2:])

        else:
            self._handle_object(line)

    def _handle_global(self, props_str: str):
        """グローバルプロパティ"""
        for prop in props_str.split(","):
            key, _, value = prop.partition("=")
            if key.strip() == "ReferenceTime":
                self._reference_time = value.strip()

    def _handle_object(self, line: str):
        """オブジェクト更新"""
        parts = line.split(",", 1)
        if len(parts) < 2:
            return

        obj_id = parts[0].strip()
        if obj_id not in self._objects:
            self._objects[obj_id] = ACMIObject(obj_id=obj_id)

        obj = self._objects[obj_id]
        remaining = parts[1]

        # T= の特別処理（| を含む）
        props = self._split_props(remaining)
        for prop in props:
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
            # ---- TacView 追加プロパティ (単位変換あり) ----
            elif key == "IAS":
                obj.ias = float(value) * KNOTS_TO_MS    # knots → m/s
            elif key == "TAS":
                obj.tas = float(value) * KNOTS_TO_MS    # knots → m/s
                obj.speed = obj.tas
                obj._has_tas_property = True
            elif key == "Mach":
                obj.mach = float(value)
            elif key == "AOA":
                obj.aoa = math.radians(float(value))    # deg → rad
            elif key == "GLoad":
                obj.g_load = float(value)

    @staticmethod
    def _split_props(s: str) -> list[str]:
        """プロパティ文字列をカンマで分割（T= 内の | は無視）"""
        result = []
        current = ""
        in_t = False
        for ch in s:
            if ch == ",":
                if in_t:
                    in_t = False
                if current.strip():
                    result.append(current.strip())
                current = ""
            else:
                current += ch
                if current.endswith("T="):
                    in_t = True
        if current.strip():
            result.append(current.strip())
        return result

    def _parse_transform(self, obj: ACMIObject, value: str):
        """T= プロパティをパース"""
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

        obj.world_x, obj.world_y, obj.world_z = _geodetic_to_cartesian(
            obj.latitude, obj.longitude, obj.altitude
        )

        # 速度: TacView TAS プロパティ優先、なければ位置差分から推定
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

    # ============================================================
    # 自機検出 & 推論
    # ============================================================

    def _find_self(self) -> ACMIObject | None:
        """自機を名前またはパイロット名で検出"""
        # キャッシュされた ID
        if self._self_id and self._self_id in self._objects:
            return self._objects[self._self_id]

        for oid, obj in self._objects.items():
            if not obj.is_aircraft():
                continue
            # プレイヤー名で照合
            if self.player_name and self.player_name in (obj.pilot or ""):
                self._self_id = oid
                print(f"[TacView] Self aircraft detected: {obj.name} (Pilot: {obj.pilot})")
                return obj
            # F/A-18C でパイロット名が設定されている場合
            if "F/A-18" in (obj.name or "") and obj.pilot:
                self._self_id = oid
                print(f"[TacView] Self aircraft detected: {obj.name} (Pilot: {obj.pilot})")
                return obj

        return None

    def _find_nearest_enemy(self, self_obj: ACMIObject) -> ACMIObject | None:
        """自機に最も近い敵機を検出"""
        best = None
        best_dist = float("inf")

        for oid, obj in self._objects.items():
            if oid == self._self_id:
                continue
            if not obj.is_aircraft():
                continue
            if obj.altitude < 100 or obj.speed < 20:
                continue
            # 異なる陣営
            if obj.coalition and self_obj.coalition and obj.coalition == self_obj.coalition:
                continue

            dist = math.sqrt(
                (obj.world_x - self_obj.world_x) ** 2
                + (obj.world_y - self_obj.world_y) ** 2
                + (obj.world_z - self_obj.world_z) ** 2
            )
            if dist < best_dist:
                best_dist = dist
                best = obj

        return best

    def _acmi_to_state(self, obj: ACMIObject) -> AircraftState:
        """ACMIObject → AircraftState（単位変換済）"""
        return AircraftState(
            x=obj.world_x,
            y=obj.world_y,
            z=obj.world_z,
            heading=math.radians(obj.yaw),      # deg → rad
            pitch=math.radians(obj.pitch),      # deg → rad
            bank=math.radians(obj.roll),        # deg → rad
            TAS=obj.speed,                       # m/s (変換済)
            mach=obj.mach if obj.mach > 0 else obj.speed / 340.0,
            alt_asl=obj.altitude,                # m
        )

    def _on_frame(self):
        """1フレーム完了時に呼ばれる → 推論実行"""
        if not self.predictor:
            return

        self_obj = self._find_self()
        if not self_obj:
            return

        enemy_obj = self._find_nearest_enemy(self_obj)
        if not enemy_obj:
            return

        # 特徴量算出
        self_ac = self._acmi_to_state(self_obj)
        enemy_ac = self._acmi_to_state(enemy_obj)
        features = self.feature_engine.compute(self_ac, enemy_ac, self._current_time)

        # 推論
        result = self.predictor.predict(features.to_dict())

        # DCS Overlay に送信
        self._send_result(result)

        self.frame_count += 1

        # コンソール表示
        score = result["score"]
        bar_len = 10
        pos = int((score + 1.0) / 2.0 * bar_len)
        score_bar = "■" * max(0, bar_len - pos) + "|" + "█" * pos
        print(
            f"\r[{self.frame_count:6d}] "
            f"[FULL/TV] "
            f"{result['category_ja']:4s} "
            f"({score:+.2f}) "
            f"[{score_bar}] "
            f"| dist={features.distance:.0f}m "
            f"| ATA={features.antenna_train_angle:.0f}° "
            f"| vs {enemy_obj.name}",
            end="",
            flush=True,
        )

    def _send_result(self, result: dict):
        """推論結果を DCS Overlay に UDP 送信"""
        if not self.send_sock:
            return
        packet = json.dumps({
            "timestamp": self._current_time,
            "score": result["score"],
            "category": result["category"],
            "category_ja": result["category_ja"],
        }).encode("utf-8")
        self.send_sock.sendto(packet, ("127.0.0.1", self.result_port))

    # ============================================================
    # メインループ
    # ============================================================

    def run(self):
        """接続してストリーム受信ループを実行"""
        if not self.connect():
            print("[TacView] Failed to connect. Exiting.")
            return

        self.running = True
        self.start_time = time.time()

        def signal_handler(sig, frame):
            print("\n[TacView] Shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print(f"[TacView] Streaming ACMI data... (Ctrl+C to stop)")
        if self.player_name:
            print(f"[TacView] Looking for player: {self.player_name}")
        print("-" * 70)

        while self.running:
            try:
                data = self.sock.recv(8192)
            except socket.timeout:
                continue
            except (ConnectionResetError, OSError):
                print("\n[TacView] Connection lost")
                break

            if not data:
                print("\n[TacView] Server closed connection")
                break

            # バッファに追加して行単位で処理
            self.buffer += data.decode("utf-8", errors="replace")
            while "\n" in self.buffer:
                line, self.buffer = self.buffer.split("\n", 1)
                self._process_line(line)

        self.shutdown()

    def shutdown(self):
        """後片付け"""
        if self.sock:
            self.sock.close()
        if self.send_sock:
            self.send_sock.close()

        elapsed = time.time() - self.start_time if self.start_time else 0
        n_aircraft = sum(1 for o in self._objects.values() if o.is_aircraft())

        print(f"\n\n{'=' * 60}")
        print(f"[TacView] Session Summary")
        print(f"  Frames:     {self.frame_count}")
        print(f"  Duration:   {elapsed:.1f}s")
        print(f"  Aircraft:   {n_aircraft}")
        print(f"{'=' * 60}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Dogfight Supporter — TacView リアルタイムクライアント"
    )
    parser.add_argument(
        "--host", required=True, help="TacView サーバーの IP アドレス"
    )
    parser.add_argument(
        "--port", type=int, default=TACVIEW_DEFAULT_PORT,
        help=f"TacView テレメトリポート (default: {TACVIEW_DEFAULT_PORT})"
    )
    parser.add_argument(
        "--password", default="", help="TacView パスワード (設定されている場合)"
    )
    parser.add_argument(
        "--model", default="", help="学習済みモデルパス (.joblib)"
    )
    parser.add_argument(
        "--player", default="",
        help="自分のパイロット名 (自機検出用)"
    )
    parser.add_argument(
        "--result-port", type=int, default=9090,
        help="DCS Overlay への結果送信ポート (default: 9090)"
    )
    args = parser.parse_args()

    client = TacViewRealtimeClient(
        host=args.host,
        port=args.port,
        password=args.password,
        model_path=args.model,
        player_name=args.player,
        result_port=args.result_port,
    )
    client.run()


if __name__ == "__main__":
    main()
