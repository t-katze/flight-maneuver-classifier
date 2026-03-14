"""
Dogfight Supporter — リアルタイム推論サーバー

DCS から UDP でフレームデータを受信し、
特徴量算出 → モデル推論 → 結果送信 をリアルタイムで実行する。

Usage:
    python realtime_server.py --model models/xgboost_model.joblib
    python realtime_server.py --model models/xgboost_model.joblib --port 9089 --result-port 9090
"""

import argparse
import json
import signal
import socket
import sys
import time
from pathlib import Path

# プロジェクトルートを sys.path に追加
sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_engine import AircraftState, FeatureEngine, FeatureVector
from inference import AdvantagePredictor


# ============================================================
# 設定
# ============================================================
DEFAULT_RECV_PORT = 9089  # DCS からデータを受信
DEFAULT_SEND_PORT = 9090  # DCS へ結果を送信
DEFAULT_HOST = "0.0.0.0"
DCS_HOST = "127.0.0.1"  # DCS へ送信する先
BUFFER_SIZE = 65536


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """ネストされた辞書をフラット化"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class RealtimeServer:
    """リアルタイム推論サーバー"""

    def __init__(
        self,
        model_path: str,
        recv_host: str = DEFAULT_HOST,
        recv_port: int = DEFAULT_RECV_PORT,
        send_port: int = DEFAULT_SEND_PORT,
    ):
        self.recv_host = recv_host
        self.recv_port = recv_port
        self.send_port = send_port

        # モジュール初期化
        self.feature_engine = FeatureEngine()
        self.predictor = AdvantagePredictor(model_path)

        # ソケット
        self.recv_sock: socket.socket | None = None
        self.send_sock: socket.socket | None = None

        self.running = False
        self.frame_count = 0
        self.start_time = None

        # パフォーマンス計測
        self._inference_times: list[float] = []

    def setup(self):
        """ソケット初期化"""
        # 受信ソケット
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.recv_sock.bind((self.recv_host, self.recv_port))
        self.recv_sock.settimeout(1.0)

        # 送信ソケット
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        print(f"[Server] Receiving on {self.recv_host}:{self.recv_port}")
        print(f"[Server] Sending results to {DCS_HOST}:{self.send_port}")

    def parse_frame(self, data: bytes) -> tuple[AircraftState, AircraftState | None, float, bool]:
        """
        受信データをパースして AircraftState に変換。

        Returns:
            (self_ac, enemy_ac_or_None, timestamp, self_only)
        """
        try:
            raw = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        flat = flatten_dict(raw)
        timestamp = flat.get("timestamp", flat.get("model_time", 0))
        self_only = flat.get("self_only", not flat.get("has_enemy", False))

        # 自機
        self_ac = AircraftState(
            x=flat.get("self_aircraft_x", 0),
            y=flat.get("self_aircraft_y", 0),
            z=flat.get("self_aircraft_z", 0),
            heading=flat.get("self_aircraft_heading", 0),
            pitch=flat.get("self_aircraft_pitch", 0),
            bank=flat.get("self_aircraft_bank", 0),
            TAS=flat.get("self_aircraft_TAS", 0),
            mach=flat.get("self_aircraft_mach", 0),
            alt_asl=flat.get("self_aircraft_alt_asl", 0),
            AoA=flat.get("self_aircraft_AoA", 0),
            Gy=flat.get("self_aircraft_Gy", 0),
            roll_rate=flat.get("self_aircraft_roll_rate", 0),
            pitch_rate=flat.get("self_aircraft_pitch_rate", 0),
            yaw_rate=flat.get("self_aircraft_yaw_rate", 0),
            throttle=flat.get("self_aircraft_throttle", 0),
        )

        # 敵機（self_only の場合は None）
        enemy_ac = None
        if not self_only:
            enemy_ac = AircraftState(
                x=flat.get("enemy_aircraft_x", 0),
                y=flat.get("enemy_aircraft_y", 0),
                z=flat.get("enemy_aircraft_z", 0),
                heading=flat.get("enemy_aircraft_heading", 0),
                pitch=flat.get("enemy_aircraft_pitch", 0),
                bank=flat.get("enemy_aircraft_bank", 0),
                TAS=flat.get("enemy_aircraft_TAS", 0),
                alt_asl=flat.get("enemy_aircraft_alt", flat.get("enemy_aircraft_y", 0)),
            )

        return self_ac, enemy_ac, timestamp, self_only

    def send_result(self, result: dict, timestamp: float):
        """推論結果を DCS に送信"""
        packet = {
            "timestamp": timestamp,
            "score": result["score"],
            "category": result["category"],
            "category_ja": result["category_ja"],
        }
        json_bytes = json.dumps(packet).encode("utf-8")
        self.send_sock.sendto(json_bytes, (DCS_HOST, self.send_port))

    def run(self):
        """メインループ"""
        self.setup()
        self.running = True
        self.start_time = time.time()

        def signal_handler(sig, frame):
            print("\n[Server] Shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("[Server] Ready. Waiting for DCS data... (Ctrl+C to stop)")
        print("-" * 60)

        while self.running:
            try:
                data, addr = self.recv_sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            # パース
            parsed = self.parse_frame(data)
            if parsed is None:
                continue

            self_ac, enemy_ac, timestamp, self_only = parsed

            # 特徴量算出 & 推論
            t0 = time.perf_counter()

            if self_only:
                # マルチプレイモード: 自機データのみ
                features = self.feature_engine.compute_self_only(self_ac, timestamp)
            else:
                # フルモード: 自機 + 敵機データ
                features = self.feature_engine.compute(self_ac, enemy_ac, timestamp)

            result = self.predictor.predict(features.to_dict())

            t1 = time.perf_counter()
            inference_ms = (t1 - t0) * 1000.0
            self._inference_times.append(inference_ms)

            # 結果送信
            self.send_result(result, timestamp)

            self.frame_count += 1

            # コンソール表示
            score = result["score"]
            # スコアバー: -1✅0 は■、0✅1 は█
            bar_len = 10
            pos = int((score + 1.0) / 2.0 * bar_len)
            score_bar = "■" * max(0, bar_len - pos) + "|" + "█" * pos
            mode_str = "SELF" if self_only else "FULL"
            print(
                f"\r[{self.frame_count:6d}] "
                f"[{mode_str}] "
                f"{result['category_ja']:4s} "
                f"({score:+.2f}) "
                f"[{score_bar}] "
                f"| {inference_ms:.1f}ms "
                f"| dist={features.distance:.0f}m "
                f"| ATA={features.antenna_train_angle:.0f}°",
                end="",
                flush=True,
            )

        self.shutdown()

    def shutdown(self):
        """後片付け"""
        if self.recv_sock:
            self.recv_sock.close()
        if self.send_sock:
            self.send_sock.close()

        elapsed = time.time() - self.start_time if self.start_time else 0
        avg_ms = (
            sum(self._inference_times) / len(self._inference_times)
            if self._inference_times
            else 0
        )

        print(f"\n\n{'=' * 60}")
        print(f"[Server] Session Summary")
        print(f"  Frames:        {self.frame_count}")
        print(f"  Duration:      {elapsed:.1f}s")
        print(f"  Avg FPS:       {self.frame_count / elapsed:.1f}" if elapsed > 0 else "")
        print(f"  Avg Inference: {avg_ms:.2f}ms")
        if self._inference_times:
            print(f"  Max Inference: {max(self._inference_times):.2f}ms")
        print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Dogfight Supporter — リアルタイム推論サーバー"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="学習済みモデルパス (.joblib)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_RECV_PORT,
        help=f"DCS データ受信ポート (default: {DEFAULT_RECV_PORT})",
    )
    parser.add_argument(
        "--result-port",
        type=int,
        default=DEFAULT_SEND_PORT,
        help=f"推論結果送信ポート (default: {DEFAULT_SEND_PORT})",
    )
    args = parser.parse_args()

    server = RealtimeServer(
        model_path=args.model,
        recv_port=args.port,
        send_port=args.result_port,
    )
    server.run()


if __name__ == "__main__":
    main()
