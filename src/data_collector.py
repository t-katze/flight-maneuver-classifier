"""
Dogfight Supporter — UDP データ収集サーバー

DCS の Export Script から UDP で送信されるフレームデータを受信し、
CSV ファイルに保存する。

Usage:
    python data_collector.py                       # デフォルト設定で起動
    python data_collector.py --port 9089 --out data/raw/session.csv
"""

import argparse
import csv
import json
import os
import signal
import socket
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# 設定
# ============================================================
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 9089
BUFFER_SIZE = 65536  # UDP バッファサイズ


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> dict:
    """ネストされた辞書をフラット化する。

    例: {"self_aircraft": {"x": 1}} → {"self_aircraft_x": 1}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class DataCollector:
    """UDP データ受信 → CSV 保存サーバー"""

    def __init__(self, host: str, port: int, output_path: str):
        self.host = host
        self.port = port
        self.output_path = Path(output_path)
        self.sock: socket.socket | None = None
        self.csv_file = None
        self.csv_writer = None
        self.header_written = False
        self.running = False
        self.frame_count = 0
        self.start_time = None

    def setup(self):
        """ソケットとCSVファイルを初期化"""
        # 出力ディレクトリ作成
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # UDP ソケット作成
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(1.0)  # 1秒タイムアウト（Ctrl+Cを受け付けるため）

        # CSV ファイルを開く
        self.csv_file = open(self.output_path, "w", newline="", encoding="utf-8")

        print(f"[DataCollector] Listening on {self.host}:{self.port}")
        print(f"[DataCollector] Output: {self.output_path}")

    def process_packet(self, data: bytes) -> dict | None:
        """受信パケットをパースしてフラット辞書として返す"""
        try:
            raw = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[DataCollector] Parse error: {e}")
            return None

        # タイムスタンプをローカル時刻で追加
        raw["recv_time"] = time.time()

        # フラット化
        flat = flatten_dict(raw)
        return flat

    def write_row(self, row: dict):
        """CSVに1行書き込む"""
        if not self.header_written:
            self.csv_writer = csv.DictWriter(
                self.csv_file, fieldnames=list(row.keys())
            )
            self.csv_writer.writeheader()
            self.header_written = True

        self.csv_writer.writerow(row)
        self.csv_file.flush()  # 安全のため毎行フラッシュ

    def run(self):
        """メインループ"""
        self.setup()
        self.running = True
        self.start_time = time.time()

        # シグナルハンドリング（Ctrl+C で安全に終了）
        def signal_handler(sig, frame):
            print("\n[DataCollector] Shutting down...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("[DataCollector] Waiting for DCS data... (Press Ctrl+C to stop)")

        while self.running:
            try:
                data, addr = self.sock.recvfrom(BUFFER_SIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            row = self.process_packet(data)
            if row is None:
                continue

            # 敵機が存在するフレームのみ記録
            if not row.get("has_enemy", False):
                continue

            self.write_row(row)
            self.frame_count += 1

            # 進捗表示（100フレームごと）
            if self.frame_count % 100 == 0:
                elapsed = time.time() - self.start_time
                fps = self.frame_count / elapsed if elapsed > 0 else 0
                print(
                    f"[DataCollector] {self.frame_count} frames collected "
                    f"({fps:.1f} fps)"
                )

        self.shutdown()

    def shutdown(self):
        """後片付け"""
        if self.csv_file:
            self.csv_file.close()
        if self.sock:
            self.sock.close()

        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"[DataCollector] Done. {self.frame_count} frames in {elapsed:.1f}s")
        print(f"[DataCollector] Saved to: {self.output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dogfight Supporter — DCS データ収集サーバー"
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"Bind host (default: {DEFAULT_HOST})"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"UDP port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: data/raw/session_YYYYMMDD_HHMMSS.csv)",
    )
    args = parser.parse_args()

    # デフォルト出力パス
    if args.out is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_root = Path(__file__).resolve().parent.parent
        args.out = str(project_root / "data" / "raw" / f"session_{timestamp}.csv")

    collector = DataCollector(args.host, args.port, args.out)
    collector.run()


if __name__ == "__main__":
    main()
