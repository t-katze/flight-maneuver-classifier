"""
Dogfight Supporter — 撃墜結果ベース有利度ラベラー

ACMI パーサーが検出した実際の撃墜イベント（KillEvent）を基に、
各フレームに有利度スコア (-1.0 ～ +1.0) を付与する。

ラベリングロジック:
    1. 撃墜イベント発生 → キラー/ビクティムを特定
    2. 撃墜前 N 秒間のフレームに、時間減衰スコアを付与
       - キラー側: +exp(-t/τ)  (撃墜に近づくほど +1.0)
       - ビクティム側: -exp(-t/τ)  (撃墜に近づくほど -1.0)
    3. 撃墜に関与しないフレームは 0.0

これにより、モデルは「ガンキルに至る状況」を学習する。
操縦者の技量差はデータ収集時に自然にカバーされる。

Usage:
    python labeler.py <acmi_file> [-o output.csv]
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from acmi_parser import (
    ACMIParser,
    KillEvent,
    DogfightExtractor,
    pairs_to_dataframe,
)


class OutcomeLabeler:
    """
    撃墜結果ベースのラベラー。

    撃墜イベント前のフレームに指数減衰スコアを付与し、
    回帰モデルの教師データを生成する。
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {
            # スコアの時間減衰パラメータ
            "lookback_seconds": 15.0,     # 撃墜前何秒のフレームにスコアを付与
            "decay_tau": 5.0,             # 減衰定数 (秒)。小さい=急減衰

            # ドッグファイト検出パラメータ
            "max_distance": 10000.0,      # ペア検出距離 (m)
        }

    def label_from_acmi(
        self, acmi_path: str, output_path: str | None = None,
    ) -> pd.DataFrame:
        """
        ACMI ファイルから撃墜イベントを検出し、
        ドッグファイトペアにスコアを付与する。

        Returns:
            スコア付き DataFrame
        """
        # ---- パース ----
        parser = ACMIParser()
        frames, kill_events = parser.parse_file(acmi_path)

        print(f"[Labeler] Title: {parser.title}")
        print(f"[Labeler] Kill events: {len(kill_events)}")

        if not kill_events:
            print("[Labeler] ⚠ 撃墜イベントが見つかりません。")
            print("[Labeler]   ACMI ファイルにドッグファイトの撃墜が含まれていることを確認してください。")

        # ---- ドッグファイトペア抽出 ----
        extractor = DogfightExtractor(
            max_distance=self.config["max_distance"]
        )
        pairs = extractor.extract_pairs(frames)

        if not pairs:
            print("[Labeler] No dogfight pairs found")
            return pd.DataFrame()

        # ---- DataFrame に変換 ----
        df = pairs_to_dataframe(pairs)

        # ---- スコア付与 ----
        df["score"] = 0.0
        df = self._apply_kill_scores(df, kill_events)

        # ---- 統計 ----
        self._print_stats(df)

        # ---- 保存 ----
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"[Labeler] Saved to: {output_path}")

        return df

    def _apply_kill_scores(
        self, df: pd.DataFrame, kill_events: list[KillEvent],
    ) -> pd.DataFrame:
        """
        撃墜イベントを基にフレームにスコアを付与する。
        """
        lookback = self.config["lookback_seconds"]
        tau = self.config["decay_tau"]

        for event in kill_events:
            kill_time = event.time

            for idx, row in df.iterrows():
                ts = row["timestamp"]

                # 撃墜前 lookback 秒以内のフレームのみ
                time_before_kill = kill_time - ts
                if time_before_kill < 0 or time_before_kill > lookback:
                    continue

                # 指数減衰スコア: 撃墜直前 → 1.0、遠い過去 → 0.0
                decay = math.exp(-time_before_kill / tau)

                # このフレームにどちらの機体が関与しているかチェック
                self_name = str(row.get("self_aircraft_name", ""))
                enemy_name = str(row.get("enemy_aircraft_name", ""))

                # キラー側のフレーム → +スコア
                if self_name and self_name in event.killer_name:
                    df.at[idx, "score"] = max(df.at[idx, "score"], decay)
                elif enemy_name and enemy_name in event.killer_name:
                    df.at[idx, "score"] = min(df.at[idx, "score"], -decay)

                # ビクティム側のフレーム → -スコア
                if self_name and self_name in event.victim_name:
                    df.at[idx, "score"] = min(df.at[idx, "score"], -decay)
                elif enemy_name and enemy_name in event.victim_name:
                    df.at[idx, "score"] = max(df.at[idx, "score"], decay)

        return df

    def _print_stats(self, df: pd.DataFrame):
        """統計情報を出力"""
        scores = df["score"].values
        n = len(scores)
        n_pos = np.sum(scores > 0.1)
        n_neg = np.sum(scores < -0.1)
        n_zero = n - n_pos - n_neg

        print(f"\n[Labeler] === ラベリング結果 ===")
        print(f"  総フレーム数:  {n}")
        print(f"  有利 (>0.1):   {n_pos} ({n_pos/n*100:.1f}%)")
        print(f"  互角:          {n_zero} ({n_zero/n*100:.1f}%)")
        print(f"  不利 (<-0.1):  {n_neg} ({n_neg/n*100:.1f}%)")
        if n_pos + n_neg > 0:
            labeled = scores[np.abs(scores) > 0.1]
            print(f"  ラベル付きスコア: mean={np.mean(labeled):.3f}, "
                  f"std={np.std(labeled):.3f}")


# ============================================================
# スタンドアロン機能: 特徴量算出 + ラベリング一括実行
# ============================================================

def generate_training_data(
    acmi_path: str,
    output_path: str,
    max_distance: float = 10000.0,
) -> pd.DataFrame:
    """
    ACMI → 特徴量算出 → ラベリング を一括実行するメインエントリポイント。

    これにより、feature_engine.py を別途実行する必要がなくなる。
    """
    labeler = OutcomeLabeler(config={
        "lookback_seconds": 15.0,
        "decay_tau": 5.0,
        "max_distance": max_distance,
    })

    df = labeler.label_from_acmi(acmi_path, output_path)
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Dogfight Supporter — 撃墜結果ベースラベラー"
    )
    parser.add_argument("acmi_file", help="TacView ACMI ファイルパス")
    parser.add_argument("-o", "--output", default=None, help="出力 CSV パス")
    parser.add_argument(
        "--max-distance", type=float, default=10000,
        help="ドッグファイト判定距離 (m, default: 10000)",
    )
    parser.add_argument(
        "--lookback", type=float, default=15.0,
        help="撃墜前のスコア付与期間 (秒, default: 15.0)",
    )
    parser.add_argument(
        "--tau", type=float, default=5.0,
        help="スコア減衰定数 (秒, default: 5.0)",
    )
    args = parser.parse_args()

    if args.output is None:
        p = Path(args.acmi_file)
        args.output = str(p.parent / f"{p.stem}_labeled.csv")

    labeler = OutcomeLabeler(config={
        "lookback_seconds": args.lookback,
        "decay_tau": args.tau,
        "max_distance": args.max_distance,
    })

    labeler.label_from_acmi(args.acmi_file, args.output)
