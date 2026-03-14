"""
OutcomeLabeler / KillEvent のユニットテスト
"""

import math
import pytest
import pandas as pd
import numpy as np
from dataclasses import dataclass

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from labeler import OutcomeLabeler
from acmi_parser import KillEvent


class TestOutcomeLabeler:
    """OutcomeLabeler のテスト"""

    def setup_method(self):
        self.labeler = OutcomeLabeler(config={
            "lookback_seconds": 10.0,
            "decay_tau": 3.0,
            "max_distance": 10000.0,
        })

    def _make_df(self, timestamps, self_names, enemy_names):
        """テスト用 DataFrame を作成"""
        return pd.DataFrame({
            "timestamp": timestamps,
            "self_aircraft_name": self_names,
            "enemy_aircraft_name": enemy_names,
            "score": [0.0] * len(timestamps),
        })

    def test_killer_gets_positive_score(self):
        """キラー側のフレームは正のスコアを得る"""
        df = self._make_df(
            timestamps=[85.0, 90.0, 95.0, 100.0],
            self_names=["F-16C"] * 4,
            enemy_names=["MiG-29"] * 4,
        )
        kill_events = [KillEvent(
            time=100.0,
            victim_id="2", victim_name="MiG-29",
            killer_id="1", killer_name="F-16C",
            victim_coalition="Red", killer_coalition="Blue",
        )]
        result = self.labeler._apply_kill_scores(df, kill_events)
        # 撃墜時刻のフレームは最大スコア
        assert result.iloc[3]["score"] > 0.9

    def test_victim_gets_negative_score(self):
        """ビクティム側のフレームは負のスコアを得る"""
        df = self._make_df(
            timestamps=[85.0, 90.0, 95.0, 100.0],
            self_names=["MiG-29"] * 4,  # 自機がビクティム
            enemy_names=["F-16C"] * 4,
        )
        kill_events = [KillEvent(
            time=100.0,
            victim_id="2", victim_name="MiG-29",
            killer_id="1", killer_name="F-16C",
            victim_coalition="Red", killer_coalition="Blue",
        )]
        result = self.labeler._apply_kill_scores(df, kill_events)
        # 撃墜時刻のフレームは最少スコア
        assert result.iloc[3]["score"] < -0.9

    def test_time_decay(self):
        """撃墜から遠いフレームほどスコアの絶対値が小さい"""
        df = self._make_df(
            timestamps=[85.0, 90.0, 95.0, 100.0],
            self_names=["F-16C"] * 4,
            enemy_names=["MiG-29"] * 4,
        )
        kill_events = [KillEvent(
            time=100.0,
            victim_id="2", victim_name="MiG-29",
            killer_id="1", killer_name="F-16C",
            victim_coalition="Red", killer_coalition="Blue",
        )]
        result = self.labeler._apply_kill_scores(df, kill_events)
        # t=100 > t=95 > t=90 > t=85
        scores = result["score"].tolist()
        assert scores[3] > scores[2] > scores[1] > scores[0]

    def test_no_kill_events_all_zero(self):
        """撃墜イベントがなければ全フレーム 0"""
        df = self._make_df(
            timestamps=[10.0, 20.0, 30.0],
            self_names=["F-16C"] * 3,
            enemy_names=["MiG-29"] * 3,
        )
        result = self.labeler._apply_kill_scores(df, [])
        assert all(s == 0.0 for s in result["score"])

    def test_outside_lookback_no_score(self):
        """lookback 期間外のフレームはスコアなし"""
        df = self._make_df(
            timestamps=[50.0, 89.0, 95.0, 100.0],
            self_names=["F-16C"] * 4,
            enemy_names=["MiG-29"] * 4,
        )
        kill_events = [KillEvent(
            time=100.0,
            victim_id="2", victim_name="MiG-29",
            killer_id="1", killer_name="F-16C",
            victim_coalition="Red", killer_coalition="Blue",
        )]
        result = self.labeler._apply_kill_scores(df, kill_events)
        # t=50 は lookback(10秒) 外 → スコア 0
        assert result.iloc[0]["score"] == 0.0
        # t=95 は lookback 内 → スコア > 0
        assert result.iloc[2]["score"] > 0

    def test_score_range(self):
        """スコアは -1 ～ +1 の範囲"""
        df = self._make_df(
            timestamps=[99.0, 99.5, 100.0],
            self_names=["F-16C"] * 3,
            enemy_names=["MiG-29"] * 3,
        )
        kill_events = [KillEvent(
            time=100.0,
            victim_id="2", victim_name="MiG-29",
            killer_id="1", killer_name="F-16C",
            victim_coalition="Red", killer_coalition="Blue",
        )]
        result = self.labeler._apply_kill_scores(df, kill_events)
        for s in result["score"]:
            assert -1.0 <= s <= 1.0

    def test_multiple_kill_events(self):
        """複数の撃墜イベント"""
        df = self._make_df(
            timestamps=[45.0, 50.0, 95.0, 100.0],
            self_names=["F-16C", "F-16C", "MiG-29", "MiG-29"],
            enemy_names=["MiG-29", "MiG-29", "F-16C", "F-16C"],
        )
        kill_events = [
            KillEvent(  # F-16C が MiG-29 を撃墜
                time=50.0,
                victim_id="2", victim_name="MiG-29",
                killer_id="1", killer_name="F-16C",
                victim_coalition="Red", killer_coalition="Blue",
            ),
            KillEvent(  # MiG-29 が F-16C を撃墜
                time=100.0,
                victim_id="1", victim_name="F-16C",
                killer_id="3", killer_name="MiG-29",
                victim_coalition="Blue", killer_coalition="Red",
            ),
        ]
        result = self.labeler._apply_kill_scores(df, kill_events)
        # 最初の撃墜: F-16Cが有利
        assert result.iloc[1]["score"] > 0
        # 2番目の撃墜: MiG-29が有利 (=F-16Cが不利)
        assert result.iloc[3]["score"] > 0  # MiG-29 as self → positive

    def test_custom_config(self):
        """設定変更可能"""
        labeler = OutcomeLabeler(config={
            "lookback_seconds": 5.0,
            "decay_tau": 1.0,
            "max_distance": 5000.0,
        })
        assert labeler.config["lookback_seconds"] == 5.0


class TestKillEvent:
    """KillEvent のテスト"""

    def test_creation(self):
        """KillEvent が正しく作成される"""
        event = KillEvent(
            time=100.0,
            victim_id="ABC", victim_name="MiG-29",
            killer_id="DEF", killer_name="F-16C",
            victim_coalition="Red", killer_coalition="Blue",
        )
        assert event.time == 100.0
        assert event.victim_name == "MiG-29"
        assert event.killer_name == "F-16C"
