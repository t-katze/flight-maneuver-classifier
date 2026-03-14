"""
Dogfight Supporter — 特徴量エンジニアリング

DCS から取得した二機の座標・姿勢データから、
ドッグファイト戦術判定に必要な特徴量を算出する。

特徴量カテゴリ:
    1. 相対幾何学的特徴量 (AA, ATA, HCA, 距離, 高度差)
    2. エネルギー状態特徴量 (比エネルギー, Ps推定, エネルギー差)
    3. 動的特徴量 (ターンレート, G荷重, 速度変化率, 距離変化率)
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 物理定数
G = 9.80665  # 重力加速度 (m/s²)


@dataclass
class AircraftState:
    """単一フレームにおける航空機の状態"""

    x: float = 0.0  # ワールド座標 X (m)
    y: float = 0.0  # ワールド座標 Y = 高度 (m)
    z: float = 0.0  # ワールド座標 Z (m)
    heading: float = 0.0  # ヘディング (rad)
    pitch: float = 0.0  # ピッチ角 (rad)
    bank: float = 0.0  # バンク角 (rad)
    TAS: float = 0.0  # 真対気速度 (m/s)
    mach: float = 0.0  # マッハ数
    alt_asl: float = 0.0  # 海面高度 (m)
    AoA: float = 0.0  # 迎角 (rad)
    Gy: float = 0.0  # 垂直G荷重
    roll_rate: float = 0.0  # ロールレート (rad/s)
    pitch_rate: float = 0.0  # ピッチレート (rad/s)
    yaw_rate: float = 0.0  # ヨーレート (rad/s)
    throttle: float = 0.0  # スロットル (%)


@dataclass
class FeatureVector:
    """算出された戦術的特徴量ベクトル"""

    # 相対幾何学的特徴量
    distance: float = 0.0  # 二機間距離 (m)
    aspect_angle: float = 0.0  # AA: アスペクト角 (deg)
    antenna_train_angle: float = 0.0  # ATA: アンテナトレイン角 (deg)
    heading_crossing_angle: float = 0.0  # HCA: ヘディング交差角 (deg)
    altitude_diff: float = 0.0  # 高度差 (m, 正=自機が上)
    bearing: float = 0.0  # 自機から敵機への方位角 (deg)

    # エネルギー状態特徴量
    self_specific_energy: float = 0.0  # 自機比エネルギー (m)
    enemy_specific_energy: float = 0.0  # 敵機比エネルギー (m)
    delta_specific_energy: float = 0.0  # 比エネルギー差 (m)
    self_speed: float = 0.0  # 自機速度 (m/s)
    enemy_speed: float = 0.0  # 敵機速度 (m/s)
    speed_ratio: float = 0.0  # 速度比

    # 動的特徴量
    self_turn_rate: float = 0.0  # 自機ターンレート (deg/s)
    enemy_turn_rate: float = 0.0  # 敵機ターンレート (deg/s, 推定)
    relative_turn_rate: float = 0.0  # 相対ターンレート (deg/s)
    self_g_load: float = 0.0  # 自機G荷重
    self_speed_rate: float = 0.0  # 自機速度変化率 (m/s²)
    distance_rate: float = 0.0  # 距離変化率 (m/s, 正=離脱)
    ata_rate: float = 0.0  # ATA変化率 (deg/s)
    self_ps_estimate: float = 0.0  # 自機Ps推定値 (m/s)

    def to_dict(self) -> dict:
        """辞書に変換"""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    def to_array(self) -> np.ndarray:
        """NumPy配列に変換（モデル入力用）"""
        return np.array(list(self.to_dict().values()), dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        """特徴量名のリストを返す"""
        return list(FeatureVector.__dataclass_fields__.keys())


class FeatureEngine:
    """
    フレームごとの航空機データから戦術的特徴量を算出するエンジン。

    時系列微分（速度変化率、距離変化率等）のために直前フレームの状態を保持する。
    """

    def __init__(self):
        self._prev_self: AircraftState | None = None
        self._prev_enemy: AircraftState | None = None
        self._prev_features: FeatureVector | None = None
        self._prev_timestamp: float | None = None

    def reset(self):
        """状態をリセットする（新しいセッション開始時）"""
        self._prev_self = None
        self._prev_enemy = None
        self._prev_features = None
        self._prev_timestamp = None

    def compute(
        self, self_ac: AircraftState, enemy_ac: AircraftState, timestamp: float
    ) -> FeatureVector:
        """
        1フレーム分の特徴量を算出する。

        Args:
            self_ac: 自機の状態
            enemy_ac: 敵機の状態
            timestamp: シミュレーション時刻 (秒)

        Returns:
            算出された特徴量ベクトル
        """
        fv = FeatureVector()
        dt = 0.0
        if self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp
            if dt <= 0:
                dt = 1.0 / 30.0  # フォールバック

        # ==== 相対幾何学的特徴量 ====
        fv.distance = self._calc_distance(self_ac, enemy_ac)
        fv.aspect_angle = self._calc_aspect_angle(self_ac, enemy_ac)
        fv.antenna_train_angle = self._calc_ata(self_ac, enemy_ac)
        fv.heading_crossing_angle = self._calc_hca(self_ac, enemy_ac)
        fv.altitude_diff = self_ac.alt_asl - enemy_ac.alt_asl
        fv.bearing = self._calc_bearing(self_ac, enemy_ac)

        # ==== エネルギー状態特徴量 ====
        fv.self_specific_energy = self._calc_specific_energy(self_ac)
        fv.enemy_specific_energy = self._calc_specific_energy(enemy_ac)
        fv.delta_specific_energy = (
            fv.self_specific_energy - fv.enemy_specific_energy
        )
        fv.self_speed = self_ac.TAS
        fv.enemy_speed = enemy_ac.TAS if enemy_ac.TAS > 0 else self._estimate_speed(enemy_ac)
        fv.speed_ratio = (
            fv.self_speed / fv.enemy_speed if fv.enemy_speed > 1.0 else 1.0
        )

        # ==== 動的特徴量 ====
        fv.self_turn_rate = self._calc_turn_rate_from_state(self_ac)
        fv.enemy_turn_rate = self._estimate_enemy_turn_rate(enemy_ac, dt)
        fv.relative_turn_rate = fv.self_turn_rate - fv.enemy_turn_rate
        fv.self_g_load = self_ac.Gy

        # 微分特徴量（前フレームが必要）
        if self._prev_self is not None and dt > 0:
            fv.self_speed_rate = (self_ac.TAS - self._prev_self.TAS) / dt
            prev_dist = self._calc_distance(self._prev_self, self._prev_enemy)
            fv.distance_rate = (fv.distance - prev_dist) / dt

            if self._prev_features is not None:
                fv.ata_rate = (
                    fv.antenna_train_angle - self._prev_features.antenna_train_angle
                ) / dt

            # Ps推定: 比エネルギーの時間変化率
            prev_es = self._calc_specific_energy(self._prev_self)
            fv.self_ps_estimate = (fv.self_specific_energy - prev_es) / dt
        else:
            fv.self_speed_rate = 0.0
            fv.distance_rate = 0.0
            fv.ata_rate = 0.0
            fv.self_ps_estimate = 0.0

        # 状態保存
        self._prev_self = self_ac
        self._prev_enemy = enemy_ac
        self._prev_features = fv
        self._prev_timestamp = timestamp

        return fv

    def compute_self_only(
        self, self_ac: AircraftState, timestamp: float
    ) -> FeatureVector:
        """
        自機データのみから特徴量を算出する（マルチプレイ用）。

        マルチプレイではクライアント側で LoGetWorldObjects() が使用不可のため、
        敵機データなしで判定可能な特徴量のみを算出する。

        相対特徴量（AA, ATA, HCA, 距離等）は 0 に設定される。

        Args:
            self_ac: 自機の状態
            timestamp: シミュレーション時刻 (秒)

        Returns:
            算出された特徴量ベクトル（自機関連のみ有効）
        """
        fv = FeatureVector()
        dt = 0.0
        if self._prev_timestamp is not None:
            dt = timestamp - self._prev_timestamp
            if dt <= 0:
                dt = 1.0 / 30.0

        # 相対特徴量は 0（敵データなし）
        # fv.distance, fv.aspect_angle, etc. は 0.0 のまま

        # ==== エネルギー状態特徴量（自機のみ） ====
        fv.self_specific_energy = self._calc_specific_energy(self_ac)
        fv.self_speed = self_ac.TAS
        fv.speed_ratio = 1.0  # 不明

        # ==== 動的特徴量（自機のみ） ====
        fv.self_turn_rate = self._calc_turn_rate_from_state(self_ac)
        fv.self_g_load = self_ac.Gy

        # 微分特徴量
        if self._prev_self is not None and dt > 0:
            fv.self_speed_rate = (self_ac.TAS - self._prev_self.TAS) / dt
            prev_es = self._calc_specific_energy(self._prev_self)
            fv.self_ps_estimate = (fv.self_specific_energy - prev_es) / dt

        # 状態保存
        self._prev_self = self_ac
        self._prev_timestamp = timestamp

        return fv

    # ============================================================
    # 幾何学計算
    # ============================================================

    @staticmethod
    def _calc_distance(a: AircraftState, b: AircraftState) -> float:
        """3D距離を算出"""
        dx = a.x - b.x
        dy = a.y - b.y
        dz = a.z - b.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _calc_bearing(self_ac: AircraftState, enemy_ac: AircraftState) -> float:
        """自機から敵機への方位角 (deg, 0=北, 時計回り)"""
        dx = enemy_ac.x - self_ac.x
        dz = enemy_ac.z - self_ac.z
        bearing = math.degrees(math.atan2(dz, dx))
        return bearing % 360.0

    @staticmethod
    def _calc_aspect_angle(
        self_ac: AircraftState, enemy_ac: AircraftState
    ) -> float:
        """
        アスペクト角 (AA): 敵機の後方を0°として、自機が敵のどの方向にいるか。
        0° = 敵の真後ろ (6 o'clock)
        180° = 敵の真正面 (12 o'clock)
        """
        # 敵から自機への方位
        dx = self_ac.x - enemy_ac.x
        dz = self_ac.z - enemy_ac.z
        bearing_to_self = math.atan2(dz, dx)

        # 敵のヘディングとの差
        delta = bearing_to_self - enemy_ac.heading
        aa = abs(math.degrees(delta))

        # 正規化: 0°-180°
        if aa > 180.0:
            aa = 360.0 - aa
        return aa

    @staticmethod
    def _calc_ata(self_ac: AircraftState, enemy_ac: AircraftState) -> float:
        """
        アンテナトレイン角 (ATA): 自機のノーズから敵機への角度。
        0° = 敵が自機の真正面
        180° = 敵が自機の真後ろ
        """
        dx = enemy_ac.x - self_ac.x
        dz = enemy_ac.z - self_ac.z
        bearing_to_enemy = math.atan2(dz, dx)

        delta = bearing_to_enemy - self_ac.heading
        ata = abs(math.degrees(delta))

        if ata > 180.0:
            ata = 360.0 - ata
        return ata

    @staticmethod
    def _calc_hca(self_ac: AircraftState, enemy_ac: AircraftState) -> float:
        """
        ヘディング交差角 (HCA): 二機のヘディングの差。
        0° = 同方向
        180° = 正面からの対面
        """
        delta = abs(math.degrees(self_ac.heading - enemy_ac.heading))
        if delta > 180.0:
            delta = 360.0 - delta
        return delta

    # ============================================================
    # エネルギー計算
    # ============================================================

    @staticmethod
    def _calc_specific_energy(ac: AircraftState) -> float:
        """
        比エネルギー (Specific Energy): E_s = h + V² / (2g)
        単位: m（エネルギー高度）
        """
        return ac.alt_asl + (ac.TAS ** 2) / (2.0 * G)

    # ============================================================
    # ターンレート計算
    # ============================================================

    @staticmethod
    def _calc_turn_rate_from_state(ac: AircraftState) -> float:
        """
        G荷重とバンク角からターンレートを算出。
        TR = g × √(n² - 1) / V  (rad/s → deg/s)

        n = G荷重 (Ny)
        """
        n = abs(ac.Gy)
        if n <= 1.0 or ac.TAS < 30.0:
            return 0.0

        tr_rad = G * math.sqrt(n * n - 1.0) / ac.TAS
        return math.degrees(tr_rad)

    def _estimate_enemy_turn_rate(
        self, enemy_ac: AircraftState, dt: float
    ) -> float:
        """
        敵機のターンレートをヘディング変化率から推定。
        DCS Export API では敵機のG荷重が取得できないため。
        """
        if self._prev_enemy is None or dt <= 0:
            return 0.0

        dh = enemy_ac.heading - self._prev_enemy.heading
        # -π～πに正規化
        while dh > math.pi:
            dh -= 2.0 * math.pi
        while dh < -math.pi:
            dh += 2.0 * math.pi

        return abs(math.degrees(dh / dt))

    @staticmethod
    def _estimate_speed(ac: AircraftState) -> float:
        """
        敵機の速度を座標変化から推定（TASが取得できない場合のフォールバック）
        注: 1フレームでは推定不可。最低2フレーム必要。
        """
        # フォールバック: マッハ数から推定（海面の音速: ~340 m/s）
        if ac.mach > 0:
            return ac.mach * 340.0
        return 200.0  # デフォルト推定値


def process_csv(
    input_path: str,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    CSV ファイルからデータを読み込み、特徴量を算出して新しい CSV に保存する。

    Args:
        input_path: data_collector.py で保存した生データCSV
        output_path: 特徴量付きCSVの保存先（Noneなら DataFrame のみ返す）

    Returns:
        特徴量付き DataFrame
    """
    df = pd.read_csv(input_path)

    engine = FeatureEngine()
    features_list = []

    for _, row in df.iterrows():
        # 自機の状態を構築
        self_ac = AircraftState(
            x=row.get("self_aircraft_x", 0),
            y=row.get("self_aircraft_y", 0),
            z=row.get("self_aircraft_z", 0),
            heading=row.get("self_aircraft_heading", 0),
            pitch=row.get("self_aircraft_pitch", 0),
            bank=row.get("self_aircraft_bank", 0),
            TAS=row.get("self_aircraft_TAS", 0),
            mach=row.get("self_aircraft_mach", 0),
            alt_asl=row.get("self_aircraft_alt_asl", 0),
            AoA=row.get("self_aircraft_AoA", 0),
            Gy=row.get("self_aircraft_Gy", 0),
            roll_rate=row.get("self_aircraft_roll_rate", 0),
            pitch_rate=row.get("self_aircraft_pitch_rate", 0),
            yaw_rate=row.get("self_aircraft_yaw_rate", 0),
            throttle=row.get("self_aircraft_throttle", 0),
        )

        # 敵機の状態を構築
        enemy_ac = AircraftState(
            x=row.get("enemy_aircraft_x", 0),
            y=row.get("enemy_aircraft_y", 0),
            z=row.get("enemy_aircraft_z", 0),
            heading=row.get("enemy_aircraft_heading", 0),
            pitch=row.get("enemy_aircraft_pitch", 0),
            bank=row.get("enemy_aircraft_bank", 0),
            TAS=row.get("enemy_aircraft_TAS", 0),  # 敵は0の可能性あり
            mach=row.get("enemy_aircraft_mach", 0),
            alt_asl=row.get("enemy_aircraft_alt", row.get("enemy_aircraft_y", 0)),
        )

        timestamp = row.get("timestamp", row.get("model_time", 0))
        fv = engine.compute(self_ac, enemy_ac, timestamp)
        features_list.append(fv.to_dict())

    features_df = pd.DataFrame(features_list)

    # 元データと結合
    result = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    if output_path:
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"[FeatureEngine] Saved {len(result)} rows to {output_path}")

    return result


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python feature_engine.py <input.csv> [output.csv]")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None

    if out is None:
        p = Path(inp)
        out = str(p.parent.parent / "processed" / f"{p.stem}_features.csv")

    result_df = process_csv(inp, out)
    print(f"\n[FeatureEngine] Feature columns:")
    for name in FeatureVector.feature_names():
        print(f"  - {name}")
    print(f"\nShape: {result_df.shape}")
