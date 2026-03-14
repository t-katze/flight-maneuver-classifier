"""
Dogfight Supporter — 推論モジュール（回帰版）

学習済みモデルを読み込み、特徴量ベクトルから
ガンキル有利度スコア (-1.0 ～ +1.0) を予測する。
"""

from pathlib import Path

import joblib
import numpy as np


class AdvantagePredictor:
    """ガンキル有利度スコアの予測器"""

    CATEGORIES = {
        "ADVANTAGE": "有利",
        "SLIGHT_ADV": "やや有利",
        "NEUTRAL": "互角",
        "SLIGHT_DIS": "やや不利",
        "DISADVANTAGE": "不利",
    }

    def __init__(self, model_path: str):
        """
        学習済みモデルを読み込む。

        Args:
            model_path: joblib ファイルパス
        """
        data = joblib.load(model_path)
        self.model = data["model"]
        self.scaler = data.get("scaler", None)
        self.feature_columns = data.get("feature_columns", [])
        print(f"[Predictor] Loaded model from: {model_path}")
        print(f"[Predictor] Task: {data.get('task', 'regression')}")
        print(f"[Predictor] Features: {len(self.feature_columns)}")

    def predict(self, features: dict | np.ndarray) -> dict:
        """
        特徴量から有利度スコアを予測する。

        Args:
            features: 特徴量辞書 (FeatureVector.to_dict()) or NumPy配列

        Returns:
            {
                "score": float (-1.0 ～ +1.0),
                "category": "ADVANTAGE" | "SLIGHT_ADV" | "NEUTRAL" | ...
                "category_ja": "有利" | "やや有利" | "互角" | ...
                "abs_score": float (0.0 ～ 1.0, 確信度),
            }
        """
        if isinstance(features, dict):
            X = np.array(
                [features.get(col, 0.0) for col in self.feature_columns],
                dtype=np.float64,
            ).reshape(1, -1)
        else:
            X = np.asarray(features, dtype=np.float64).reshape(1, -1)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if self.scaler is not None:
            X = self.scaler.transform(X)

        score = float(np.clip(self.model.predict(X)[0], -1.0, 1.0))

        # カテゴリ判定
        if score > 0.5:
            cat = "ADVANTAGE"
        elif score > 0.2:
            cat = "SLIGHT_ADV"
        elif score > -0.2:
            cat = "NEUTRAL"
        elif score > -0.5:
            cat = "SLIGHT_DIS"
        else:
            cat = "DISADVANTAGE"

        return {
            "score": score,
            "category": cat,
            "category_ja": self.CATEGORIES[cat],
            "abs_score": abs(score),
        }

    def predict_batch(self, features_list: list[dict]) -> list[dict]:
        """複数フレームをまとめて予測"""
        return [self.predict(f) for f in features_list]
