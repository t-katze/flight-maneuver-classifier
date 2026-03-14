"""
Flight Maneuver Classifier — maneuver_classifier のテスト
"""

import sys
from pathlib import Path
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from maneuver_classifier import train_and_evaluate, save_model, load_model
from maneuver_feature_engine import FEATURE_COLUMNS


# ============================================================
# ヘルパー: 合成データ生成
# ============================================================

def _generate_synthetic_data(n_per_class: int = 100, n_classes: int = 6):
    """
    学習テスト用の合成データを生成する。

    各クラスのデータは特徴量空間で分離可能なように設計。
    """
    rng = np.random.RandomState(42)
    n_features = len(FEATURE_COLUMNS)
    X_all, y_all = [], []

    for cls_id in range(n_classes):
        # クラスごとに異なるセンターとノイズ
        center = rng.randn(n_features) * (cls_id + 1)
        noise = rng.randn(n_per_class, n_features) * 0.5
        X = noise + center
        y = np.full(n_per_class, cls_id, dtype=int)
        X_all.append(X)
        y_all.append(y)

    return np.vstack(X_all), np.concatenate(y_all)


# ============================================================
# テスト
# ============================================================

class TestTrainAndEvaluate:
    """train_and_evaluate のテスト"""

    def test_returns_dict(self):
        """結果が辞書で返る"""
        X, y = _generate_synthetic_data(n_per_class=50)
        result = train_and_evaluate(X, y, FEATURE_COLUMNS)
        assert isinstance(result, dict)

    def test_contains_metrics(self):
        """accuracy, precision, recall, f1 が含まれる"""
        X, y = _generate_synthetic_data(n_per_class=50)
        result = train_and_evaluate(X, y, FEATURE_COLUMNS)
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in result, f"Missing key: {key}"
            assert 0.0 <= result[key] <= 1.0

    def test_contains_classification_report(self):
        """classification_report (文字列) が含まれる"""
        X, y = _generate_synthetic_data(n_per_class=50)
        result = train_and_evaluate(X, y, FEATURE_COLUMNS)
        assert "classification_report" in result
        assert isinstance(result["classification_report"], str)
        assert len(result["classification_report"]) > 0

    def test_contains_confusion_matrix(self):
        """confusion_matrix (ndarray) が含まれる"""
        X, y = _generate_synthetic_data(n_per_class=50)
        result = train_and_evaluate(X, y, FEATURE_COLUMNS)
        assert "confusion_matrix" in result
        cm = result["confusion_matrix"]
        assert isinstance(cm, np.ndarray)
        assert cm.shape[0] == cm.shape[1]  # 正方行列

    def test_contains_feature_importances(self):
        """feature_importances が含まれる"""
        X, y = _generate_synthetic_data(n_per_class=50)
        result = train_and_evaluate(X, y, FEATURE_COLUMNS)
        assert "feature_importances" in result
        imps = result["feature_importances"]
        assert isinstance(imps, dict)
        assert len(imps) == len(FEATURE_COLUMNS)
        # 重要度の合計 ≈ 1.0
        total = sum(imps.values())
        assert abs(total - 1.0) < 0.01

    def test_high_accuracy_synthetic(self):
        """合成データ (分離可能) で高精度"""
        X, y = _generate_synthetic_data(n_per_class=100)
        result = train_and_evaluate(X, y, FEATURE_COLUMNS)
        assert result["accuracy"] > 0.85, f"Accuracy too low: {result['accuracy']}"

    def test_with_nan(self):
        """NaN を含むデータでもクラッシュしない"""
        X, y = _generate_synthetic_data(n_per_class=50)
        X[0, 0] = np.nan
        X[10, 5] = np.inf
        result = train_and_evaluate(X, y, FEATURE_COLUMNS)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_model_in_result(self):
        """学習済みモデルが結果に含まれる"""
        X, y = _generate_synthetic_data(n_per_class=50)
        result = train_and_evaluate(X, y, FEATURE_COLUMNS)
        assert "model" in result
        assert result["model"] is not None

    def test_save_plots(self):
        """output_dir 指定で画像が保存される"""
        X, y = _generate_synthetic_data(n_per_class=50)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = train_and_evaluate(X, y, FEATURE_COLUMNS, output_dir=tmpdir)
            cm_path = Path(tmpdir) / "confusion_matrix.png"
            fi_path = Path(tmpdir) / "feature_importances.png"
            assert cm_path.exists(), "confusion_matrix.png not found"
            assert fi_path.exists(), "feature_importances.png not found"


class TestModelPersistence:
    """モデル保存/読込のテスト"""

    def test_save_and_load(self):
        """モデルを保存して読み込める"""
        X, y = _generate_synthetic_data(n_per_class=50)
        result = train_and_evaluate(X, y, FEATURE_COLUMNS)
        model = result["model"]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "test_model.joblib")
            save_model(model, FEATURE_COLUMNS, path)
            assert Path(path).exists()

            loaded_model, loaded_features = load_model(path)
            assert loaded_features == FEATURE_COLUMNS

            # 予測が同じ
            y_pred_orig = model.predict(X[:10])
            y_pred_loaded = loaded_model.predict(X[:10])
            np.testing.assert_array_equal(y_pred_orig, y_pred_loaded)
