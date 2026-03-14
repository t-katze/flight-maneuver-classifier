"""
Dogfight Supporter — 回帰モデル学習スクリプト

ガンキル有利度スコアを予測する回帰モデルを学習する。
段階的アプローチ: リッジ回帰 → XGBoost Regressor

Usage:
    python train.py <labeled_data.csv>
    python train.py <labeled_data.csv> --model xgboost --output models/model.joblib
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 特徴量カラム名（feature_engine.py の FeatureVector に対応）
FEATURE_COLUMNS = [
    "distance",
    "aspect_angle",
    "antenna_train_angle",
    "heading_crossing_angle",
    "altitude_diff",
    "bearing",
    "self_specific_energy",
    "enemy_specific_energy",
    "delta_specific_energy",
    "self_speed",
    "enemy_speed",
    "speed_ratio",
    "self_turn_rate",
    "enemy_turn_rate",
    "relative_turn_rate",
    "self_g_load",
    "self_speed_rate",
    "distance_rate",
    "ata_rate",
    "self_ps_estimate",
]

SCORE_COLUMN = "score"


def load_data(csv_path: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """データ読み込みと特徴量/スコアの分離"""
    df = pd.read_csv(csv_path)

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    if SCORE_COLUMN not in df.columns:
        raise ValueError(f"Missing score column: {SCORE_COLUMN}")

    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    y = df[SCORE_COLUMN].values.astype(np.float64)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -1.0, 1.0)

    print(f"[Train] Loaded {len(df)} samples, {X.shape[1]} features")
    print(f"[Train] Score range: [{y.min():.3f}, {y.max():.3f}], "
          f"mean={y.mean():.3f}, std={y.std():.3f}")

    return df, X, y


def train_ridge(X_train, y_train, X_test, y_test):
    """リッジ回帰（ベースライン）"""
    print("\n" + "=" * 60)
    print("リッジ回帰 (ベースライン)")
    print("=" * 60)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    evaluate_model("Ridge", y_test, y_pred)

    importances = np.abs(model.coef_)
    print_feature_importance(importances, "係数の絶対値")

    return model, scaler


def train_xgboost(X_train, y_train, X_test, y_test):
    """XGBoost (GradientBoosting) 回帰モデルの学習"""
    print("\n" + "=" * 60)
    print("Gradient Boosting Regressor")
    print("=" * 60)

    try:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model_name = "XGBoost"
    except ImportError:
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model_name = "GradientBoosting (sklearn)"

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    evaluate_model(model_name, y_test, y_pred)

    importances = model.feature_importances_
    print_feature_importance(importances, "Feature Importance")

    return model, None


def evaluate_model(name: str, y_true, y_pred):
    """モデルの評価指標を出力"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n--- {name} 評価結果 ---")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"R²:    {r2:.4f}")

    # カテゴリ別精度（スコアを3分割）
    y_cat_true = np.where(y_true > 0.3, 1, np.where(y_true < -0.3, -1, 0))
    y_cat_pred = np.where(y_pred > 0.3, 1, np.where(y_pred < -0.3, -1, 0))
    cat_acc = np.mean(y_cat_true == y_cat_pred)
    print(f"カテゴリ精度 (ADV/NEU/DIS): {cat_acc:.4f}")


def print_feature_importance(importances: np.ndarray, title: str):
    """特徴量重要度を表示"""
    indices = np.argsort(importances)[::-1]
    print(f"\n--- 特徴量重要度 ({title}) ---")
    for i, idx in enumerate(indices):
        print(f"  {i + 1:2d}. {FEATURE_COLUMNS[idx]:30s} {importances[idx]:.4f}")


def save_model(
    model, scaler, model_path: str, metadata: dict | None = None
):
    """モデルとスケーラーを保存"""
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        "model": model,
        "scaler": scaler,
        "feature_columns": FEATURE_COLUMNS,
        "task": "regression",
        "target": "advantage_score",
    }
    joblib.dump(save_data, model_path)
    print(f"\n[Train] Model saved to: {model_path}")

    if metadata:
        meta_path = Path(model_path).with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"[Train] Metadata saved to: {meta_path}")


def plot_results(y_true, y_pred, output_dir: str):
    """評価結果の可視化"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 散布図: 予測 vs 実際
    axes[0].scatter(y_true, y_pred, alpha=0.3, s=5)
    axes[0].plot([-1, 1], [-1, 1], "r--", lw=1)
    axes[0].set_xlabel("True Score")
    axes[0].set_ylabel("Predicted Score")
    axes[0].set_title("Prediction vs Ground Truth")
    axes[0].set_xlim(-1.1, 1.1)
    axes[0].set_ylim(-1.1, 1.1)

    # 残差ヒストグラム
    residuals = y_pred - y_true
    axes[1].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residual (Predicted - True)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residual Distribution (MAE={np.mean(np.abs(residuals)):.3f})")

    fig.tight_layout()
    fig.savefig(output_path / "regression_results.png", dpi=150)
    plt.close(fig)

    print(f"[Train] Plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Dogfight Supporter — 回帰モデル学習")
    parser.add_argument("data", help="スコア付きCSVファイル")
    parser.add_argument(
        "--model",
        choices=["ridge", "xgboost", "both"],
        default="both",
        help="学習するモデル (default: both)",
    )
    parser.add_argument("--output", default=None, help="モデル保存先")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="テストデータ割合"
    )
    args = parser.parse_args()

    df, X, y = load_data(args.data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42
    )
    print(f"[Train] Train: {len(X_train)}, Test: {len(X_test)}")

    project_root = Path(__file__).resolve().parent.parent
    best_model = None
    best_scaler = None
    best_name = ""
    best_mae = float("inf")

    if args.model in ("ridge", "both"):
        lr_model, lr_scaler = train_ridge(X_train, y_train, X_test, y_test)
        y_pred = lr_model.predict(
            lr_scaler.transform(X_test) if lr_scaler else X_test
        )
        mae = mean_absolute_error(y_test, y_pred)
        if mae < best_mae:
            best_model, best_scaler, best_name, best_mae = (
                lr_model, lr_scaler, "ridge", mae
            )

    if args.model in ("xgboost", "both"):
        xgb_model, xgb_scaler = train_xgboost(X_train, y_train, X_test, y_test)
        y_pred = xgb_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        if mae < best_mae:
            best_model, best_scaler, best_name, best_mae = (
                xgb_model, xgb_scaler, "xgboost", mae
            )

    if best_model is not None:
        output_path = args.output or str(
            project_root / "models" / f"{best_name}_advantage.joblib"
        )
        metadata = {
            "model_type": best_name,
            "task": "regression",
            "target": "advantage_score (-1 to +1)",
            "mae": float(best_mae),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "feature_columns": FEATURE_COLUMNS,
        }
        save_model(best_model, best_scaler, output_path, metadata)

        if best_scaler:
            y_pred = best_model.predict(best_scaler.transform(X_test))
        else:
            y_pred = best_model.predict(X_test)

        plot_results(y_test, y_pred, str(project_root / "models" / "plots"))


if __name__ == "__main__":
    main()
