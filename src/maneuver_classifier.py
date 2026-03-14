"""
Flight Maneuver Classifier — Random Forest 分類・評価

スライディングウィンドウ特徴量 + ルールベース仮ラベルを用いて
Random Forest で航空機機動を6クラスに分類し、評価指標を出力する。

出力:
    - Accuracy, Precision, Recall, F1-score
    - Confusion Matrix (画像)
    - 特徴量重要度 (画像)
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from maneuver_labeler import CLASS_NAMES, MANEUVER_CLASSES


# ============================================================
# 学習・評価
# ============================================================

def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    test_size: float = 0.2,
    n_estimators: int = 200,
    random_state: int = 42,
    output_dir: str | None = None,
) -> dict:
    """
    Random Forest で機動分類モデルを学習し、評価結果を返す。

    Args:
        X: 特徴量行列 (n_samples, n_features)
        y: ラベル配列 (n_samples,)  int 0–5
        feature_names: 特徴量名リスト
        test_size: テストデータ割合
        n_estimators: Random Forest のツリー数
        random_state: 乱数シード
        output_dir: 結果画像の保存先 (None なら保存しない)

    Returns:
        評価結果の辞書:
            accuracy, precision, recall, f1, classification_report,
            confusion_matrix, feature_importances, model
    """
    # NaN / inf 処理
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 学習/テスト分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    print(f"[ManeuverClassifier] Train: {len(X_train)}, Test: {len(X_test)}")

    # ---- Random Forest 学習 ----
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    # ---- 予測 ----
    y_pred = model.predict(X_test)

    # ---- 評価指標 ----
    present_labels = sorted(set(y_test) | set(y_pred))
    present_names = [MANEUVER_CLASSES.get(l, str(l)) for l in present_labels]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report = classification_report(
        y_test, y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=present_labels)

    # ---- コンソール出力 ----
    print(f"\n{'=' * 60}")
    print("Random Forest — 機動分類結果")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (weighted)")
    print(f"Recall:    {rec:.4f} (weighted)")
    print(f"F1-score:  {f1:.4f} (weighted)")
    print(f"\n--- Classification Report ---")
    print(report)

    # ---- 特徴量重要度 ----
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"--- 特徴量重要度 (Top 10) ---")
    for i in range(min(10, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {i + 1:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")

    # ---- 画像保存 ----
    if output_dir:
        _save_plots(cm, present_names, importances, feature_names,
                    sorted_idx, output_dir)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importances": dict(zip(feature_names, importances.tolist())),
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


# ============================================================
# 可視化
# ============================================================

def _save_plots(
    cm: np.ndarray,
    class_names: list[str],
    importances: np.ndarray,
    feature_names: list[str],
    sorted_idx: np.ndarray,
    output_dir: str,
):
    """Confusion Matrix と 特徴量重要度のプロットを保存する。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Confusion Matrix ----
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Flight Maneuver Classifier")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"[ManeuverClassifier] Confusion Matrix → {out / 'confusion_matrix.png'}")

    # ---- 特徴量重要度 ----
    n_show = min(15, len(sorted_idx))
    top_idx = sorted_idx[:n_show]
    top_names = [feature_names[i] for i in top_idx]
    top_imps = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(n_show), top_imps[::-1], color="steelblue")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top Feature Importances — Random Forest")
    fig.tight_layout()
    fig.savefig(out / "feature_importances.png", dpi=150)
    plt.close(fig)
    print(f"[ManeuverClassifier] Feature Importances → {out / 'feature_importances.png'}")


# ============================================================
# モデル保存/読込
# ============================================================

def save_model(model, feature_names: list[str], path: str):
    """Random Forest モデルを保存する。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_columns": feature_names,
        "task": "classification",
        "classes": MANEUVER_CLASSES,
    }, path)
    print(f"[ManeuverClassifier] Model saved → {path}")


def load_model(path: str) -> tuple:
    """モデルを読み込む。Returns (model, feature_names)."""
    data = joblib.load(path)
    return data["model"], data["feature_columns"]
