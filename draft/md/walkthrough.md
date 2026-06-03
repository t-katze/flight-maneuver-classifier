# Walkthrough — Flight Maneuver Classifier

## 概要

Tacview ACMI ログから航空機機動を6クラスに分類するパイプラインを、既存の Dogfight Supporter プロジェクトに追加しました。

## 新規ファイル

| ファイル | 役割 |
|---------|------|
| [maneuver_feature_engine.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_feature_engine.py) | ACMI → 時系列 → unwrap/補間 → 5秒窓で23特徴量抽出 |
| [maneuver_labeler.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_labeler.py) | ルールベース6クラス仮ラベル付与 |
| [maneuver_classifier.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_classifier.py) | Random Forest 学習・評価・可視化・モデル保存 |
| [maneuver_main.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_main.py) | CLI エントリポイント |

## 処理パイプライン

```mermaid
flowchart LR
    A["ACMI ファイル"] --> B["ACMIParser\n(既存)"]
    B --> C["航空機ごと\n時系列 DataFrame"]
    C --> D["前処理\n(unwrap/補間)"]
    D --> E["5秒窓\n特徴量抽出"]
    E --> F["ルールベース\n仮ラベル"]
    F --> G["Random Forest\n学習・評価"]
    G --> H["結果出力\n(CSV/画像)"]
```

## 使い方

```bash
# 基本実行
python src/maneuver_main.py flight.acmi

# オプション付き
python src/maneuver_main.py flight.acmi --window 5 --step 1 --output-dir results/

# 航空機一覧のみ
python src/maneuver_main.py flight.acmi --list-aircraft

# 閾値カスタマイズ
python src/maneuver_main.py flight.acmi --roll-std-th 20 --heading-delta-th 8
```

## 特徴量一覧 (23個/窓)

各チャネル (altitude, speed, heading, pitch, roll) × 4統計量 (mean, std, delta, slope) = 20 + 派生3 (altitude_rate, heading_rate, roll_abs_mean)

## テスト結果

```
80 passed in 4.28s
```

- 既存テスト 46件: 全パス (既存コードへの影響なし)
- 新規テスト 34件: 全パス
  - [test_maneuver_feature_engine.py](file:///home/t-kat/src/flight-maneuver-classifier/tests/test_maneuver_feature_engine.py) (13件): unwrap, 補間, 窓数, 特徴量名, エッジケース
  - [test_maneuver_labeler.py](file:///home/t-kat/src/flight-maneuver-classifier/tests/test_maneuver_labeler.py) (14件): 6クラス判定, 優先度, カスタム閾値
  - [test_maneuver_classifier.py](file:///home/t-kat/src/flight-maneuver-classifier/tests/test_maneuver_classifier.py) (10件): 合成データ学習, 評価指標, NaN処理, モデル保存/読込
