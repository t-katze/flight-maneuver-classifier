# Flight Maneuver Classifier — タスクリスト

## Planning
- [x] 既存コードベースの調査
- [x] 実装計画書の作成・レビュー依頼

## Implementation
- [x] [src/maneuver_feature_engine.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_feature_engine.py) — スライディングウィンドウ特徴量抽出
- [x] [src/maneuver_labeler.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_labeler.py) — ルールベース仮ラベル付与
- [x] [src/maneuver_classifier.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_classifier.py) — Random Forest 分類・評価
- [x] [src/maneuver_main.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_main.py) — エントリポイント (main関数)

## Verification
- [x] [tests/test_maneuver_feature_engine.py](file:///home/t-kat/src/flight-maneuver-classifier/tests/test_maneuver_feature_engine.py) — 特徴量抽出テスト
- [x] [tests/test_maneuver_labeler.py](file:///home/t-kat/src/flight-maneuver-classifier/tests/test_maneuver_labeler.py) — ルールベースラベルテスト
- [x] [tests/test_maneuver_classifier.py](file:///home/t-kat/src/flight-maneuver-classifier/tests/test_maneuver_classifier.py) — 分類パイプラインテスト
- [x] 既存テストとの整合性確認 (`pytest tests/` — 80 passed)
