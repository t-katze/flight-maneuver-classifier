# Flight Maneuver Classifier — タスクリスト

## Planning
- [x] 既存コードベースの調査
- [/] 実装計画書の作成・レビュー依頼

## Implementation
- [ ] `src/maneuver_feature_engine.py` — スライディングウィンドウ特徴量抽出
- [ ] `src/maneuver_labeler.py` — ルールベース仮ラベル付与
- [ ] `src/maneuver_classifier.py` — Random Forest 分類・評価
- [ ] `src/maneuver_main.py` — エントリポイント (main関数)

## Verification
- [ ] `tests/test_maneuver_feature_engine.py` — 特徴量抽出テスト
- [ ] `tests/test_maneuver_labeler.py` — ルールベースラベルテスト
- [ ] `tests/test_maneuver_classifier.py` — 分類パイプラインテスト
- [ ] 既存テストとの整合性確認 (`pytest tests/`)
