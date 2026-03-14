# Tacview ログからの短時間窓ベース航空機機動分類

既存の Dogfight Supporter プロジェクトに、航空機の機動（Level Flight, Climb, Descent, Left/Right Turn, Roll Maneuver）を分類するパイプラインを追加する。既存の [acmi_parser.py](file:///home/t-kat/src/flight-maneuver-classifier/src/acmi_parser.py) を再利用し、ACMIデータを各航空機ごとの時系列 DataFrame に変換した上で、スライディングウィンドウ特徴量 → ルールベース仮ラベル → Random Forest 分類 の流れを構築する。

## 意見・設計判断

> [!IMPORTANT]
> **ルールベースラベルの信頼性**: 教師データをルールベースで自動生成するため、ルールの閾値がモデル性能に直結します。ルールの閾値はコマンドライン引数で調整可能にしますが、実データで必ず確認してください。
>
> **クラスの拡張性**: 現在の6クラスに加えて、将来的に「Barrel Roll」「Split-S」「Chandelle」等のACM機動を追加する場合は、特徴量窓を10秒程度に拡張し、時系列形状マッチング（DTW等）を検討する価値があります。
>
> **Heading/Rollの角度 unwrap**: ACMI では yaw(heading) が 0°/360° 境界を跨ぐことがあり、差分計算で不連続が生じます。`np.unwrap` を適用し、Roll も同様に処理します。
>
> **ログ単位分割**: 複数のACMIファイルを結合する場合、ファイル間の境界でスライディングウィンドウが不正なデータを含まないよう、ソートースID ごとにグループ化して処理します。

## Proposed Changes

### データ入力 (既存モジュール再利用)

#### [REUSE] [acmi_parser.py](file:///home/t-kat/src/flight-maneuver-classifier/src/acmi_parser.py)
- `ACMIParser.parse_file()` で ACMI → フレーム列を取得
- フレームから航空機ごとの時系列 DataFrame を構築する関数を `maneuver_feature_engine.py` 側で実装

---

### 特徴量抽出

#### [NEW] [maneuver_feature_engine.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_feature_engine.py)

**役割**: ACMI フレームデータ → 航空機ごとの時系列 DataFrame → スライディングウィンドウ特徴量

1. `frames_to_aircraft_df(frames, aircraft_id)` — フレーム列から特定航空機の時系列を DataFrame (`time, altitude, speed, heading, pitch, roll`) に変換
2. `preprocess_timeseries(df)` — 角度 unwrap (`np.unwrap` for heading/roll)、欠損補間 (`interpolate(method='linear')`)、ソート
3. `extract_window_features(df, window_sec=5.0, step_sec=1.0)` — スライディングウィンドウで以下を抽出:
   - **平均**: `altitude_mean`, `speed_mean`, `heading_mean`, `pitch_mean`, `roll_mean`
   - **標準偏差**: `altitude_std`, `speed_std`, `heading_std`, `pitch_std`, `roll_std`
   - **始端終端差 (delta)**: `altitude_delta`, `speed_delta`, `heading_delta`, `pitch_delta`, `roll_delta`
   - **線形傾き (slope)**: `altitude_slope`, `speed_slope`, `heading_slope`, `pitch_slope`, `roll_slope` (`np.polyfit(x, y, 1)`)
   - **派生特徴量**: `altitude_rate` (高度変化率 m/s), `heading_rate` (°/s), `roll_abs_mean` (|roll|平均)

---

### ルールベースラベル

#### [NEW] [maneuver_labeler.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_labeler.py)

**役割**: 窓内変化量に基づくルールベース仮ラベル付与

ルール（優先度順）:
1. **Roll Maneuver**: `roll_std > 15°` または `|roll_delta| > 30°`
2. **Left Turn**: `heading_delta < -5°` かつ `roll_std ≤ 15°`
3. **Right Turn**: `heading_delta > 5°` かつ `roll_std ≤ 15°`
4. **Climb**: `altitude_slope > 2.0 m/s` かつ `|heading_delta| ≤ 5°`
5. **Descent**: `altitude_slope < -2.0 m/s` かつ `|heading_delta| ≤ 5°`
6. **Level Flight**: 上記に該当しない (デフォルト)

> [!NOTE]
> 閾値はデフォルト値であり、`ManeuverLabeler` の [config](file:///home/t-kat/src/flight-maneuver-classifier/tests/test_labeler.py#163-171) 辞書で変更可能。

---

### Random Forest 分類

#### [NEW] [maneuver_classifier.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_classifier.py)

**役割**: 特徴量 + ラベル → Random Forest 学習 → 評価出力

1. `train_and_evaluate(X, y, test_size=0.2)`:
   - `train_test_split` → `RandomForestClassifier(n_estimators=200)` で学習
   - **評価指標**: Accuracy, Precision, Recall, F1-score (`classification_report`)
   - **Confusion Matrix**: `seaborn.heatmap` で画像保存
   - **特徴量重要度**: 棒グラフで画像保存
2. [save_model(model, path)](file:///home/t-kat/src/flight-maneuver-classifier/src/train.py#171-193) / `load_model(path)`: `joblib` での保存/読込

---

### エントリポイント

#### [NEW] [maneuver_main.py](file:///home/t-kat/src/flight-maneuver-classifier/src/maneuver_main.py)

**役割**: CLI エントリポイント

```
python src/maneuver_main.py flight.acmi [--window 5] [--step 1] [--output-dir results/]
```

処理フロー:
1. [ACMIParser](file:///home/t-kat/src/flight-maneuver-classifier/src/acmi_parser.py#149-508) で ACMI パース
2. 全航空機抽出 → 航空機ごとに時系列 DataFrame 構築
3. 前処理（unwrap, 欠損補間）
4. スライディングウィンドウ特徴量抽出
5. ルールベース仮ラベル付与
6. Random Forest 学習・評価
7. 結果出力（コンソール + 画像 + CSV）

---

## Verification Plan

### Automated Tests

既存テスト (`pytest tests/`) と新規テスト3ファイルを実行:

```bash
cd /home/t-kat/src/flight-maneuver-classifier && python -m pytest tests/ -v
```

#### [NEW] `tests/test_maneuver_feature_engine.py`
- `preprocess_timeseries` の unwrap / 欠損補間の正しさ
- `extract_window_features` のウィンドウ数・特徴量名の正しさ
- エッジケース（データ不足時の空出力等）

#### [NEW] `tests/test_maneuver_labeler.py`
- 各6クラスの判定ロジック
- 優先度ルールの正しさ（Roll Maneuver が Turn より優先）

#### [NEW] `tests/test_maneuver_classifier.py`
- 合成データで Random Forest が学習・評価できること
- 出力に Accuracy, F1-score, Confusion Matrix が含まれること

### Manual Verification
- 実際の ACMI ファイルがあれば `python src/maneuver_main.py flight.acmi` を実行し、出力を確認（ユーザーに依頼）
