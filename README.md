# Flight Maneuver Classifier

Tacview ACMI ログから固定翼機の単機時系列を抽出し、短時間窓ごとに機動ラベルを付与して学習・推論・Tacview 注釈書き戻しを行うツール群です。

この README は、現在のコード実装を「短時間窓ベース航空機機動ラベリング仕様書・設計書」に対応づけて説明します。

## 概要

このプロジェクトは次の処理を行います。

1. ACMI から `Air+FixedWing` のみを抽出
2. 単機時系列を構築
3. 前処理、リサンプリング、平滑化
4. 5 秒窓ごとの特徴量抽出
5. 5 秒主窓の中で 2 秒補助窓の短周期特徴も抽出
6. ルールベース主ラベル・属性ラベル付与
7. そのラベルを教師信号として分類モデルを学習
8. 推論結果を CSV と Tacview Raw Telemetry に出力

## 設計方針

本実装は単機運動学だけを使います。

- 相手機との相対幾何は使いません
- BFM 戦術そのものではなく、観測可能な運動形態を扱います
- 1 窓につき主ラベルは 1 つだけ付与します
- 補助属性ラベルは複数同時に付与できます
- `Reversal` や `Extension` のような戦術語は主ラベルに使いません
- `Transition` と `Uncertain` を導入し、無理な一意分類を避けます

## ラベル体系

### 主ラベル

| ID | Label |
| --- | --- |
| 0 | `Straight_Level` |
| 1 | `Straight_Climb` |
| 2 | `Straight_Descent` |
| 3 | `Level_Turn` |
| 4 | `Climbing_Turn` |
| 5 | `Descending_Turn` |
| 6 | `Dive` |
| 7 | `Zoom_Climb` |
| 8 | `Large_Heading_Change` |
| 9 | `Oscillatory_Maneuver` |
| 10 | `Transition` |
| 11 | `Uncertain` |

### 補助属性ラベル

- `High_G`
- `Unloaded`
- `Accelerating`
- `Decelerating`
- `High_Roll_Rate`
- `High_Turn_Rate`
- `Rapid_Altitude_Change`
- `Heading_Reversal_Like`
- `Jinking_Like`

### 主ラベル優先順位

1. `Uncertain`
2. `Transition`
3. `Oscillatory_Maneuver`
4. `Large_Heading_Change`
5. `Dive`
6. `Zoom_Climb`
7. `Climbing_Turn`
8. `Descending_Turn`
9. `Level_Turn`
10. `Straight_Climb`
11. `Straight_Descent`
12. `Straight_Level`

## 入力仕様

### 必須入力

- `time`
- `altitude`
- `speed`
- `pitch`
- `roll`
- `heading`
- `g_load`

### 任意入力

コードは入力として直接は要求しませんが、内部で次を導出して使います。

- `vertical_speed`
- `roll_rate`
- `pitch_rate`
- `turn_rate`

### 単位

前提は次の通りです。

- `time`: 秒
- `altitude`: m
- `speed`: m/s
- `pitch`, `roll`, `heading`: deg
- `g_load`: G

## 時間窓仕様

- 主ラベル判定窓: 5 秒
- 補助短窓: 2 秒
- デフォルトステップ: 2.5 秒
- オーバーラップ: 50%
- リサンプリング周波数: 5 Hz

仕様書にある 10 Hz / 5 Hz のうち、現実装は 5 Hz を採用しています。
短周期の揺れに敏感な `Oscillatory_Maneuver` と `Jinking_Like` には、5 秒主窓の内部で 2 秒補助窓を走査した特徴を使います。

## 前処理仕様

[src/maneuver_feature_engine.py](/home/t-kat/src/flight-maneuver-classifier/src/maneuver_feature_engine.py) で次を実施します。

1. 時刻順ソート
2. `heading` と `roll` の unwrap
3. 欠損を含む数値列の補間
4. 5 Hz への等間隔リサンプリング
5. 移動平均による平滑化
6. `heading_rate`, `roll_rate`, `pitch_rate`, `vertical_speed` の導出
7. 長い欠損区間の比率と、物理的に不自然な値の検出

## 特徴量仕様

各 5 秒窓で、主に次の特徴量を計算します。

### 基本特徴量

- `speed_mean`
- `speed_delta`
- `speed_slope`
- `alt_mean`
- `alt_delta`
- `alt_slope`
- `pitch_mean`
- `roll_mean`
- `g_mean`
- `g_std`

### 微分特徴量

- `turn_rate_mean`
- `turn_rate_peak`
- `roll_rate_mean`
- `roll_rate_peak`
- `pitch_rate_mean`
- `vertical_speed_mean`
- `vertical_speed_peak`

### 方位変化・振動特徴量

- `heading_delta`
- `turn_sign_changes`
- `roll_sign_changes`
- `pitch_sign_changes`
- `alt_residual_std`

### 補助派生特徴量

- `dive_score_raw`
- `zoom_score_raw`
- `straightness_score`
- `turn_dominance`

### `Transition` / `Uncertain` 判定補助

- `first_half_turn_rate_mean`
- `second_half_turn_rate_mean`
- `first_half_vertical_speed_mean`
- `second_half_vertical_speed_mean`
- `first_half_pitch_mean`
- `second_half_pitch_mean`
- `first_half_heading_delta`
- `second_half_heading_delta`
- `missing_ratio`
- `long_gap_ratio`
- `sensor_conflict`
- `feature_out_of_range`

### 2 秒補助窓特徴量

5 秒主窓の内部で 2 秒サブ窓を走査し、短周期検出用に次を集約します。

- `short_window_count`
- `short_turn_sign_changes_max`
- `short_roll_sign_changes_max`
- `short_pitch_sign_changes_max`
- `short_turn_rate_peak_max`
- `short_roll_rate_peak_max`
- `short_g_std_max`

## 閾値仕様

主なデフォルト閾値は [src/maneuver_labeler.py](/home/t-kat/src/flight-maneuver-classifier/src/maneuver_labeler.py) の `DEFAULT_THRESHOLDS` にあります。

| Key | Default |
| --- | --- |
| `turn_small` | `3.0 deg/s` |
| `turn_large` | `10.0 deg/s` |
| `turn_high` | `18.0 deg/s` |
| `vs_small` | `5.0 m/s` |
| `vs_up` | `10.0 m/s` |
| `vs_down` | `10.0 m/s` |
| `vs_rapid` | `30.0 m/s` |
| `vs_rapid_up` | `30.0 m/s` |
| `pitch_small` | `5.0 deg` |
| `pitch_dive` | `15.0 deg` |
| `pitch_zoom` | `15.0 deg` |
| `g_unloaded_upper` | `0.7 G` |
| `g_high` | `5.5 G` |
| `g_pull` | `2.0 G` |
| `g_std_high` | `0.8 G` |
| `roll_sign_changes` | `2` |
| `turn_sign_changes` | `2` |
| `roll_rate_high` | `60.0 deg/s` |
| `heading_large` | `110.0 deg` |
| `acc` | `1.0 m/s^2` |
| `dec` | `1.0 m/s^2` |

## 主ラベル定義

### `Straight_Level`

- `turn_rate_mean < turn_small`
- `abs(vertical_speed_mean) < vs_small`
- `abs(pitch_mean) < pitch_small`

### `Straight_Climb`

- `turn_rate_mean < turn_small`
- `vertical_speed_mean > vs_up`

### `Straight_Descent`

- `turn_rate_mean < turn_small`
- `vertical_speed_mean < -vs_down`

### `Level_Turn`

- `turn_rate_mean >= turn_large`
- `abs(vertical_speed_mean) < vs_small`

### `Climbing_Turn`

- `turn_rate_mean >= turn_large`
- `vertical_speed_mean > vs_up`

### `Descending_Turn`

- `turn_rate_mean >= turn_large`
- `vertical_speed_mean < -vs_down`

### `Dive`

- `pitch_mean < -pitch_dive`
- `vertical_speed_mean < -vs_rapid`

### `Zoom_Climb`

- `pitch_mean > pitch_zoom`
- `vertical_speed_mean > vs_rapid_up`

### `Large_Heading_Change`

- `heading_delta >= heading_large`
- `Oscillatory_Maneuver` に該当する場合は優先しません
- `Dive` / `Zoom_Climb` が明確に支配的ならそちらを優先します

### `Oscillatory_Maneuver`

- 5 秒窓特徴と 2 秒補助窓特徴の強い方を使って判定します
- `roll_sign_changes >= threshold` または `turn_sign_changes >= threshold`
- かつ `roll_rate_peak >= roll_rate_high` または `g_std >= g_std_high`

### `Transition`

現実装では次のいずれかを満たすと `Transition` にします。

- どの具体ラベルにも十分当てはまらない
- 複数フラグが立ち、上位 2 候補のスコア差が小さい
- 窓前半と後半で粗い運動状態が異なる

### `Uncertain`

現実装では次のいずれかを満たすと `Uncertain` にします。

- `missing_ratio` が高い
- `long_gap_ratio` が高い
- `sensor_conflict` が立つ
- `feature_out_of_range` が立つ
- 主要特徴量が非有限値になる

## 補助属性ラベル定義

- `High_G`: `g_mean >= g_high`
- `Unloaded`: `g_mean <= g_unloaded_upper`
- `Accelerating`: `speed_slope > acc`
- `Decelerating`: `speed_slope < -dec`
- `High_Roll_Rate`: `roll_rate_peak >= roll_rate_high`
- `High_Turn_Rate`: `turn_rate_peak >= turn_high`
- `Rapid_Altitude_Change`: `abs(vertical_speed_mean) >= vs_rapid`
- `Heading_Reversal_Like`: `heading_delta >= heading_large`
- `Jinking_Like`: 5 秒窓特徴と 2 秒補助窓特徴の強い方で `roll_sign_changes` または `turn_sign_changes` が閾値以上

## 判定ロジック

各窓について次の順に処理します。

1. 特徴量計算
2. `Uncertain` 判定
3. 各主ラベル条件フラグ計算
4. 各主ラベルスコア計算
5. `Transition` 判定
6. 優先順位に従って主ラベル決定
7. 補助属性ラベル決定
8. CSV 出力レコード生成

出力には以下が含まれます。

- `label`
- `label_name`
- `main_label`
- `attributes`
- `uncertain_reason`
- `transition_reason`
- `flag_*`
- `score_*`
- `attr_*`

## 出力仕様

`maneuver_features_labeled.csv` には少なくとも次が含まれます。

- `window_start`
- `window_end`
- `window_start_time`
- `window_end_time`
- `aircraft_id`
- `label`
- `label_name`
- `main_label`
- `attributes`
- 各特徴量列
- `uncertain_reason`
- `transition_reason`
- `flag_*`
- `score_*`
- `attr_*`

推論時はさらに次を追加します。

- `predicted_label`
- `predicted_label_name`

## CLI

### 1 本の ACMI を処理

```bash
. .venv/bin/activate
python src/maneuver_main.py Tacview/example.acmi --window 5 --step 2.5
python src/maneuver_main.py Tacview/example.acmi --list-aircraft
```

### 複数 ACMI から学習

```bash
. .venv/bin/activate
python src/maneuver_train.py Tacview/ --recursive --output-dir results/train
```

### 学習済みモデルで推論

```bash
. .venv/bin/activate
python src/maneuver_predict.py Tacview/ \
  --recursive \
  --model results/train/maneuver_rf_model.joblib
```

### 共通閾値引数

[src/maneuver_main.py](/home/t-kat/src/flight-maneuver-classifier/src/maneuver_main.py#L273)、[src/maneuver_train.py](/home/t-kat/src/flight-maneuver-classifier/src/maneuver_train.py#L154)、[src/maneuver_predict.py](/home/t-kat/src/flight-maneuver-classifier/src/maneuver_predict.py#L84) は共通で次を受け付けます。

- `--turn-small-th`
- `--turn-large-th`
- `--turn-high-th`
- `--vs-small-th`
- `--vs-up-th`
- `--vs-down-th`
- `--vs-rapid-th`
- `--vs-rapid-up-th`
- `--pitch-small-th`
- `--pitch-dive-th`
- `--pitch-zoom-th`
- `--g-unloaded-th`
- `--g-high-th`
- `--g-pull-th`
- `--g-std-high-th`
- `--roll-sign-changes-th`
- `--turn-sign-changes-th`
- `--roll-rate-high-th`
- `--heading-large-th`
- `--acc-th`
- `--dec-th`
- `--uncertain-missing-ratio-th`
- `--uncertain-long-gap-ratio-th`
- `--transition-score-gap-th`

## 実装上の近似と未実装

仕様書に対して、現実装は以下を近似しています。

1. `T_acc`, `T_dec`, `T_g_std_high` はデータ分布からの自動再調整ではなく固定初期値です。
2. `Transition` は本格スコアリングではなく、フラグ数、簡易スコア差、前半/後半判定差で近似しています。
3. `Uncertain` の `sensor_conflict` は単機量だけから判定できる物理矛盾チェックに限定しています。
4. 平滑化は Savitzky-Golay ではなく移動平均です。

## 仕様上扱わないもの

相手機情報を使わないため、次は厳密には扱いません。

- `Reversal`
- `Extension`
- `Lead_Pursuit`
- `Lag_Pursuit`
- `Pure_Pursuit`
- Offensive / Defensive / Neutral
- Overshoot

## モデル互換性

- ラベル体系と特徴量列が旧実装から変わっているため、旧モデルとの互換性はありません
- 既存モデルを使い回すのではなく再学習が必要です
- `label` はルールベースラベル、`predicted_label` は学習済みモデルの出力です

## 関連ファイル

- [src/maneuver_feature_engine.py](/home/t-kat/src/flight-maneuver-classifier/src/maneuver_feature_engine.py)
- [src/maneuver_labeler.py](/home/t-kat/src/flight-maneuver-classifier/src/maneuver_labeler.py)
- [src/maneuver_main.py](/home/t-kat/src/flight-maneuver-classifier/src/maneuver_main.py)
- [src/maneuver_train.py](/home/t-kat/src/flight-maneuver-classifier/src/maneuver_train.py)
- [src/maneuver_predict.py](/home/t-kat/src/flight-maneuver-classifier/src/maneuver_predict.py)
