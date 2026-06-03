# Flight Maneuver Classifier 詳細解説

## 1. この文書の目的

この文書は、このリポジトリのプログラムが実際に何をしているかを、コード実装に沿ってできるだけ具体的に説明するためのものである。

特に次を対象にする。

- ACMI から何を読み取っているか
- どのように前処理しているか
- どのような時間窓・特徴量でラベリングしているか
- どのようなルールで主ラベルと補助属性ラベルを決めているか
- どのようなモデルで学習しているか
- どのような評価と出力を行っているか
- Tacview への書き戻しと手動修正をどう扱っているか

README よりも一段深く、実装の流れとデータの中身が追える説明を目指す。

---

## 2. このプログラム全体の役割

このプロジェクトは、Tacview の ACMI ログから固定翼機の単機時系列を抽出し、短時間窓ごとに機動ラベルを付け、そのラベルを教師信号として機械学習モデルを学習させるツール群である。

単純に言うと、処理は次の順で進む。

1. ACMI を読む
2. 固定翼機だけを抜き出す
3. 1 機ごとの時系列を作る
4. リサンプリング、平滑化、微分量導出を行う
5. 5 秒主窓と 2 秒補助窓から特徴量を作る
6. ルールベースで主ラベルと補助属性ラベルを付ける
7. そのルールラベルを教師にして分類モデルを学習する
8. 予測結果と判定根拠を CSV と Annotated ACMI に出力する
9. 必要なら Tacview 上で人手修正し、その修正を使って再学習する

重要なのは、このプロジェクトが単なる「分類器」ではないことだ。実際には以下をまとめて持っている。

- ACMI パーサ
- 単機時系列の前処理器
- 短時間窓特徴量エンジン
- ルールベースラベラー
- 学習器
- 推論器
- Tacview への注釈書き戻し器
- Tacview 手動ラベル修正アドオン
- 手動修正を再学習に戻すユーティリティ

---

## 3. 想定している設計思想

このツールは、相手機との相対位置関係を使わず、自機単独の運動学だけでラベル付けする設計になっている。

そのため、扱うのは以下である。

- 直進
- 上昇
- 降下
- 旋回
- 急降下
- ズーム上昇
- 大きな方位変化
- 揺れや切り返しの多い機動

逆に、厳密には扱っていないものは以下である。

- `Reversal`
- `Extension`
- `Lead / Lag / Pure pursuit`
- Offensive / Defensive / Neutral
- Overshoot

これらは相手機との相対幾何なしでは厳密に定義できないため、現実装では採用していない。

---

## 4. リポジトリ内の主要ファイル

中心になるファイルは次の通りである。

- `src/acmi_parser.py`
  - Tacview ACMI のパーサ
- `src/maneuver_feature_engine.py`
  - 単機時系列の前処理と特徴量抽出
- `src/maneuver_labeler.py`
  - ルールベース主ラベル・補助属性ラベル付与
- `src/maneuver_classifier.py`
  - Random Forest / XGBoost の学習・評価・保存
- `src/maneuver_pipeline.py`
  - 学習・推論で共通に使うパイプライン部品
- `src/maneuver_main.py`
  - 1 本の ACMI を一括処理するエントリポイント
- `src/maneuver_train.py`
  - 複数 ACMI をまとめて学習
- `src/maneuver_predict.py`
  - 学習済みモデルで推論
- `src/maneuver_acmi_export.py`
  - Annotated ACMI の生成
- `src/maneuver_manual_labels.py`
  - Tacview 手動ラベル修正 CSV の取り込み
- `src/maneuver_retrain_from_labels.py`
  - 手動修正を反映した再学習
- `ManualLabelEditor/main.lua`
  - Tacview アドオン

---

## 5. エンドツーエンドの流れ

エンドツーエンドでは概ね次のように動く。

```text
Tacview ACMI
  -> ACMI parser
  -> fixed-wing aircraft extraction
  -> per-aircraft time series
  -> preprocessing
  -> 5 s main windows + 2 s auxiliary windows
  -> feature matrix
  -> rule-based labels
  -> ML training or inference
  -> CSV / model / plots / annotated ACMI
  -> optional Tacview manual edits
  -> retraining
```

用途ごとの入口は次のように分かれている。

- 単発で 1 本を回す: `src/maneuver_main.py`
- 複数 ACMI から学習したい: `src/maneuver_train.py`
- 既存モデルを別ログへ適用したい: `src/maneuver_predict.py`
- 人手修正を反映して再学習したい: `src/maneuver_retrain_from_labels.py`

---

## 6. 入力データ

### 6.1 ファイル形式

入力の中心は Tacview ACMI ファイルである。現在のコードは次を扱う。

- `.acmi`
- `.txt.acmi`
- `.zip.acmi`

ZIP の場合は内部の最初の ACMI テキストを読み取る。

### 6.2 必須に近い系列

単機時系列の処理で使う系列は次である。

- `time`
- `altitude`
- `yaw`
- `pitch`
- `roll`
- `world_x`
- `world_y`
- `world_z`

このうち後段で直接使う派生量は以下で、前処理段階で差分から導出される。

- `speed`
- `g_load`
- `yaw_rate`
- `turn_rate`
- `roll_rate`
- `pitch_rate`
- `vertical_speed`

### 6.3 単位の前提

コードが前提としている単位は次の通りである。

- `time`: 秒
- `altitude`: m
- `speed`: m/s
- `pitch`, `roll`, `yaw`: deg
- `g_load`: G

`speed` と `g_load` は Tacview の追加プロパティをそのまま信頼せず、位置差分から再計算する。

---

## 7. ACMI パースの詳細

### 7.1 何を読み取るか

`src/acmi_parser.py` は ACMI の各行を読んで、フレーム境界ごとに各オブジェクトのスナップショットを保持する。

ACMI の主な行種別は以下である。

- `#<time>`
  - フレーム時刻
- `0,...`
  - グローバルプロパティ
- `<object_id>,...`
  - オブジェクト更新
- `-<object_id>`
  - オブジェクト削除

### 7.2 パーサ内部のデータ構造

主なデータ構造は以下である。

- `ACMIObject`
  - 1 オブジェクトの現在状態
- `ACMIFrame`
  - 1 フレーム時点での全オブジェクト状態
- `ACMIParser`
  - フレーム列とイベントを組み立てる本体

### 7.3 オブジェクトから読んでいる情報

主に次を `ACMIObject` に入れている。

- 識別情報
  - `obj_id`, `name`, `pilot`, `obj_type`, `coalition`, `country`
- 位置
  - `longitude`, `latitude`, `altitude`
- 姿勢
  - `roll`, `pitch`, `yaw`
- 速度・状態
  - `speed`, `ias`, `tas`, `mach`, `aoa`, `g_load`
- 補助
  - `world_x`, `world_y`, `world_z`

### 7.4 速度の扱い

速度は Tacview の速度プロパティを主入力には使わず、ワールド座標差分から常時計算する。

したがって、このプロジェクトの `speed` は常に位置差分由来の近似値である。

### 7.5 固定翼機の判定

解析対象は `Type=Air+FixedWing` 相当である。`src/maneuver_feature_engine.py` の `is_analysis_target_aircraft()` では `obj_type.lower() == "air+fixedwing"` を条件にしている。

したがって、Tacview 側で型が異なる記録になっている機体は対象外になり得る。

---

## 8. 単機時系列の構築

`src/maneuver_feature_engine.py` の `frames_to_aircraft_df()` は、全フレームから指定した `aircraft_id` の行だけを抜き出して、1 機分の DataFrame を作る。

生成される基本列は以下である。

- `time`
- `altitude`
- `world_x`
- `world_y`
- `world_z`
- `yaw`
- `pitch`
- `roll`

この段階では `speed` や `g_load` はまだ最終列として持たず、位置と姿勢を元に後段で導出する。時間間隔は不均一でもよく、欠損もあり得る。整形は次の前処理段階で行う。

### 8.1 単機時系列の構築モデルとは何か

このプロジェクトでいう「単機時系列の構築」は、Tacview ACMI のフレーム列から、ある 1 機体について

- 時刻 `t`
- その時刻に観測された機体状態 `x(t)`

を順番に並べた系列

```text
{ (t_0, x_0), (t_1, x_1), ..., (t_n, x_n) }
```

を作る処理である。

ここで各状態ベクトル `x(t)` は、現実装では主に次で構成される。

```text
x(t) = [
  altitude(t),
  yaw(t),
  pitch(t),
  roll(t),
  world_x(t),
  world_y(t),
  world_z(t)
]
```

つまり、単機時系列の構築モデルは「フレームベース ACMI を、単一機体の状態ベクトル列へ射影するモデル」と考えればよい。

### 8.2 元データがフレームベースであることの意味

ACMI は本質的には、各時刻フレームごとに「その時点で存在している全オブジェクトの状態」を持っている。

言い換えると、元データは

```text
frame_0 = { object_a: state_a0, object_b: state_b0, ... }
frame_1 = { object_a: state_a1, object_c: state_c1, ... }
...
```

のような形であり、単機ごとに最初から分かれているわけではない。

そのため単機時系列を作るには、

1. 各フレームを見る
2. そのフレームに対象 `aircraft_id` が存在するか確認する
3. 存在すればその状態だけを抜き出す
4. 時刻順に並べる

という投影操作が必要になる。

### 8.3 パーサ側で先に「状態を持続」させている

ここで重要なのは、`frames_to_aircraft_df()` より前の `src/acmi_parser.py` が、すでに「オブジェクト状態を持続するモデル」になっていることだ。

ACMI のオブジェクト更新行は差分更新のように来ることがある。つまりある行では `Name` だけ、別の行では `T=` だけ、ということがあり得る。

そこでパーサは内部に `self._objects` を持ち、各オブジェクトの最新状態を保持している。新しい更新が来たときは、そのオブジェクトの一部の属性だけを書き換える。

その上で `#<time>` が来るたびに `_flush_frame()` を呼び、**その瞬間の全オブジェクト状態をスナップショット化して `ACMIFrame` に固定する**。

したがって `frames_to_aircraft_df()` が読む `frames` は、単なる生の差分列ではなく、すでに

- その時刻における姿勢
- その時刻における速度
- その時刻における G

などが復元された「時刻付き完全状態列」に近いものになっている。

### 8.4 単機時系列の 1 行は何を意味するか

`frames_to_aircraft_df()` の 1 行は、「対象機がそのフレームに存在していたなら、そのフレーム時刻における機体状態 1 サンプル」を意味する。

具体的には、各フレームについて以下を行う。

1. `frame.objects.get(aircraft_id)` を見る
2. 対象オブジェクトがいなければそのフレームはスキップする
3. いれば状態を辞書へ変換して `rows` に追加する

抜き出す列対応は次の通りである。

- `time <- frame.time`
- `altitude <- obj.altitude`
- `world_x <- obj.world_x`
- `world_y <- obj.world_y`
- `world_z <- obj.world_z`
- `yaw <- obj.yaw`
- `pitch <- obj.pitch`
- `roll <- obj.roll`

つまり、生の単機 DataFrame は `heading` ではなく `yaw` を持ち、速度や G はこの後に再構成する。

### 8.5 速度・姿勢・G の各値がどこから来るか

単機時系列の各列は、パーサ内では次のように形成されている。

#### `altitude`, `world_x`, `world_y`, `world_z`, `roll`, `pitch`, `yaw`

これらは `T=longitude|latitude|altitude|roll|pitch|yaw` から読む。値が空欄の項目は前回値を維持する。

#### `speed`

`speed` は最終的に前処理でワールド座標の時間微分から求める。つまり

```text
v(t) = d/dt [world_x, world_y, world_z]
speed(t) = ||v(t)||
```

で計算する。

#### `g_load`

`g_load` も Tacview 追加プロパティには依存せず、軌跡の曲率から求める。

概念的には

```text
v(t) = d/dt [world_x, world_y, world_z]
a(t) = dv/dt
a_normal = |v x a| / |v|
g_load = sqrt(1 + (a_normal / g)^2)
```

である。これは coordinated turn に近い状況では自然な近似になる。

### 8.6 「存在しないフレーム」はどう扱うか

対象機があるフレームに存在しない場合、`frames_to_aircraft_df()` はそのフレームの行を作らない。

つまり、この段階の単機時系列は

- 全フレームに対する固定長系列

ではなく

- 対象機が観測された時刻だけの疎な系列

である。

結果として時刻間隔は不均一になり得るし、途中で機体が出現・消滅した場合は、その出現区間だけが切り出される。

この疎さは前処理段階で 5 Hz 等間隔系列へ変換される。

### 8.7 時系列としての性質

この段階で得られる単機時系列には、次の性質がある。

- 時刻順には並んでいる
- 各行は 1 フレーム由来の観測値である
- サンプリング周期はまだ不均一
- 欠損値があり得る
- 角度 wrap もまだ残っている
- 速度の一部は推定値かもしれない
- G はログによっては情報量が低い

つまり、これは「そのまま学習に入れる完成データ」ではなく、**特徴量抽出のための中間観測系列**である。

### 8.8 数式的に書くと何をしているか

`F_k` を時刻 `t_k` のフレーム、`O_k(i)` をフレーム `k` における機体 `i` の状態とすると、対象機 `i` の単機時系列は

```text
S_i = [ (t_k, g(O_k(i))) | O_k(i) が存在する ]
```

で定義できる。

ここで `g(.)` は、ACMI オブジェクト状態から

- altitude
- yaw
- pitch
- roll
- world_x
- world_y
- world_z

を抜き出してベクトル化する写像である。

実装の `frames_to_aircraft_df()` は、ほぼこの `S_i` を DataFrame として具体化している。

### 8.9 なぜこのモデルが必要か

後段の特徴量計算はすべて「1 機体の時間変化」を前提にしている。

たとえば次の量は、同じ機体について時刻方向に差分を取らなければ意味を持たない。

- `yaw_rate`
- `turn_rate`
- `roll_rate`
- `vertical_speed`
- `heading_delta`
- `speed_slope`

したがって、まず ACMI の多オブジェクト・多時刻データを、機体ごとの単機時系列へ正しく分解する必要がある。

この単機時系列構築モデルが崩れると、以降の特徴量、ラベル、学習器はすべて崩れる。言い換えると、ここはパイプライン全体の土台である。

---

## 9. 前処理の詳細

前処理は `src/maneuver_feature_engine.py` の `preprocess_timeseries()` が担当する。

### 9.1 時刻ソートと数値化

最初に `time` を数値化し、時刻昇順で並べ替える。異常な `time` は落とす。

また、各必須信号を数値列へ変換する。

### 9.2 `speed` と `g_load` の扱い

現実装では `speed` と `g_load` を入力の既存列としては使わず、位置差分から再計算する。

そのため、Tacview 側に `TAS` や `GLoad` が無くても処理自体は成立する。

### 9.3 欠損の粗い指標

各行について、必須信号が全て埋まっているかを見て `raw_missing_any` を持たせる。

- 全て埋まっていれば `0.0`
- どれか欠けていれば `1.0`

この値は後で窓平均され、`missing_ratio` になる。

### 9.4 角度 unwrap

そのまま角度差分を取ると `359 deg -> 1 deg` のようなまたぎで偽の急旋回が発生する。そこで以下を unwrap する。

- `yaw`
- `roll`

`pitch` は通常そのままだが、急に 150 deg 以上飛ぶような不連続が見えたときだけ保険的に unwrap する。

### 9.5 等間隔リサンプリング

解析は 5 Hz に揃えて行う。つまり 0.2 秒刻みの時系列へ変換する。

流れは以下である。

1. 元の時刻列から開始時刻と終了時刻を得る
2. 5 Hz の `target_times` を作る
3. 各信号を線形補間で `target_times` に写す

この処理により、ログごとのサンプル密度差を吸収し、窓特徴量を安定して比較できる。

### 9.6 長欠損区間のマーキング

元時系列のサンプル間隔が `long_gap_sec` より長い箇所は「長いギャップ」とみなし、そのギャップ内部に `long_gap_flag=1.0` を立てる。

デフォルトでは以下である。

- `resample_hz = 5.0`
- `smoothing_window_sec = 0.6`
- `long_gap_sec = 1.0`

ここで重要なのは、短い欠損は補間で吸収する一方、長い欠損は別途フラグとして残す点である。

### 9.7 欠損補間

リサンプリング後の各信号は以下で埋める。

- 線形補間
- 前方埋め
- 後方埋め

これにより差分計算が NaN で壊れるのを防いでいる。

### 9.8 平滑化

微分前のノイズを抑えるため、位置と姿勢の元信号に移動平均をかける。

対象は以下である。

- `altitude`
- `world_x`
- `world_y`
- `world_z`
- `yaw`
- `pitch`
- `roll`

平滑化窓サイズは `round(smoothing_window_sec * resample_hz)` で決まり、デフォルトでは約 3 サンプル前後になる。

### 9.9 微分量の導出

平滑化後に `np.gradient()` で以下を求める。

- `speed`
- `g_load`
- `yaw_rate`
- `turn_rate`
- `roll_rate`
- `pitch_rate`
- `vertical_speed`

ここで `turn_rate` は `yaw_rate` ではない。3 次元速度ベクトル `v` と加速度ベクトル `a` から

```text
turn_rate = |v x a| / |v|^2
```

を計算し、deg/s に変換した空間的な旋回率を使っている。

`g_load` も同じ `v` と `a` から得られる法線加速度を使って

```text
g_load = sqrt(1 + (a_normal / g)^2)
```

で推定する。以前のように加速度ベクトル全体から直接荷重倍数を作る方法は、位置ノイズの 2 階微分を強く拾ってしまい、格闘戦ログで 10G 超が過剰発生しやすかった。

---

## 10. 時間窓の切り方

### 10.1 主窓

主ラベル判定窓は 5 秒である。

デフォルト値は以下。

- `window_sec = 5.0`
- `step_sec = 2.5`

したがって 50% オーバーラップのスライディングウィンドウになる。

### 10.2 2 秒補助窓

`Oscillatory_Maneuver` や `Jinking_Like` のような短周期変化を見落としにくくするため、5 秒主窓の内部をさらに 2 秒窓で走査する。

実装上は `SHORT_WINDOW_SEC = 2.0` で、主窓のサンプル周期を利用して 2 秒区間を細かく滑らせながら集約値を取る。

### 10.3 窓の出力単位

最終的に 1 窓につき 1 行のレコードが生成される。この 1 行が、学習・推論・Tacview 書き戻しの共通単位になる。

---

## 11. 特徴量設計の詳細

特徴量は `src/maneuver_feature_engine.py` の `FEATURE_COLUMNS` に定義されている。実装では 5 秒主窓から 40 以上の特徴量を作っている。

### 11.1 窓メタ情報

まず各窓には特徴量以外に次のメタ情報が付く。

- `window_start`
- `window_end`
- `window_start_time`
- `window_end_time`
- `window_duration`
- `sample_count`

### 11.2 基本特徴量

以下は窓全体の統計量である。

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

定義の考え方は以下。

- `mean`: 窓内平均
- `delta`: 窓末尾値 - 窓先頭値
- `slope`: 窓内時系列に対する 1 次回帰の傾き

### 11.3 微分特徴量

以下は微分系列から作る。

- `turn_rate_mean`
- `turn_rate_peak`
- `roll_rate_mean`
- `roll_rate_peak`
- `pitch_rate_mean`
- `vertical_speed_mean`
- `vertical_speed_peak`

ここで `turn_rate_mean` は 3 次元 `turn_rate` の平均であり、`vertical_speed_mean` は `world_y` の時間微分平均である。

### 11.4 振動・方位変化特徴量

以下は振れや切り返しを把握するための特徴量である。

- `heading_delta`
- `turn_sign_changes`
- `roll_sign_changes`
- `pitch_sign_changes`
- `alt_residual_std`

`sign_changes` は deadband 付きの符号反転回数で、実装上の deadband は以下である。

- `yaw_rate`: `0.2`
- `roll_rate`: `1.0`
- `pitch_rate`: `0.5`

### 11.5 派生特徴量

ルールで直接参照しないものも含め、説明力を持たせるために次を作る。

- `dive_score_raw`
  - `max(0, -pitch_mean) + max(0, -vertical_speed_mean)`
- `zoom_score_raw`
  - `max(0, pitch_mean) + max(0, vertical_speed_mean)`
- `straightness_score`
  - `1 / (1 + turn_rate_mean)`
- `turn_dominance`
  - `turn_rate_mean / (abs(vertical_speed_mean) + 1e-3)`

### 11.6 前半 / 後半特徴量

5 秒窓を前半と後半に分けて、`Transition` 判定用に次を作る。

- `first_half_turn_rate_mean`
- `second_half_turn_rate_mean`
- `first_half_vertical_speed_mean`
- `second_half_vertical_speed_mean`
- `first_half_pitch_mean`
- `second_half_pitch_mean`
- `first_half_heading_delta`
- `second_half_heading_delta`

目的は、同じ 5 秒の中で挙動が切り替わっている窓を検出することにある。

### 11.7 2 秒補助窓特徴量

2 秒窓からは最大値集約として以下を作る。

- `short_window_count`
- `short_turn_sign_changes_max`
- `short_roll_sign_changes_max`
- `short_pitch_sign_changes_max`
- `short_turn_rate_peak_max`
- `short_roll_rate_peak_max`
- `short_g_std_max`

5 秒平均では薄まる短い切り返しを拾うのが目的で、`Oscillatory_Maneuver` と `Jinking_Like` の感度向上に効く。

### 11.8 品質管理特徴量

`Uncertain` 判定や健全性確認用に次を持つ。

- `missing_ratio`
- `long_gap_ratio`
- `sensor_conflict`
- `feature_out_of_range`

#### `sensor_conflict`

次のどれかがあると 1 になる。

- `speed < 0`
- `abs(pitch) > 100`
- `g_load < -3` または `g_load > 12`
- `abs(turn_rate) > 200`

#### `feature_out_of_range`

次のどれかがあると 1 になる。

- 必須信号または微分信号に非有限値がある
- `altitude < -1000`
- `abs(vertical_speed) > 300`
- `abs(turn_rate) > 300`
- `abs(roll_rate) > 400`
- `abs(pitch_rate) > 200`

---

## 12. 主ラベルと補助属性ラベル

### 12.1 主ラベル

現在の主ラベルは以下の 12 種である。

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

### 12.2 補助属性ラベル

現在の補助属性は以下の 9 種である。

- `High_G`
- `Unloaded`
- `Accelerating`
- `Decelerating`
- `High_Roll_Rate`
- `High_Turn_Rate`
- `Rapid_Altitude_Change`
- `Heading_Reversal_Like`
- `Jinking_Like`

主ラベルは排他的だが、補助属性は複数同時に立つ。

---

## 13. ルールベース判定ロジック

ルールベース判定は `src/maneuver_labeler.py` が担う。

### 13.1 基本フロー

各窓について以下の順で処理する。

1. `Uncertain` 判定
2. 具体主ラベルのフラグ判定
3. 各主ラベルの簡易スコア計算
4. `Transition` 判定
5. 優先順位に従って主ラベルを 1 つ決定
6. 補助属性を独立付与

### 13.2 主な閾値

デフォルト閾値は次の通りである。

- `turn_small = 3.0 deg/s`
- `turn_large = 10.0 deg/s`
- `turn_high = 18.0 deg/s`
- `vs_small = 5.0 m/s`
- `vs_up = 10.0 m/s`
- `vs_down = 10.0 m/s`
- `vs_rapid = 30.0 m/s`
- `vs_rapid_up = 30.0 m/s`
- `pitch_small = 5.0 deg`
- `pitch_dive = 15.0 deg`
- `pitch_zoom = 15.0 deg`
- `g_unloaded_upper = 0.7`
- `g_high = 5.5`
- `g_pull = 2.0`
- `g_std_high = 0.8`
- `roll_sign_changes = 2`
- `turn_sign_changes = 2`
- `roll_rate_high = 60.0 deg/s`
- `heading_large = 110.0 deg`
- `acc = 1.0 m/s^2`
- `dec = 1.0 m/s^2`
- `uncertain_missing_ratio = 0.35`
- `uncertain_long_gap_ratio = 0.25`
- `transition_score_gap = 0.001`

### 13.3 具体主ラベルの条件

実装上の考え方は次の通りである。

- `Straight_Level`
  - `turn_rate_mean < turn_small`
  - `abs(vertical_speed_mean) < vs_small`
  - `abs(pitch_mean) < pitch_small`
- `Straight_Climb`
  - `turn_rate_mean < turn_small`
  - `vertical_speed_mean > vs_up`
- `Straight_Descent`
  - `turn_rate_mean < turn_small`
  - `vertical_speed_mean < -vs_down`
- `Level_Turn`
  - `turn_rate_mean >= turn_large`
  - `abs(vertical_speed_mean) < vs_small`
- `Climbing_Turn`
  - `turn_rate_mean >= turn_large`
  - `vertical_speed_mean > vs_up`
- `Descending_Turn`
  - `turn_rate_mean >= turn_large`
  - `vertical_speed_mean < -vs_down`
- `Dive`
  - `pitch_mean < -pitch_dive`
  - `vertical_speed_mean < -vs_rapid`
- `Zoom_Climb`
  - `pitch_mean > pitch_zoom`
  - `vertical_speed_mean > vs_rapid_up`
- `Large_Heading_Change`
  - `heading_delta >= heading_large`
- `Oscillatory_Maneuver`
  - 切り返し回数条件と `roll_rate_peak` または `g_std` 条件を同時に満たす

### 13.4 2 秒補助窓の反映

`Oscillatory_Maneuver` と `Jinking_Like` は、5 秒主窓の値だけでなく、以下のように 2 秒補助窓の最大値も加味する。

- `turn_sign_changes`
  - `max(turn_sign_changes, short_turn_sign_changes_max)`
- `roll_sign_changes`
  - `max(roll_sign_changes, short_roll_sign_changes_max)`
- `turn_rate_peak`
  - `max(turn_rate_peak, short_turn_rate_peak_max)`
- `roll_rate_peak`
  - `max(roll_rate_peak, short_roll_rate_peak_max)`
- `g_std`
  - `max(g_std, short_g_std_max)`

つまり、短周期の鋭い変化を優先的に拾う設計である。

### 13.5 `Uncertain`

以下のいずれかがあると `Uncertain` になる。

- `missing_ratio` が閾値以上
- `long_gap_ratio` が閾値以上
- `sensor_conflict`
- `feature_out_of_range`
- 主要特徴量に非有限値がある

理由は `uncertain_reason` 列に保存される。

### 13.6 `Transition`

以下のどれかで `Transition` になる。

- 具体ラベルが 1 つも成立しない
- 複数具体ラベルが成立し、上位 2 スコア差が小さい
- 前半と後半の粗ラベルが一致しない

理由は `transition_reason` に入る。

代表的な理由は以下。

- `no_rule_matched`
- `multiple_flags_no_dominant`
- `window_halves_disagree:<A>-><B>`

### 13.7 `Large_Heading_Change` の支配性調整

`Large_Heading_Change` は少し特別扱いされている。

- `Oscillatory_Maneuver` が立っている場合は抑制する
- `Dive` / `Zoom_Climb` のスコアが十分高い場合も抑制する

つまり、大きく向きが変わっていても、それが本質的に急降下やズーム上昇として見た方が自然なら、そちらを優先する。

### 13.8 優先順位

最終主ラベルの優先順位は以下である。

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

### 13.9 補助属性の判定

属性は主ラベルとは独立に付く。

- `High_G`
  - `g_mean >= g_high`
- `Unloaded`
  - `g_mean <= g_unloaded_upper`
- `Accelerating`
  - `speed_slope > acc`
- `Decelerating`
  - `speed_slope < -dec`
- `High_Roll_Rate`
  - `roll_rate_peak >= roll_rate_high`
- `High_Turn_Rate`
  - `turn_rate_peak >= turn_high`
- `Rapid_Altitude_Change`
  - `abs(vertical_speed_mean) >= vs_rapid`
- `Heading_Reversal_Like`
  - `heading_delta >= heading_large`
- `Jinking_Like`
  - ロールまたは旋回の符号反転回数が閾値以上

---

## 14. ルールラベルから教師データを作る仕組み

このプロジェクトの特徴は、最初の教師データが人手ラベルではなく、ルールベースラベルである点にある。

つまり最初の学習は次の構造になる。

1. ルールベースで窓ラベルを付ける
2. そのラベルを正解として ML モデルを学習する

このため、初期段階では「モデルが人間の真の正解を学ぶ」のではなく、「モデルがルールベースラベラーを近似する」側面が強い。

ただし、Tacview アドオンで人手修正を加えられるため、運用上は次のように精度改善できる。

1. ルールで初期ラベル生成
2. Tacview 上で誤りを人手修正
3. 手動修正済み窓を優先した再学習

この流れにより、徐々に人手監修データへ寄せていける。

---

## 15. 学習用データの整形

`src/maneuver_pipeline.py` の `prepare_training_dataframe()` が学習前の整形を行う。

### 15.1 学習不能なケース

以下の場合は学習しない。

- ラベル種類が 1 つしかない
- 2 サンプル以上あるクラスが 2 種類未満

これは `train_test_split(..., stratify=y)` を成立させるためである。

### 15.2 サンプル不足クラスの除外

サンプル数 1 のクラスは除外する。結果として、元 CSV には存在していても、学習時には使われないクラスがあり得る。

したがって、出力 CSV と実際の学習対象分布は完全には一致しない場合がある。

---

## 16. どのようなモデルで学習しているか

学習本体は `src/maneuver_classifier.py` の `train_and_evaluate()` にある。

### 16.1 入力特徴量

モデル入力は `FEATURE_COLUMNS` の全列である。つまり、以下をまとめて入れている。

- 基本特徴量
- 微分特徴量
- 振動特徴量
- 派生特徴量
- 前半 / 後半特徴量
- 2 秒補助窓特徴量
- 品質管理特徴量

### 16.2 NaN / inf の扱い

学習前に `np.nan_to_num()` を通して、以下を 0 に寄せる。

- `NaN`
- `+inf`
- `-inf`

### 16.3 学習 / テスト分割

分割は窓単位で次の設定になっている。

- `test_size = 0.2`
- `random_state = 42`
- `stratify = y`

重要な注意点として、これは sortie 単位分割でも ACMI 単位分割でもない。つまり同じ ACMI 由来の窓が train/test の両方に入ることがある。

そのため、得られる指標は「未知 sortie への汎化」よりも、「同分布の窓分類再現性」に近い。

### 16.4 Random Forest

デフォルトモデルは `RandomForestClassifier` である。

主な設定は以下。

- `n_estimators = 200`
- `max_depth = None`
- `min_samples_split = 5`
- `min_samples_leaf = 2`
- `class_weight = "balanced"`
- `n_jobs = -1`

この設定の意味は大まかに次である。

- 木を 200 本使う
- 少数クラスを少し救済するために `balanced`
- CPU 並列を使う
- 極端な過学習を少し抑えるために葉サイズと分割条件を設ける

### 16.5 XGBoost

`--model-type xgboost` を選ぶと `XGBClassifier` を使う。

主な設定は以下。

- `n_estimators = 200`
- `max_depth = 8`
- `learning_rate = 0.1`
- `subsample = 0.9`
- `colsample_bytree = 0.9`
- `objective = "multi:softmax"`
- `eval_metric = "mlogloss"`
- `tree_method = "hist"`
- `num_class = max(y) + 1`

`--use-gpu` を付けると `device = "cuda"` が入る。

### 16.6 現在の性格

現状の学習は、深い時系列モデルではなく、窓特徴量を入力とする表形式分類である。したがって、窓内の詳細波形そのものを end-to-end で学習するわけではない。

この設計の利点は、説明しやすく、Tacview の目視検証と対応付けやすいことにある。

---

## 17. どのように評価しているか

学習後に出す評価は以下である。

- Accuracy
- weighted Precision
- weighted Recall
- weighted F1-score
- classification report
- confusion matrix
- feature importances

### 17.1 Classification report

各クラスについて precision / recall / f1 を出す。出力対象は test 側に出現したクラスと予測側に出現したクラスの和集合である。

### 17.2 Confusion matrix

混同行列は `confusion_matrix.png` として保存される。

これは「どのラベルをどのラベルと取り違えているか」を見るための主な可視化である。

### 17.3 Feature importances

木モデルの `feature_importances_` を使い、上位 15 特徴量を `feature_importances.png` に保存する。

これにより、どの特徴量が現在の学習器に強く効いているかをざっくり把握できる。

---

## 18. モデル保存

学習済みモデルは joblib で保存する。

保存内容は以下である。

- モデル本体
- `feature_columns`
- `task = "classification"`
- `classes`
- 任意メタデータ

出力名は通常次のどちらかである。

- `maneuver_rf_model.joblib`
- `maneuver_xgboost_model.joblib`

推論側はこの `feature_columns` を読み出して、列順を合わせて予測する。

---

## 19. 主な実行スクリプトごとの処理内容

### 19.1 `src/maneuver_main.py`

単一 ACMI を一括処理する。

処理は以下。

1. ACMI パース
2. 固定翼機一覧取得
3. 各機体の特徴量抽出
4. ルールベースラベル付与
5. 学習可能ならモデル学習
6. 学習済みモデルで全窓へ予測列追加
7. CSV 保存
8. Annotated ACMI 保存

したがって「とりあえず 1 本回したい」用途に最も向く。

### 19.2 `src/maneuver_train.py`

複数 ACMI をまとめて学習する。

主な流れは以下。

1. 入力パス展開
2. 各 ACMI からラベル付き特徴量抽出
3. 全 ACMI の窓を連結
4. 学習可能なクラスだけ残す
5. モデル学習
6. モデル保存
7. 連結 CSV 保存
8. 入力ごとの予測付き CSV と Annotated ACMI も自動生成

### 19.3 `src/maneuver_predict.py`

既存モデルを別ログへ適用する。

主な流れは以下。

1. 学習済みモデル読込
2. 新しい ACMI 群からラベル付き特徴量抽出
3. `predicted_label` を追加
4. CSV 保存
5. Annotated ACMI 保存

このときルールベースラベルも同時に再計算されるため、予測結果とルール結果を並べて比較できる。

### 19.4 `src/maneuver_retrain_from_labels.py`

既存の `maneuver_features_labeled.csv` に Tacview 上の人手修正をマージして再学習する。

このスクリプトでは、最終的に `training_label` を本当の教師として使う。

---

## 20. 出力 CSV には何が入るか

最も重要な出力は `maneuver_features_labeled.csv` である。

この CSV は 1 行 1 窓で、概ね次の 6 群の情報を持つ。

### 20.1 窓情報

- `aircraft_id`
- `aircraft_name`
- `window_start`
- `window_end`
- `sample_count`
- `source_acmi`
- `source_title`

### 20.2 特徴量

- `speed_mean`
- `alt_slope`
- `turn_rate_mean`
- `heading_delta`
- `short_roll_rate_peak_max`
- `missing_ratio`

など、`FEATURE_COLUMNS` 全体。

### 20.3 主ラベル

- `label`
- `label_name`
- `main_label`
- `main_label_id`

### 20.4 補助属性

- `attributes`
- `attribute_count`
- `attr_high_g`
- `attr_unloaded`
- `attr_accelerating`

など。

### 20.5 判定根拠

- `uncertain_reason`
- `transition_reason`
- `flag_straight_level`
- `flag_dive`
- `score_dive`
- `score_zoom_climb`

など。

### 20.6 モデル予測

モデルで予測した後の CSV には次が追加される。

- `predicted_label`
- `predicted_label_name`

再学習系ではさらに以下が入る。

- `training_label`
- `training_label_name`
- `training_label_origin`
- `manual_label_name`
- `manual_label_id`

---

## 21. Tacview へどのように書き戻しているか

`src/maneuver_acmi_export.py` が Annotated ACMI を作る。

### 21.1 基本の考え方

各窓について、以下のどこをその窓の代表時刻とするかを決める。

- `start`
- `center`
- `end`

デフォルトは `center` である。

その代表時刻に最も近い ACMI フレームを見つけ、そのフレーム直後にカスタムプロパティ行を差し込む。

### 21.2 書き戻す情報

主ラベルだけではなく、可能な限り多くの情報を書き戻している。

- 予測ラベル
- ルールベースラベル
- 採用された主ラベル
- 属性一覧
- `Uncertain` 理由
- `Transition` 理由
- 主ラベル成立フラグ
- 属性ブール
- 各主ラベルスコア
- 上位 1 位 / 2 位スコア
- 主要数値特徴量

### 21.3 Tacview Raw Telemetry 用の命名規則

Tacview の Raw Telemetry 画面では `Name` ソートすることが多いため、見やすさのために番号付き接頭辞を使っている。

代表例は以下。

- `Maneuver010PredLabel`
- `Maneuver011PredLabelId`
- `Maneuver020RuleLabel`
- `Maneuver021RuleLabelId`
- `Maneuver030MainLabel`
- `Maneuver031MainLabelId`
- `Maneuver040Attributes`
- `Maneuver041AttributeCount`
- `Maneuver050UncertainReason`
- `Maneuver051TransitionReason`
- `Maneuver190ActiveFlags`
- `Maneuver200FlagStraightLevel`
- `Maneuver290ActiveAttributes`
- `Maneuver300AttrAccelerating`
- `Maneuver400ScoreDive`
- `Maneuver410TopScoreLabel`
- `Maneuver411TopScore`
- `Maneuver412SecondScoreLabel`
- `Maneuver413SecondScore`
- `Maneuver500GMean`

これにより、Tacview 上で「モデル予測」「ルール判定」「採用ラベル」「理由」「スコア」「数値特徴量」がまとまりで見える。

---

## 22. Tacview アドオンによる手動ラベル修正

Tacview から直接ラベル修正できるように、Lua アドオンも入れている。

### 22.1 できること

`ManualLabelEditor/main.lua` では以下ができる。

1. Tacview 上で固定翼機を右クリック
2. `Manual Label` メニューから主ラベルを選ぶ
3. その時刻・その機体に対する人手主ラベルを記録する
4. `manual_label_edits.csv` に追記する
5. Raw Telemetry に人手ラベル列も書き込む

### 22.2 アドオンが書く Telemetry

代表的には以下を書き込む。

- `Maneuver015HumanLabel`
- `Maneuver016HumanLabelId`
- `Maneuver017HumanAction`
- `Maneuver018HumanEditedUtc`
- `Maneuver019HumanSource`

### 22.3 編集 CSV の中身

手動修正は `manual_label_edits.csv` に保存され、典型的に以下を含む。

- `edit_index`
- `edit_action`
- `edited_utc`
- `aircraft_id`
- `aircraft_name`
- `sample_time`
- `window_start_hint`
- `window_end_hint`
- `manual_label_name`
- `manual_label_id`
- `previous_predicted_label`
- `previous_rule_label`
- `previous_main_label`
- `addon_version`

---

## 23. 手動修正を再学習にどう使うか

`src/maneuver_manual_labels.py` と `src/maneuver_retrain_from_labels.py` がここを担当する。

### 23.1 どの窓に人手修正を当てるか

`apply_manual_label_edits()` では次の順で対応窓を探す。

1. `aircraft_id` が一致する窓だけを候補にする
2. `source_acmi` があればそれでも絞る
3. `window_start_hint` と `window_end_hint` があれば、それに一致する窓を優先する
4. それが無ければ `sample_time` に最も近い窓中心を選ぶ
5. 許容距離 `match_tolerance_sec` を超える場合は unmatched にする

デフォルト許容距離は `2.5 sec` である。

### 23.2 再学習時の教師列

元のルールラベルに対して、次の列を持たせる。

- `training_label`
- `training_label_name`
- `training_label_origin`

動作は次の通り。

- 手動修正が無ければ `training_label = rule label`
- 手動修正があれば `training_label = manual label`
- `clear` 操作なら rule に戻す

したがって再学習時の教師は `label` ではなく、最終的に組み立て直された `training_label` になる。

### 23.3 再学習の出力

再学習では主に次を出す。

- `maneuver_features_with_manual_labels.csv`
- `manual_label_unmatched.csv`
- 再学習済み model joblib
- 評価画像

---

## 24. 実際にユーザが受け取る結果

このプログラムを回したあと、ユーザが確認する主要成果物は以下である。

### 24.1 ラベル付き特徴量 CSV

最も重要な成果物で、機械学習にも目視レビューにも使う。

見るポイントは次。

- ラベル分布
- `Transition` と `Uncertain` の割合
- 主ラベル条件フラグ
- スコアの競合具合
- G 系特徴量の健全性
- 人手修正の反映状況

### 24.2 モデルファイル

学習済みモデルそのもの。別 ACMI への再推論に使う。

### 24.3 評価画像

混同行列と特徴量重要度。モデルが現ルールをどの程度再現できているか、どの特徴に依存しているかを見る。

### 24.4 Annotated ACMI

Tacview 上で実機動とラベルを照合するための出力である。実運用では、これが最も価値の高い確認手段の一つになる。

### 24.5 手動修正ログ

Tacview 上で行った監修結果そのものであり、改善サイクルの種データになる。

---

## 25. この設計の長所

- 相手機情報なしでも初期ラベル生成ができる
- 5 秒主窓と 2 秒補助窓で、安定機動と短周期機動を両方ある程度拾える
- ルールベースと ML を組み合わせているので、初期構築が速い
- CSV がかなりリッチで、誤判定解析がしやすい
- Tacview に判定根拠まで戻せる
- Tacview 上で修正して再学習できる

---

## 26. この設計の限界

### 26.1 単機運動学だけを見ている

戦術的意味のあるラベルは厳密には扱えない。単に「どう動いているか」を見ているだけで、「何を意図していたか」は分からない。

### 26.2 教師が最初はルールベース

初期学習では、モデルは人手真値ではなくルールを学ぶ。ルールの偏りや誤りは、そのままモデルにも入りやすい。

### 26.3 train/test 分割が窓単位

厳密な sortie 汎化評価ではない。同じログ由来の近い窓が train/test に分かれる可能性がある。

### 26.4 `g_load` の品質に依存する

`g_load` は位置差分からの近似なので、座標ノイズやサンプリング粗さの影響を受ける。とはいえ、法線加速度ベースにしたことで、以前の全加速度ノルム方式よりはかなり安定している。G 系属性の妥当性確認では、元 ACMI の時間分解能と座標品質が依然として重要である。

### 26.5 `Transition` は簡易実装

現在の `Transition` は、厳密な系列モデルではなく、フラグ多重成立・スコア差・前半後半不一致を使った簡易判定である。

---

## 27. 今後さらに改善しやすい箇所

このコードベースで次に強化しやすいのは以下である。

- sortie 単位やファイル単位の train/test 分割
- クラス不均衡に対する再重み付けやサンプリング
- `Transition` のスコアベース高度化
- 2 秒補助窓特徴の追加拡張
- `g_load` 品質チェックの強化
- 相手機情報を導入した戦術ラベル拡張
- 時系列モデルへの置換

---

## 28. まとめ

このプログラムは、Tacview ACMI から単機運動時系列を取り出し、前処理、短時間窓特徴量抽出、ルールベースラベリング、ML 学習、Tacview への書き戻し、人手修正、再学習までを一つの流れにまとめたパイプラインである。

技術的な中身を一言で言うと、以下の構成になっている。

- 入力: Tacview ACMI
- 前処理: 5 Hz リサンプリング、補間、移動平均、微分
- 窓設計: 5 秒主窓 + 2 秒補助窓
- ラベル設計: 排他的主ラベル + 非排他的補助属性
- 学習器: Random Forest または XGBoost
- 出力: CSV、モデル、評価画像、Annotated ACMI
- 改善ループ: Tacview 手動修正 -> 再学習

したがってこのプロジェクトは、「ラベル付けされた航空機機動データを実務的に育てていくための基盤」として位置付けられる。
