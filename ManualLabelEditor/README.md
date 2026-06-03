# Manual Label Editor Add-on

Tacview 上で機体を右クリックし、その時刻の窓に対する手動主ラベルを付与するためのアドオンです。

## できること

- 右クリックした固定翼機に対して主ラベルを手動設定
- `Clear Human Label` で手動上書きを解除
- 編集内容を `manual_label_edits.csv` に追記
- Tacview の Raw Telemetry に `Maneuver015HumanLabel` などの手動ラベル列を書き込み

## 出力 CSV

アドオンは次のような列を出力します。

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

## 使い方

1. このフォルダを Tacview の add-ons ディレクトリに配置
2. Tacview を再起動
3. 注釈付き ACMI を開く
4. 3D 画面上で対象機を右クリック
5. `Manual Label` メニューから主ラベルを選択
6. `manual_label_edits.csv` を Python 側で取り込んで再学習

## Python 側の再学習

```bash
. .venv/bin/activate
python src/maneuver_retrain_from_labels.py \
  results/run1/maneuver_features_labeled.csv \
  --manual-edits-csv Tacview_Addon_Guide/ManualLabelEditor/manual_label_edits.csv \
  --output-dir results/retrain_manual
```
