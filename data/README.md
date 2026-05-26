# data

`data/` 配下の整理方針です。

## 現在の役割

- `clinical_cases/`
  阪大データなどの実症例画像置き場
- `dataset/`
  既存の学習用データ構成
- `inputs/`
  単発実験用の入力画像
- `inputs_normal/`
  正常顔の入力画像群
- `manifests/`
  研究用 manifest とサンプルラベル
- `references/`
  参照症例画像と説明ファイル
- `SOURCES.md`
  参照画像の出典情報

## 整理ルール

- 実画像はフォルダに置く
- manifest やラベル例は `manifests/` に置く
- 実症例統計ベース変形に使う画像は `clinical_cases/` に置く
- 単発の動作確認画像は `inputs/` に置く

## 補足

以前 `data/` 直下にあった

- `research_manifest.json`
- `research_manifest_geometry.json`
- `research_manifest_geometry_smoke.json`
- `dataset_labels_example.json`

は、整理のため `data/manifests/` に移動しています。
