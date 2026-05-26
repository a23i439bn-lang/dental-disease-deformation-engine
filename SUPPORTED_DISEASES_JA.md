# 対応疾患一覧

現在このリポジトリで扱っている疾患と、その生成方式をまとめます。

## 1. 口腔 ROI ベース疾患

実行ファイル:

- `infer_mouth_template.py`

### `spacing`

- 上顎前歯のすきっ歯を想定
- 前歯 2 本の分離、gap cue、local inpaint を使う

### `openbite`

- 前歯部開咬を想定
- 上下方向の gap を geometric warp で作る

### `protrusion`

- 口腔 ROI 内での前突表現
- 上顎前歯の突出感を ROI 幾何変形で出す

### `caries`

- 虫歯の texture cue を想定
- 幾何変形より texture / inpaint の比重が高い

## 2. 顔貌疾患ベース

実行ファイル:

- `infer_face_dysmorph.py`

### `mandibular_protrusion`

- 下顎前突
- chin、jawline、lower lip を連動させて変形
- 下顔面 mask を中心に warp

### `maxillary_protrusion`

- 上顎前突
- upper lip、upper mouth、nose base 周辺を連動させて変形
- 中顔面寄りの変化を追加

### `chin_deviation_left`

- 左方向のオトガイ偏位
- chin の横偏位、jawline asymmetry、lower lip の追従を表現

### `chin_deviation_right`

- 右方向のオトガイ偏位
- 左右反転版の非対称変形

### `occlusal_plane_cant`

- 咬合平面傾斜
- mouth line の傾き、口角高低差、下顔面の補償変形を表現

## 3. 自然化の対応状況

### 口腔 ROI 側

- Stable Diffusion img2img 対応
- Stable Diffusion Inpaint local edit 対応
- `sd-render-main` 対応

### 顔貌側

- Stable Diffusion Inpaint local edit 対応
- `sd-render-main` 対応
- `chin_deviation` と `occlusal_plane_cant` は疾患別 prompt / edit mask を追加済み

## 4. いまの位置づけ

現在の顔貌疾患生成は `heuristic / anatomical template` ベースです。

今後は:

- `data/clinical_cases/normal`
- `data/clinical_cases/<disease>`

に実症例画像を集め、

`疾患画像 -> landmark 抽出 -> 正常平均との差分 -> 平均 delta 作成`

を行って statistical template へ移行する予定です。

## 5. 注意

- `generate.py` の疾患表現は学習・条件付け系の流れを含むため、`infer_mouth_template.py` / `infer_face_dysmorph.py` の heuristic template とは役割が少し異なります。
- 本命は現在 `顔貌疾患生成` 側です。
