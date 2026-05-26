# 卒業研究1

このリポジトリは、歯科矯正の視診トレーニング用症例画像生成のための研究コードです。

現在の中心テーマは、正常顔画像から顔貌疾患画像を生成することです。
特に、

- 下顎前突
- 上顎前突
- オトガイ偏位
- 咬合平面傾斜

のような顔貌特徴を、解剖学ベース変形と生成 AI 自然化の組み合わせで表現することを目指しています。

## 研究方針

重要な方針は次の 2 点です。

1. 疾患構造は geometry engine で決める
2. diffusion は写真感と自然化に限定する

つまり、

`解剖学ベース変形 + 生成 AI 自然化`

を研究の基本設計にしています。

## 現在の主な実装

### 顔貌疾患生成

- `infer_face_dysmorph.py`
  顔貌疾患生成の入口
- `disease_templates/`
  疾患別テンプレート構造
- `utils/face_landmarks.py`
  landmark 検出
- `utils/face_regions.py`
  顔領域定義
- `utils/face_warp.py`
  dense warp / remap / blend
- `utils/face_refine.py`
  Stable Diffusion Inpaint による局所自然化

対応済み顔貌疾患:

- `mandibular_protrusion`
- `maxillary_protrusion`
- `chin_deviation_left`
- `chin_deviation_right`
- `occlusal_plane_cant`

### 口腔 ROI ベース生成

- `infer_mouth_template.py`

対応済み ROI 疾患:

- `spacing`
- `openbite`
- `protrusion`
- `caries`

## 実行例

### 顔貌疾患生成

```powershell
.\.venv\Scripts\python.exe infer_face_dysmorph.py `
  --input data\inputs\face.png `
  --disease mandibular_protrusion `
  --severity 0.8 `
  --output-dir outputs\face_demo
```

### 顔貌疾患生成 + 自然化

```powershell
.\.venv\Scripts\python.exe infer_face_dysmorph.py `
  --input data\inputs\face.png `
  --disease occlusal_plane_cant `
  --severity 0.8 `
  --enable-sd-refine `
  --sd-render-main `
  --output-dir outputs\face_refine_demo
```

### 口腔 ROI ベース生成

```powershell
.\.venv\Scripts\python.exe infer_mouth_template.py `
  --input data\inputs\face.png `
  --disease spacing `
  --severity 1.0 `
  --output-dir outputs\mouth_demo
```

## データ構成

```text
data/
  inputs/          単発実験用の入力画像
  inputs_normal/   正常顔画像群
  references/      参照症例画像
  manifests/       研究用 manifest
  clinical_cases/  実症例統計ベース変形へ移行するための箱
```

`clinical_cases/` は、阪大データなどの実症例画像を受けるためのフォルダです。
今後はここから

`疾患画像 -> landmark 抽出 -> 正常平均との差分 -> 平均 delta 作成`

を行い、手作り変形から実症例統計ベース変形へ移行する予定です。

## 現在の位置づけ

今の顔貌生成は heuristic / anatomical template ベースです。
次の研究上の大きな段階は、

- 現在: 手作り変形
- 次: 実症例統計ベース変形

への移行です。
