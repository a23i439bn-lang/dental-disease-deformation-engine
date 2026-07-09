
# プログラム対応表

このファイルは、現在の卒業研究コードをどこから読めばよいかをまとめた対応表です。

## まず見るファイル

- `README.md`
  リポジトリ全体の目的と現在の方針
- `infer_face_dysmorph.py`
  顔貌疾患生成の実行入口
- `infer_mouth_template.py`
  口腔 ROI ベース生成の実行入口
- `SUPPORTED_DISEASES_JA.md`
  現在扱っている疾患と生成方式の一覧

## 顔貌疾患生成

- `infer_face_dysmorph.py`
  顔画像入力、landmark 検出、疾患テンプレート呼び出し、warp、任意の SD 自然化までを実行
- `disease_templates/__init__.py`
  顔貌疾患テンプレートの登録
- `disease_templates/base.py`
  顔貌テンプレート共通の基底クラス
- `disease_templates/mandibular_protrusion.py`
  下顎前突テンプレート
- `disease_templates/maxillary_protrusion.py`
  上顎前突テンプレート
- `disease_templates/chin_deviation_left.py`
  左オトガイ偏位テンプレート
- `disease_templates/chin_deviation_right.py`
  右オトガイ偏位テンプレート
- `disease_templates/occlusal_plane_cant.py`
  咬合平面傾斜テンプレート

## 顔貌生成の共通処理

- `utils/face_landmarks.py`
  MediaPipe Face Landmarker / FaceMesh による landmark 検出と画像 I/O
- `utils/face_regions.py`
  顔領域 index 群、下顔面・中顔面・咬合平面傾斜用マスク定義
- `utils/face_warp.py`
  dense displacement field、OpenCV remap、blend、debug 描画
- `utils/face_refine.py`
  顔貌疾患用 Stable Diffusion Inpaint 自然化
  `chin_deviation` と `occlusal_plane_cant` 用の疾患別 edit mask もここにある

## 口腔 ROI ベース生成

- `infer_mouth_template.py`
  口腔 ROI 抽出、疾患別変形、texture cue、SD img2img / inpaint まで実行
- `utils/disease_priors.py`
  既存の口腔 structural prior と teacher target 生成

## 学習関連

- `train_deformation.py`
  teacher target landmark を使った変形学習
- `train.py`
  diffusion renderer を含む学習系の入口
- `infer.py`
  学習済みモデル推論の入口

## モデル関連

- `models/deformation_policy.py`
  landmark delta を予測する変形モデル
- `models/disease_encoder.py`
  疾患名と severity を埋め込むモデル
- `models/texture_branch.py`
  texture 補助分岐
- `models/severity_policy.py`
  severity 自動探索用ポリシー
- `models/clip_loss.py`
  疾患らしさを補助する loss

## Diffusion 関連

- `diffusion/pipeline.py`
  Dense warp、texture branch、ControlNet、diffusion renderer の中核
- `diffusion/controlnet_conditioning.py`
  ControlNet 用条件画像生成
- `diffusion/disease_attention.py`
  疾患条件を attention に入れる補助

## データ関連

```text
data/
  README.md
  SOURCES.md
  inputs/                  単発実験用の入力画像
  inputs_normal/           正常顔画像群
  references/              参照症例画像
  manifests/               研究用 manifest とラベル例
  clinical_cases/          実症例統計ベース変形へ移行するための箱

dataset/
  dataset_builder.py       学習用データセット構築
```

## 実症例統計ベース変形の入口

- `data/clinical_cases/`
  阪大データなどの実症例を疾患別に格納するフォルダ
- `data/clinical_cases/README.md`
  `疾患画像 -> landmark 抽出 -> 正常平均との差分 -> 平均 delta 作成`
  という今後の研究フローを記述

## 現在の設計思想

現在の本命は顔貌疾患生成です。

- 幾何変形は heuristic / anatomical template で決める
- diffusion は自然化と質感補助に使う
- 将来は `clinical_cases/` を使って実症例統計ベース template に更新する

## よく使う実行例

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
  --disease chin_deviation_left `
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
