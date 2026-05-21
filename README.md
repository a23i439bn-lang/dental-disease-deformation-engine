# 卒業研究1 - コピー

このリポジトリは、正常な顔画像から口元の変形や局所生成を使って、歯科疾患らしい見た目を合成するための研究用コードです。

いまの中心テーマは次の2段構成です。

1. 幾何変形で「どこをどう変えるか」を制御する
2. Diffusion / Inpainting で「写真として自然に見える質感」を補う

## 全体の流れ

```text
正常顔
  ↓
顔・口ランドマーク検出
  ↓
口 ROI 抽出
  ↓
疾患ごとの幾何変形や局所マスク生成
  ↓
必要に応じて Stable Diffusion で局所自然化
  ↓
疾患画像として合成
```

## 主要ファイル

- `infer_mouth_template.py`
  現在の主力確認スクリプトです。口 ROI ベースで `protrusion`、`openbite`、`spacing`、`caries` を試せます。
- `generate.py`
  より大きな生成パイプラインです。ControlNet や参照画像、口元マスクなども含めた実験向けです。
- `train_deformation.py`
  Diffusion を使わず、まず口元の形だけを学習させるフェーズ1の学習スクリプトです。
- `train.py`
  Stable Diffusion ベースの最小学習スクリプトです。
- `infer.py`
  学習済みパイプラインや U-Net を使って推論する最小スクリプトです。
- `PROGRAM_MAP_JA.md`
  ファイルごとの役割を日本語で一覧化したガイドです。
- `SUPPORTED_DISEASES_JA.md`
  どの疾患がどのスクリプトで扱えるかを整理した表です。
- `EXPERIMENT_REPORT_JA.md`
  実験条件や気づきをまとめたメモです。

## ディレクトリ構成

```text
data/
  inputs_normal/      正常顔の入力画像
  dataset/            学習用データ
  manifests/          データセット manifest

dataset/
  dataset_builder.py  データセット構築

diffusion/
  pipeline.py                     幾何変形 + テクスチャ + Diffusion の中核
  controlnet_conditioning.py      ControlNet 用条件画像の構築
  disease_attention.py            疾患埋め込みを attention に渡す補助

models/
  deformation_policy.py   ランドマーク変形量を予測するモデル
  disease_encoder.py      疾患名と severity を埋め込みへ変換
  texture_branch.py       見た目補助用の分岐
  severity_policy.py      severity 制御用モデル
  clip_loss.py            疾患らしさを補助する loss

utils/
  disease_priors.py   疾患ごとの教師変形・ルール
  prompts.py          Diffusion 用プロンプト生成
  factory.py          モデルやパイプラインの生成
  checkpointing.py    checkpoint の保存と再開
  lora.py             LoRA 補助
```

## よく使う実行例

### 口 ROI ベースで疾患を試す

```powershell
.\.venv\Scripts\python.exe infer_mouth_template.py `
  --input data\inputs\face.png `
  --disease spacing `
  --severity 1.0 `
  --output-dir outputs\demo_spacing
```

### 局所 Inpaint まで使う

```powershell
.\.venv\Scripts\python.exe infer_mouth_template.py `
  --input data\inputs\face.png `
  --disease protrusion `
  --severity 1.0 `
  --enable-sd-refine `
  --sd-local-edit `
  --sd-render-main `
  --output-dir outputs\demo_protrusion
```

### 幾何変形だけを学習する

```powershell
.\.venv\Scripts\python.exe train_deformation.py --device cpu
```

## 初めて読むときのおすすめ順

1. `README.md`
2. `PROGRAM_MAP_JA.md`
3. `infer_mouth_template.py`
4. `utils/disease_priors.py`
5. `diffusion/pipeline.py`
