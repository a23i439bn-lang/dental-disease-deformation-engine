# プログラム対応表

このファイルは、「どのファイルが何のためにあるのか」を短時間で把握するための一覧です。

## まず見るファイル

- `README.md`
  リポジトリ全体の目的と流れを説明します。
- `infer_mouth_template.py`
  現在の主力確認スクリプトです。
- `SUPPORTED_DISEASES_JA.md`
  対応疾患を確認できます。

## 学習系

- `train_deformation.py`
  口元の形だけを学習するフェーズ1です。
- `train.py`
  Stable Diffusion ベースの最小学習です。

## 推論・生成系

- `infer_mouth_template.py`
  口 ROI 抽出、歯マスク推定、幾何変形、局所 inpaint をまとめた確認用スクリプトです。
- `infer.py`
  学習済みモデルを使う最小推論です。
- `generate.py`
  ControlNet や参照画像も含めた大きめの生成パイプラインです。

## データ系

- `dataset/dataset_builder.py`
  学習・推論用データセットを組み立てます。
- `data/manifests/README.md`
  manifest の置き場を説明します。
- `data/dataset/README.md`
  学習データの配置ルールを説明します。
- `data/inputs_normal/README.md`
  正常顔入力画像の置き場を説明します。
- `data/SOURCES.md`
  使用データの出典メモです。

## モデル系

- `models/deformation_policy.py`
  ランドマーク変形量を予測するモデルです。
- `models/disease_encoder.py`
  疾患名と severity を埋め込みに変えます。
- `models/texture_branch.py`
  見た目補助の分岐です。
- `models/severity_policy.py`
  severity 制御用モデルです。
- `models/clip_loss.py`
  疾患らしさに寄せる補助 loss です。

## Diffusion 系

- `diffusion/pipeline.py`
  幾何変形、テクスチャ、Diffusion をつなぐ中核コードです。
- `diffusion/controlnet_conditioning.py`
  ControlNet に渡す条件画像を作ります。
- `diffusion/disease_attention.py`
  疾患埋め込みを attention 側へ渡す補助コードです。

## 補助コード

- `utils/disease_priors.py`
  疾患ごとの教師変形やルールを定義します。
- `utils/prompts.py`
  Diffusion 用プロンプトを組み立てます。
- `utils/factory.py`
  モデルやパイプラインの生成をまとめます。
- `utils/checkpointing.py`
  checkpoint の保存・読込を補助します。
- `utils/lora.py`
  LoRA 関連の補助です。
- `debug_visualize_delta.py`
  ランドマーク変形の見え方を画像で確認する補助です。
