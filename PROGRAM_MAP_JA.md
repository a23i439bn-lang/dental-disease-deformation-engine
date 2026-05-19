# 主要プログラム一覧

## 今いちばん重要

- `infer_mouth_template.py`
  学習なし・Diffusionなしで、口ROIと歯マスクを使って疾患テンプレを直接当てる確認用スクリプト。
  今の「まず形だけ確認する」段階の主役。

## 変形確認系

- `train_deformation.py`
  DeformationPolicyNetwork を teacher target 付きで学習する。
  「疾患らしい形」を学習で出したいときの中心。

## 全体推論・全体学習

- `infer.py`
  学習済みの変形 + テクスチャ + 描画の全体推論を回す。
  rendered画像を出す本番寄りの入口。

- `train.py`
  パイプライン全体を学習する。
  形状変形、テクスチャ、描画条件付けまで含めて学習したいときに使う。

- `generate.py`
  研究用の大きな実験スクリプト。
  参照画像、口領域マスク、ランドマーク変形、最終画像生成まで含む。

## 前処理・中核部品

- `dataset/dataset_builder.py`
  学習/推論前処理。
  画像、ランドマーク、疾患ラベル、severity を整える。

- `diffusion/pipeline.py`
  Dense warp、texture branch、renderer を束ねる中核コード。

## モデル群

- `models/deformation_policy.py`
  ランドマークをどの方向へどれだけ動かすかを予測する主役モデル。

- `models/disease_encoder.py`
  疾患名と severity を embedding に変換する。

- `models/texture_branch.py`
  変形後画像へ見た目の変化を足す。

- `models/severity_policy.py`
  severity 制御を扱う補助モデル。

- `models/clip_loss.py`
  生成結果が疾患表現に合っているかを測る補助loss。
