# Disease-Aware Research Pipeline

疾患条件付きの顔口腔変形・異常テクスチャ生成・Diffusion 条件付けを統合した研究用実装です。

## 構成

- `models/disease_encoder.py`: multi-hot 疾患埋め込み + severity 埋め込み
- `models/deformation_policy.py`: landmark graph ベースの DeformationPolicyNetwork
- `models/texture_branch.py`: FiLM 条件付き U-Net 風 TextureBranch
- `models/clip_loss.py`: CLIP disease loss
- `diffusion/disease_attention.py`: disease-aware cross attention processor
- `diffusion/pipeline.py`: Dense Warp + Texture + Disease-aware Diffusion 統合
- `dataset/dataset_builder.py`: 研究用 manifest / dataset 構築
- `train.py`: 学習
- `infer.py`: 推論
- `config.yaml`: ハイパーパラメータ

## 学習

```powershell
.\.venv\Scripts\python.exe train.py --config config.yaml --device cpu
```

## 推論

```powershell
.\.venv\Scripts\python.exe infer.py `
  --config config.yaml `
  --input data\inputs\sample_face_256.png `
  --disease 虫歯 `
  --severity 0.6 `
  --device cpu
```

## 補足

- Diffusers の事前学習モデルがローカルで使える場合は disease-aware attention を注入します。
- ローカルにモデルがない場合でも、研究コード確認用の differentiable fallback renderer で動作します。
- 既定疾患は `虫歯`、`出っ歯`、`すきっ歯` の 3 種です。
