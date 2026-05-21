# 卒業研究 進捗報告スライド原稿

このファイルは、そのまま PowerPoint に貼り付けて使えるように作った進捗報告用の原稿です。

---

## 1. タイトル

- 卒業研究 進捗報告
- 顔画像からの歯科疾患合成
- 幾何変形と Diffusion を組み合わせた口腔画像生成

発表者名、日付、所属はここに追記してください。

---

## 2. 研究の目的

- 正常な顔画像から、歯科疾患らしい口元画像を生成したい
- ただ自然に見えるだけでなく、疾患の形状を制御できる方法を目指す
- 研究上は「形の制御」と「見た目の自然さ」を分けて設計することが重要

話すポイント:
- 最初から生成 AI に全部任せると、どこが原因で失敗したか分からなくなる
- そのため、まず幾何変形で疾患構造を作り、後から Diffusion で自然化する方針を採用した

---

## 3. 現在の全体パイプライン

```text
正常顔
  ↓
顔・口ランドマーク検出
  ↓
mouth ROI 抽出
  ↓
疾患ごとの幾何変形 / 疾患マスク生成
  ↓
Stable Diffusion Inpaint による局所自然化
  ↓
最終合成画像
```

話すポイント:
- いまは warp を最終出力ではなく、Diffusion に渡す条件として使う方向へ移行中

---

## 4. 実装済みの主な要素

- MediaPipe による顔・口ランドマーク検出
- mouth ROI 抽出
- 歯マスク推定
- 疾患別テンプレ変形
  - `protrusion`
  - `openbite`
  - `spacing`
  - `caries`
- Stable Diffusion の img2img 接続
- Stable Diffusion Inpaint による local edit
- `condition image + disease mask` を使う構成

関連ファイル:
- `infer_mouth_template.py`
- `utils/disease_priors.py`
- `diffusion/pipeline.py`

---

## 5. ここまでの考え方

- まず「疾患の形」を作る
- 次に「写真として自然な見た目」を作る
- 形の制御は幾何変形が担当
- 質感、境界、影、口腔内の暗部などは Diffusion が担当

話すポイント:
- ルールベース変形だけではリアリティに限界がある
- ただし、幾何変形は無駄ではなく、Diffusion の条件として重要

---

## 6. spacing で分かったこと

- 単純な ROI 全体の滑らかな変形では、数値上変形していても「すきっ歯」に見えにくい
- 原因は、前歯 2 本が独立して動いているように見えず、中央がなめらかに引き伸ばされるため
- そのため、`spacing` では次の方向へ改良した
  - 前歯 2 本を意識したマスク生成
  - 前歯 2 本の cut-and-paste 的な再配置
  - gap 部の暗部補完
  - local inpaint による自然化

---

## 7. protrusion で分かったこと

- `protrusion` は `spacing` より ROI 変形と相性が良い
- ただし、自然に見せるには局所 inpaint の補助が必要
- 現在は `condition image + disease mask` を使って上顎前歯周辺だけを編集している

実行結果:
- 出力先: `outputs/sd_refine_protrusion_render_main`

---

## 8. 最近の実験構成

- `sd-local-edit`
  - 疾患部だけを Stable Diffusion Inpaint で編集
- `sd-render-main`
  - warp 結果をそのまま最終画像にせず、condition image として利用

この構成の狙い:
- Diffusion が不自然な warp を消すのではなく
- 疾患構造をヒントに、より自然な写真表現を描くようにする

---

## 9. 実験結果の例: spacing

参照ファイル:
- `outputs/sd_refine_spacing_render_main/mouth_roi.png`
- `outputs/sd_refine_spacing_render_main/warped_roi.png`
- `outputs/sd_refine_spacing_render_main/condition_roi.png`
- `outputs/sd_refine_spacing_render_main/disease_edit_mask.png`
- `outputs/sd_refine_spacing_render_main/refined_roi.png`
- `outputs/sd_refine_spacing_render_main/final_output.png`

主要数値:
- disease: `spacing`
- severity: `1.0`
- `delta_abs_max = 52.87`
- `disease_edit_pixels = 7873`
- `renderer_condition_enabled = true`

話すポイント:
- gap を条件として与え、局所 inpaint で自然化している

---

## 10. 実験結果の例: protrusion

参照ファイル:
- `outputs/sd_refine_protrusion_render_main/mouth_roi.png`
- `outputs/sd_refine_protrusion_render_main/warped_roi.png`
- `outputs/sd_refine_protrusion_render_main/condition_roi.png`
- `outputs/sd_refine_protrusion_render_main/disease_edit_mask.png`
- `outputs/sd_refine_protrusion_render_main/refined_roi.png`
- `outputs/sd_refine_protrusion_render_main/final_output.png`

主要数値:
- disease: `protrusion`
- severity: `1.0`
- `delta_abs_max = 10.11`
- `disease_edit_pixels = 12243`
- `renderer_condition_enabled = true`

話すポイント:
- spacing よりも geometry と相性はよいが、まだ強い疾患感の表現は改善余地がある

---

## 11. 現時点の到達点

- 口元 ROI ベースの疾患生成パイプラインを構築できた
- 疾患ごとの幾何変形と local inpaint を接続できた
- `warp を最終画像として使う` から `warp を condition として使う` へ方針転換できた
- `spacing` と `protrusion` で、条件画像ベースの Inpaint 実験まで進んだ

---

## 12. 現在の課題

- `spacing`
  - まだ「前歯 2 本が独立して離れた」見え方が弱い場合がある
- `openbite`
  - 上下接触の消失をもっと明確に条件化する必要がある
- `protrusion`
  - 上顎前歯の押し出し感をさらに強める必要がある
- 共通課題
  - Nano Banana のような強い写真感には、より強い条件設計が必要

---

## 13. 次にやること

- 歯 instance mask の精密化
- disease mask の改善
- `spacing` の前歯 2 本分離をさらに明確化
- `openbite` の接触消失領域の明確化
- ControlNet の追加
  - edge map
  - tooth segmentation map
  - gap mask

---

## 14. まとめ

- ルールベース変形だけでは限界がある
- ただし、幾何変形は Diffusion の条件として非常に重要
- 現在は `condition image + local inpaint` 構成まで到達している
- 今後は `歯 instance aware + disease mask aware + ControlNet` の方向へ進める

---

## 15. 補足資料

必要なら最後に以下を添付してください。

- `README.md`
- `PROGRAM_MAP_JA.md`
- `SUPPORTED_DISEASES_JA.md`
- `EXPERIMENT_REPORT_JA.md`
- `outputs/` 以下の代表画像
