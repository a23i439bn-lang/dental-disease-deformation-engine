# 対応疾患一覧

このファイルは、「どのスクリプトでどの疾患を扱えるか」をまとめた一覧です。

## `infer_mouth_template.py`

現在の主力確認スクリプトです。口 ROI を切り出して局所変形や inpaint を試せます。

- `protrusion`
  出っ歯系の変形を試します。
- `openbite`
  前歯の上下接触が減る方向の変形を試します。
- `spacing`
  前歯のすき間を作る方向の変形を試します。
- `caries`
  虫歯らしい局所的な色・質感変化を試します。

## `generate.py`

より大きい生成パイプラインです。疾患名の別名も多めに受け付けます。

- `caries`
  例: `Dental Caries`, `dental_caries`, `caries`
- `protrusion`
  例: `maxillary_protrusion`, `protrusion`, `malocclusion`
- `spacing`
  例: `diastema`, `spacing`
- `normal`
  比較用の正常系
