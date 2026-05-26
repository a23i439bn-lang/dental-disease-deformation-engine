# 実験レポート

このメモは、現時点での実装進捗と研究上の意味を簡潔にまとめたものです。

## 現在の到達点

研究は大きく 2 本で進んでいます。

1. 幾何変形で疾患構造を作る
2. Diffusion / Inpainting で写真感を補う

現在は特に `顔貌疾患生成` 側の整備が進みました。

## 顔貌側の進捗

- `infer_face_dysmorph.py` を入口専用に整理
- `disease_templates/` を導入し、疾患別構造へ移行
- `mandibular_protrusion`
- `maxillary_protrusion`
- `chin_deviation_left`
- `chin_deviation_right`
- `occlusal_plane_cant`

を geometric template として実装

さらに、

- `utils/face_landmarks.py`
- `utils/face_regions.py`
- `utils/face_warp.py`
- `utils/face_refine.py`

を分離し、顔貌生成エンジンとして再利用しやすい構造にした。

## 顔貌自然化の進捗

顔貌側にも Stable Diffusion Inpaint を導入した。

- 疾患領域だけを局所編集
- `sd-render-main` により元顔の文脈を保持
- `chin_deviation` と `occlusal_plane_cant` に専用 prompt / edit mask を追加

この方針により、形は geometry が主、自然感は diffusion が補助、という役割分担が維持できている。

## 研究上の意味

この段階で重要なのは、単なる画像生成ではなく、

- 疾患っぽさ
- 解剖学的一貫性
- 視診教育との関連

をコード構造として説明できるようになったことです。

特に `disease_templates/` 化によって、各疾患で

- どの landmark 群を動かすか
- どの領域を固定するか
- どの mask で自然化するか

が読み取れるようになりました。

## 次の研究段階

次の大きな段階は、`clinical_cases/` を使った実症例統計ベース変形です。

想定フロー:

```text
疾患画像
  -> landmark 抽出
  -> 正常平均との差分計算
  -> 平均 delta 作成
  -> statistical template 化
```

ここが進むと、

- 現在: 手作り変形
- 次: 実症例統計ベース変形

という形で研究が一段上がります。

## 現在の課題

- face edit mask は二値より重みマップ化した方がさらに良い
- 疾患別自然化はまだ改善余地がある
- 実症例統計はまだ未導入
- 視診教育としての評価実験設計はこれから深める必要がある
