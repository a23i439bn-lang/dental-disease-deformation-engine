# 卒業研究 進捗スライド用メモ

このファイルは、そのままスライドへ写しやすいように現状の進捗を整理したメモです。

---

## 1. 研究タイトル

- 歯科矯正の視診トレーニング用症例画像生成
- 正常顔画像から顔貌疾患画像を生成する研究
- 幾何変形と生成 AI を組み合わせた視診教材生成

---

## 2. 研究目的

- 患者プライバシーを侵害しない症例画像生成
- 稀な症例も含む多様な症例の疑似生成
- 視診教育に使える顔貌・口腔所見の生成
- 将来は実症例統計に基づく変形モデルへ移行

---

## 3. 現在の設計思想

- diffusion に全部を任せない
- geometry engine で疾患構造を決める
- diffusion は写真感と自然化に限定する

まとめると:

`解剖学ベース変形 + 生成 AI 自然化`

---

## 4. 現在のパイプライン

```text
正常顔画像
  -> MediaPipe landmark 抽出
  -> 疾患別 landmark deformation
  -> dense warp / remap
  -> 必要に応じて Stable Diffusion Inpaint 自然化
  -> 最終症例画像
```

---

## 5. 実装済みの2本柱

### 口腔 ROI ベース

- `spacing`
- `openbite`
- `protrusion`
- `caries`

### 顔貌疾患ベース

- `mandibular_protrusion`
- `maxillary_protrusion`
- `chin_deviation_left`
- `chin_deviation_right`
- `occlusal_plane_cant`

---

## 6. 顔貌側の進捗

- `disease_templates/` へテンプレート構造を分離
- landmark 群、顔領域 mask、warp を共通化
- 顔貌疾患ごとの geometric template を実装
- `infer_face_dysmorph.py` を入口専用に整理

研究上の意味:

- 疾患らしさをコード上で説明できる
- 将来の統計テンプレート化に接続しやすい
- 教育用の ablation がしやすい

---

## 7. 顔貌側で追加した自然化

- 顔貌変形後に Stable Diffusion Inpaint を局所適用
- 編集領域は disease mask に限定
- `sd-render-main` により元顔の文脈を保った condition image を作成

重要点:

- 形は geometry が主
- diffusion は texture / photo realism の補助

---

## 8. 疾患別自然化の進捗

### `chin_deviation`

- 専用 prompt を追加
- 偏位側 jawline と chin 周辺を重点化する edit mask を追加

### `occlusal_plane_cant`

- 専用 prompt を追加
- 口角高低差と mouth line 周辺を重点化する edit mask を追加

---

## 9. データ整理の進捗

- `data/` を再整理
- manifest 類を `data/manifests/` に集約
- 実症例受け皿として `data/clinical_cases/` を作成

現在の主な構成:

```text
data/
  inputs/
  inputs_normal/
  references/
  manifests/
  clinical_cases/
```

---

## 10. 阪大データ到着後の次段階

```text
疾患画像
  -> landmark 抽出
  -> 正常群平均との差分計算
  -> 平均 delta 作成
  -> statistical disease template 化
```

ここで研究が一段上がる:

- 今: 手作り変形
- 次: 実症例統計ベース変形

---

## 11. 現在の強み

- heuristic でも顔貌疾患テンプレート構造まで実装済み
- 口腔 ROI 系と顔貌系を分けて考えられている
- geometry first の思想が一貫している
- 実症例統計ベースへ移るためのデータ構造も準備済み

---

## 12. 現在の課題

- 顔貌 edit mask はまだ重みマップ化の余地がある
- chin deviation の中央連結と occlusal cant の口角差強調はさらに改善余地あり
- 実症例統計はまだ未導入
- 視診教育としての評価実験はこれから整理が必要

---

## 13. 今後やること

1. `clinical_cases/` を読む解析スクリプトを作る
2. 正常群平均 landmark を作る
3. 疾患別平均 delta を算出する
4. heuristic template と statistical template を比較する
5. 視診教育への有効性を評価する

---

## 14. ひとことでまとめ

- 口腔 ROI ベース生成はすでに一通り動作
- 顔貌疾患生成はテンプレート構造へ移行済み
- 次の研究の核は `実症例統計ベース変形` への移行
