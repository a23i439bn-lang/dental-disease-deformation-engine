# clinical_cases

阪大データなどの実症例画像を置くためのフォルダです。

このフォルダは、将来的に

`疾患画像 -> landmark抽出 -> 正常平均との差分計算 -> 平均delta作成`

を行い、手作り変形から実症例統計ベース変形へ移行するための入口です。

## 目的

- 手作り template と分離して実症例データを管理する
- 疾患ごとの landmark delta を統計化する
- 平均 delta だけでなく分散や左右差も研究対象にする
- 将来の阪大データ到着後にそのまま解析へ進める

## フォルダ構成

- `normal/`
  正常群の顔画像
- `mandibular_protrusion/`
  下顎前突の実症例画像
- `maxillary_protrusion/`
  上顎前突の実症例画像
- `chin_deviation_left/`
  左方向オトガイ偏位の実症例画像
- `chin_deviation_right/`
  右方向オトガイ偏位の実症例画像
- `occlusal_plane_cant/`
  咬合平面傾斜の実症例画像
- `metadata/`
  症例属性や匿名化済み付帯情報
- `manifests/`
  学習・解析用 manifest

## 想定する次段階

1. 各フォルダに匿名化済み画像を格納する
2. MediaPipe などで landmark を抽出する
3. 正常群平均 landmark を作る
4. 疾患群平均との差分 delta を算出する
5. 疾患 template を heuristic 版から statistical 版へ更新する

## 研究上の意味

今の顔貌生成は heuristic な変形です。
ここに実症例統計を入れることで、

- 疾患っぽさ
- 解剖学的一貫性
- 教育的妥当性

を一段上げられます。
