# 幾何変形エンジン実験メモ

## 実験条件

- 疾患: `openbite`, `protrusion`
- severity sweep: `0.0, 0.4, 0.8, 1.2`
- 入力画像数: 5枚

## 入力画像

1. `007776139.jpg`
   正面笑顔の顔写真
2. `data/inputs/face.png`
   正面笑顔の顔写真
3. `base.jpg`
   正面笑顔の顔写真
4. `data/references/すきっ歯/live268570_bondingpre.jpg`
   口元クローズアップ
5. `data/references/出っ歯/rei_crown44b1-1.jpg`
   口元クローズアップ

## 成功判定の基準

- `detector_mode` が `mediapipe_tasks_face_landmarker`
- `severity_strip.png` で強度変化が追える
- `teeth_pixels` が十分あり、ROI が口元に乗っている

## 結果まとめ

### 成功例

- `exp_openbite_007776139`
  顔全体写真。MediaPipe成功。歯マスク 854 px。
- `exp_openbite_facepng`
  顔全体写真。MediaPipe成功。歯マスク 13592 px。
- `exp_openbite_basejpg`
  顔全体写真。MediaPipe成功。歯マスク 2664 px。
- `exp_protrusion_007776139`
  顔全体写真。MediaPipe成功。歯マスク 854 px。
- `exp_protrusion_facepng`
  顔全体写真。MediaPipe成功。歯マスク 13592 px。
- `exp_protrusion_basejpg`
  顔全体写真。MediaPipe成功。歯マスク 2664 px。

### 失敗例

- `exp_openbite_live268570`
  口元クローズアップ。`heuristic_fallback`。delta が弱く、変化がほぼ出ない。
- `exp_openbite_rei`
  口元クローズアップ。`heuristic_fallback`。delta が弱く、変化がほぼ出ない。
- `exp_protrusion_live268570`
  口元クローズアップ。`heuristic_fallback`。delta が弱く、変化がほぼ出ない。
- `exp_protrusion_rei`
  口元クローズアップ。`heuristic_fallback`。delta が弱く、変化がほぼ出ない。

## 観察

- 顔全体が入っている画像では MediaPipe Face Landmarker が安定しやすい。
- 口元だけのクローズアップ画像では MediaPipe が顔として認識できず、ヒューリスティックに落ちやすい。
- ヒューリスティックに落ちたケースでは `delta_abs_max` が小さく、疾患らしい形変化が十分に出ない。
- `severity_strip.png` は連続制御の可視化として有効。

## 代表出力

- `outputs/exp_openbite_007776139/severity_strip.png`
- `outputs/exp_openbite_facepng/severity_strip.png`
- `outputs/exp_protrusion_007776139/severity_strip.png`
- `outputs/exp_protrusion_facepng/severity_strip.png`
- `outputs/exp_openbite_live268570/severity_strip.png`
- `outputs/exp_protrusion_rei/severity_strip.png`

## 次にやると強いこと

- 顔全体5枚以上でさらに再現性確認を増やす
- 口元クローズアップ専用の検出モードを追加する
- 成功例について `spacing` も同じ5枚で sweep する
- 成功/失敗を表で整理して卒研資料へ流し込む
