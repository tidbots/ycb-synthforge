# リアルタイム物体検出ガイド

Webカメラを使用したリアルタイム物体検出とトラッキングの使用方法を説明します。

## 概要

`scripts/evaluation/realtime_detection.py` は以下の機能を提供します：

- **ByteTrackによるオブジェクトトラッキング** - 検出のチラつきを軽減
- **座標の移動平均表示** - 1秒間の履歴で平滑化
- **信頼度の平滑化** - 指数移動平均でスコアの急変動を抑制
- **クラス予測の安定化** - 多数決で同一IDのクラスを固定
- **ヒステリシス** - 出現/消滅に閾値を設けてチラつき防止
- **軌跡描画** - 過去1秒間の移動経路を可視化
- **速度表示** - 移動速度をpx/secで表示
- **可変フレームレート** - ターゲット30Hz、動的に調整可能

## 基本的な使用方法

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/tidbots_6class/weights/best.pt \
  --camera 0 \
  --conf 0.5
```

## コマンドライン引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--model` | `outputs/trained_models/ycb_yolo26_run/weights/best.pt` | モデルのパス |
| `--camera` | `0` | カメラデバイスID |
| `--conf` | `0.5` | 信頼度閾値 |
| `--iou` | `0.45` | NMSのIoU閾値 |
| `--imgsz` | `640` | 入力画像サイズ |
| `--device` | `0` | GPUデバイス |
| `--save-video` | `False` | 動画を保存する場合に指定 |
| `--output` | `outputs/realtime_detection.mp4` | 出力動画のパス |
| `--target-fps` | `30.0` | 目標フレームレート (Hz) |

## キー操作

| キー | 機能 |
|------|------|
| `q` | 終了 |
| `s` | スクリーンショット保存 |
| `c` | 信頼度表示のON/OFF |
| `p` | 座標表示のON/OFF |
| `t` | 軌跡表示のON/OFF |
| `v` | 速度表示のON/OFF |
| `+` / `=` | 目標FPSを5上げる（最大60） |
| `-` | 目標FPSを5下げる（最小5） |

## 安定化機能

### 1. 信頼度の平滑化（Confidence Smoothing）

指数移動平均（EMA）を使用してスコアの急激な変動を抑制します。

```
smoothed_conf = α × current_conf + (1 - α) × previous_smoothed_conf
α = 0.3（デフォルト）
```

### 2. クラス予測の安定化（Class Stabilization）

同一トラックIDに対して、最初の5フレームの多数決でクラスを固定します。
これにより、類似オブジェクト間での誤分類のチラつきを防止します。

### 3. ヒステリシス（Hysteresis）

- **出現閾値**: 3フレーム連続検出で表示開始
- **消滅閾値**: 5フレーム連続消失で表示終了

これにより、一瞬だけの誤検出や検出漏れによるチラつきを防止します。

### 4. 可変フレームレート

- デフォルト30Hz
- `+`/`-`キーで動的に調整可能（5-60Hz）
- GPU負荷に応じて適応

## 画面表示

```
+------------------------------------------+
| FPS: 28.5/30 | Visible: 2                |
|                                          |
|   +---------------+                      |
|   | cracker_box   |                      |
|   |    0.85       |  ← 平滑化された信頼度 |
|   +----~~~~-------+  ← 軌跡（黄色）      |
|   ID:1 (230,250)     ← 平滑化座標（シアン）|
|   45 px/s            ← 速度（緑）        |
|                                          |
|          +--------+                      |
|          | banana |  ← 固定されたクラス   |
|          |  0.72  |                      |
|          +--------+                      |
|          ID:2 (450,280)                  |
|          12 px/s                         |
+------------------------------------------+
```

### 表示要素の説明

| 要素 | 色 | 説明 |
|------|-----|------|
| バウンディングボックス | トラックID依存 | 各トラックに固有の色 |
| 軌跡 | 黄色 `(0,255,255)` | 過去1秒間の中心点の移動経路 |
| 座標 | シアン `(255,255,0)` | 移動平均で平滑化された中心座標 |
| 速度 | 緑 `(0,255,0)` | 1秒間の移動距離から算出したpx/sec |

## 安定化パラメータ

ソースコード内で以下のパラメータを調整できます：

```python
APPEAR_THRESHOLD = 3      # 出現に必要な連続フレーム数
DISAPPEAR_THRESHOLD = 5   # 消滅に必要な連続フレーム数
CONF_SMOOTHING_ALPHA = 0.3  # EMAのα値（小さいほど滑らか）
CLASS_VOTE_FRAMES = 5     # クラス決定に使用するフレーム数
```

## 使用例

### 動画を保存する場合

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/tidbots_6class/weights/best.pt \
  --camera 0 \
  --conf 0.5 \
  --save-video \
  --output outputs/detection_demo.mp4
```

### 高フレームレートで実行

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/tidbots_6class/weights/best.pt \
  --camera 0 \
  --target-fps 60 \
  --conf 0.5
```

### CPU実行

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/tidbots_6class/weights/best.pt \
  --camera 0 \
  --device cpu \
  --conf 0.5
```

## トラブルシューティング

### カメラが開けない

```
Error: Cannot open camera 0
```

- カメラが接続されているか確認: `ls /dev/video*`
- 他のアプリケーションがカメラを使用していないか確認
- カメラIDを変更: `--camera 1` など

### FPSがターゲットに達しない

- `--imgsz` を小さくする（例: 320, 416）
- `--conf` を上げて検出数を減らす
- GPU使用を確認: `--device 0`
- `--target-fps` を下げる

### トラッキングIDが頻繁に変わる

- `--conf` を下げて検出を安定させる
- `--iou` を調整
- ヒステリシス閾値を上げる（ソースコード内）

### オブジェクトの出現が遅い

`APPEAR_THRESHOLD` を小さくする（デフォルト: 3）

## 関連ドキュメント

- [パイプライン実行ガイド](pipeline.md)
- [トラブルシューティング](troubleshooting.md)
