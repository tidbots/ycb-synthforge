# リアルタイム物体検出ガイド

Webカメラを使用したリアルタイム物体検出とトラッキングの使用方法を説明します。

## 概要

`scripts/evaluation/realtime_detection.py` は以下の機能を提供します：

- **ByteTrackによるオブジェクトトラッキング** - 検出のチラつきを軽減
- **座標の移動平均表示** - 1秒間の履歴で平滑化
- **軌跡描画** - 過去1秒間の移動経路を可視化
- **速度表示** - 移動速度をpx/secで表示

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

## キー操作

| キー | 機能 |
|------|------|
| `q` | 終了 |
| `s` | スクリーンショット保存 |
| `c` | 信頼度表示のON/OFF |
| `p` | 座標表示のON/OFF |
| `t` | 軌跡表示のON/OFF |
| `v` | 速度表示のON/OFF |

## 画面表示

```
+------------------------------------------+
| FPS: 30.0 | Detections: 2                |
|                                          |
|   +---------------+                      |
|   | cracker_box   |                      |
|   |    0.85       |                      |
|   +----~~~~-------+  ← 軌跡（黄色）      |
|   ID:1 (230,250)     ← 平滑化座標（シアン）|
|   45 px/s            ← 速度（緑）        |
|                                          |
|          +--------+                      |
|          | banana |                      |
|          |  0.72  |                      |
|          +--------+                      |
|          ID:2 (450,280)                  |
|          12 px/s                         |
+------------------------------------------+
```

### 表示要素の説明

| 要素 | 色 | 説明 |
|------|-----|------|
| バウンディングボックス | クラス依存 | YOLOのデフォルト描画 |
| 軌跡 | 黄色 `(0,255,255)` | 過去1秒間の中心点の移動経路 |
| 座標 | シアン `(255,255,0)` | 移動平均で平滑化された中心座標 |
| 速度 | 緑 `(0,255,0)` | 1秒間の移動距離から算出したpx/sec |

## トラッキング機能

### ByteTrack

本スクリプトはByteTrackアルゴリズムを使用してオブジェクトをトラッキングします。

- 同一オブジェクトに一貫したIDを割り当て
- 一時的な検出漏れにも対応
- `persist=True` でフレーム間の状態を維持

### 移動平均による平滑化

検出座標のチラつきを軽減するため、1秒間（約30フレーム）の履歴を保持し、移動平均を計算します。

```
平滑化座標 = mean(過去1秒間の中心座標)
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

### 高解像度で実行（処理が重くなる可能性あり）

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/tidbots_6class/weights/best.pt \
  --camera 0 \
  --imgsz 1280 \
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

### FPSが低い

- `--imgsz` を小さくする（例: 320, 416）
- `--conf` を上げて検出数を減らす
- GPU使用を確認: `--device 0`

### トラッキングIDが頻繁に変わる

- `--conf` を下げて検出を安定させる
- `--iou` を調整

## 関連ドキュメント

- [パイプライン実行ガイド](pipeline.md)
- [トラブルシューティング](troubleshooting.md)
