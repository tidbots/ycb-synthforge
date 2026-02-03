# パイプライン実行ガイド

このドキュメントでは、データ生成から推論までのパイプライン実行手順を説明します。

## 1. 合成データ生成

### 基本実行

```bash
# バックグラウンドで生成（config.yamlのnum_images設定に従う）
docker compose run -d blenderproc

# コンテナIDを確認
docker ps | grep blenderproc

# 進捗確認（生成された画像数）
ls data/synthetic/coco/images/ | wc -l

# リアルタイムログ確認
docker logs -f <container_id>

# または最新ログのみ
docker logs --tail 30 <container_id>
```

### 設定

生成枚数は `scripts/blenderproc/config.yaml` の `scene.num_images` で設定（デフォルト: 30,000枚）。

詳細な設定については[設定ファイルガイド](configuration.md)を参照してください。

### 生成時間の目安

| 設定 | samples | 速度 | 30,000枚の所要時間 |
|-----|---------|------|-------------------|
| 高速 | 32 | ~45枚/分 | ~11時間 |
| 高品質 | 128 | ~10枚/分 | ~50時間 |

### 生成画像サンプル

![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000009.png)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000013.png)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000029.png)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000455.png)

## 2. COCO→YOLO形式変換

```bash
docker compose run --rm yolo26_train python \
  scripts/data_processing/coco_to_yolo.py \
  --coco_json data/synthetic/coco/annotations.json \
  --output_dir data/synthetic/yolo \
  --train_ratio 0.833 \
  --val_ratio 0.083 \
  --test_ratio 0.083
```

### データセット構成

| Split | 枚数 | 割合 | 用途 |
|-------|------|------|------|
| Train | 25,000 | 83.3% | モデル学習 |
| Val | 2,500 | 8.3% | ハイパーパラメータ調整 |
| Test | 2,500 | 8.3% | 最終評価 |

## 3. YOLO26学習

```bash
# YOLO26m (Medium) モデルで学習
docker compose run --rm \
  -v $(pwd)/yolo_dataset:/workspace/yolo_dataset:ro \
  yolo26_train python3 /workspace/scripts/training/train_yolo26.py \
  --data /workspace/yolo_dataset/dataset.yaml \
  --weights /workspace/weights/yolo26m.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --project /workspace/outputs/trained_models \
  --name ycb_yolo26_run \
  --device 0 \
  --workers 8
```

### 学習結果の例 (YOLO26m, 50エポック)

| メトリクス | 値 |
|-----------|-----|
| **mAP50** | 97.52% |
| **mAP50-95** | 95.30% |
| **Precision** | 97.32% |
| **Recall** | 94.43% |
| 学習時間 | 約59分 (RTX 4090) |

### 出力ファイル

学習済み重みは `outputs/trained_models/ycb_yolo26_run/weights/` に保存されます:
- `best.pt` - 最高精度のモデル（推論用に推奨）
- `last.pt` - 最終エポックのモデル

```
outputs/trained_models/ycb_yolo26_run/
├── weights/
│   ├── best.pt              # ベストモデル (mAP基準)
│   ├── last.pt              # 最終エポックモデル
│   └── epoch*.pt            # チェックポイント (10エポックごと)
├── args.yaml                # 学習パラメータ
├── results.csv              # エポックごとのメトリクス
├── labels.jpg               # ラベル分布
└── train_batch*.jpg         # 訓練バッチサンプル
```

## 4. 評価

```bash
docker compose run --rm yolo26_train python \
  scripts/training/evaluate.py \
  --weights outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --data yolo_dataset/dataset.yaml
```

## 5. 推論

### バッチ推論

```bash
# 検証画像に対して推論を実行
docker compose run --rm \
  -v $(pwd)/yolo_dataset:/workspace/yolo_dataset:ro \
  yolo26_inference python3 /workspace/scripts/evaluation/inference.py \
  --model /workspace/outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --source /workspace/yolo_dataset/images/val \
  --output /workspace/outputs/inference_results \
  --conf 0.5 \
  --device 0
```

### 出力ファイル

推論結果は `outputs/inference_results/predictions/` に保存されます:
- `*.jpg` - バウンディングボックス付きの認識結果画像
- `labels/` - YOLO形式のラベルファイル
- `results.json` - 全検出結果のJSON

### 認識結果のサンプル

![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/inference_sample1.jpg)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/inference_sample2.jpg)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/inference_sample3.jpg)

## 6. リアルタイム検出 (Webカメラ)

USB Webカメラを使用したリアルタイム物体検出:

```bash
# Docker経由で実行
./run_realtime_detection.sh

# オプション指定
./run_realtime_detection.sh --camera 0 --conf 0.5

# ホストで直接実行（要: pip install ultralytics opencv-python）
./run_realtime_detection_host.sh
```

### 操作方法

| キー | 動作 |
|------|------|
| `q` | 終了 |
| `s` | スクリーンショット保存 |
| `c` | 信頼度表示の切り替え |

## ユーティリティ

### メッシュ検証

YCBオブジェクトのメッシュ品質を自動検証:

```bash
docker compose run --rm mesh_validator
```

結果は `data/mesh_validation_results.json` に保存されます。

### サムネイル生成

全オブジェクトのgoogle_16k/tsdf形式を比較するサムネイルを生成:

```bash
docker compose run --rm thumbnail_generator
```

結果:
- `data/thumbnails/*.png` - 個別サムネイル
- `data/thumbnails/comparison_grid.png` - 比較グリッド

### 全形式サムネイル比較

全オブジェクトの4形式（clouds/google_16k/poisson/tsdf）を比較:

```bash
docker compose run --rm thumbnail_all_formats
```

結果:
- `data/thumbnails_all_formats/*.png` - 個別サムネイル
- `data/thumbnails_all_formats/comparison_grid_all.png` - 全形式比較グリッド

**注意**: clouds形式（点群）とpoisson形式はテクスチャをサポートしていないため、グレーで表示されます。

## 関連ドキュメント

- [設定ファイルガイド](configuration.md)
- [ドメインランダム化](domain-randomization.md)
- [追加学習](incremental-learning.md)
- [アンサンブル推論](ensemble-inference.md)
