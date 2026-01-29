# YCB SynthForge

BlenderProcによる合成データ生成と転移学習を組み合わせた、YCBオブジェクト検出のた
めのYOLO26ファインチューニングパイプライン

## 概要

YCB SynthForgeは、以下の機能を提供する統合パイプラインです：

- **合成データ生成**: BlenderProcを使用したフォトリアリスティックな学習データの>自動生成
- **ドメインランダマイゼーション**: Sim-to-Realギャップを軽減する高度なランダム>化
- **転移学習**: COCO事前学習済みYOLO26モデルのファインチューニング
- **カスタムデータ対応**: 独自オブジェクトの追加が容易

## プロジェクト構造
```
ycb_synthforge/
├── docker/
│   ├── Dockerfile.blenderproc    # BlenderProc環境
│   └── Dockerfile.yolo26         # YOLO26学習環境
├── docker-compose.yml
├── models/
│   └── ycb/                      # YCB 3Dモデル (103クラス)
├── resources/
│   └── cctextures/               # CC0テクスチャ (2022枚)
├── weights/
│   ├── yolo26n.pt                # YOLO26 Nano (2.6M params)
│   └── yolo26s.pt                # YOLO26 Small
├── scripts/
│   ├── blenderproc/              # データ生成スクリプト
│   │   ├── generate_dataset.py   # メイン生成スクリプト
│   │   ├── config.yaml           # 生成設定
│   │   ├── scene_setup.py        # シーン構築
│   │   ├── lighting.py           # 照明ランダム化
│   │   ├── camera.py             # カメラ効果
│   │   ├── materials.py          # マテリアルランダム化
│   │   └── ycb_classes.py        # 103クラス定義
│   ├── data_processing/
│   │   ├── coco_to_yolo.py       # COCO→YOLO変換 (train/val/test分割)
│   │   └── merge_datasets.py     # データセット結合
│   └── training/
│       ├── train_yolo26.py       # 学習スクリプト
│       ├── train_config.yaml     # 学習設定
│       ├── evaluate.py           # 評価スクリプト
│       └── inference.py          # 推論スクリプト
├── data/
│   └── synthetic/
│       ├── coco/                 # 生成データ (COCO形式)
│       │   ├── images/           # レンダリング画像
│       │   └── annotations.json  # アノテーション
│       └── yolo/                 # 変換データ (YOLO形式)
│           ├── images/{train,val,test}/
│           ├── labels/{train,val,test}/
│           └── dataset.yaml
├── runs/                         # 学習結果
│   └── */weights/{best,last}.pt
└── logs/                         # ログファイル
```

## セットアップ

### 必要環境

- Docker & Docker Compose v2+
- NVIDIA GPU (CUDA対応)
- NVIDIA Container Toolkit
- 推奨: RTX 3090/4090 (VRAM 24GB)

### Dockerイメージのビルド

```bash
# 全イメージをビルド
docker compose build

# 個別ビルド
docker compose build blenderproc
docker compose build yolo26_train
```

## パイプライン実行

### 1. 合成データ生成

```bash
# バックグラウンドで12,000枚生成
docker compose run -d --name ycb_generation blenderproc \
  blenderproc run /workspace/scripts/blenderproc/generate_dataset.py \
  --config /workspace/scripts/blenderproc/config.yaml \
  --output /workspace/data/synthetic/coco \
  --num_scenes 12000

# 進捗確認
watch -n 30 'ls data/synthetic/coco/images/ | wc -l'

# ログ確認
docker logs ycb_generation --tail 20
```

### 2. COCO→YOLO形式変換

```bash
docker compose run --rm yolo26_train python \
  /workspace/scripts/data_processing/coco_to_yolo.py \
  --coco_json /workspace/data/synthetic/coco/annotations.json \
  --output_dir /workspace/data/synthetic/yolo \
  --train_ratio 0.833 \
  --val_ratio 0.083 \
  --test_ratio 0.083
```

### 3. YOLO26学習

```bash
docker compose run --rm yolo26_train python \
  /workspace/scripts/training/train_yolo26.py \
  --config /workspace/scripts/training/train_config.yaml
```

### 4. 評価

```bash
docker compose run --rm yolo26_train python \
  /workspace/scripts/training/evaluate.py \
  --weights /workspace/runs/ycb_yolo26/weights/best.pt \
  --data /workspace/data/synthetic/yolo/dataset.yaml
```

### 5. 推論

```bash
docker compose run --rm yolo26_inference python \
  /workspace/scripts/training/inference.py \
  --weights /workspace/runs/ycb_yolo26/weights/best.pt \
  --source /path/to/images
```

## ドメインランダム化

Sim-to-Realギャップを軽減するため、以下の要素をランダム化:

| カテゴリ | ランダム化項目 | 範囲 |
|---------|---------------|------|
| **背景** | 床テクスチャ | Wood, Concrete, Tiles, Marble, Metal, Fabric |
| | 壁テクスチャ | Concrete, Plaster, Brick, Paint, Wallpaper |
| | テーブル材質 | Wood, Metal, Plastic |
| **照明** | 光源数 | 1-4個 |
| | 色温度 | 2700K-6500K |
| | 強度 | 100-1000W相当 |
| | 影の柔らかさ | 0.3-0.9 |
| **カメラ** | 距離 | 0.4-2.0m |
| | 仰角 | 10-70° |
| | 方位角 | 0-360° |
| | 露出 | EV -1.5〜+1.5 |
| | ISO | 100-3200 |
| | 被写界深度 | f/1.8-11.0 |
| **マテリアル** | 金属度 | 0.8-1.0 (金属物体) |
| | 粗さ | 0.05-0.6 |
| | 色相シフト | ±10° |
| **オブジェクト** | 位置 | X,Y: ±0.3m |
| | 回転 | 0-360° (各軸) |
| | スケール | ±5% |

## データセット構成

| Split | 枚数 | 割合 | 用途 |
|-------|------|------|------|
| Train | 10,000 | 83.3% | モデル学習 |
| Val | 1,000 | 8.3% | ハイパーパラメータ調整 |
| Test | 1,000 | 8.3% | 最終評価 |

## YCBオブジェクトクラス (103種)

<details>
<summary>クラス一覧を表示</summary>

### 食品・飲料 (ID: 0-9)
| ID | 名前 | ID | 名前 |
|----|------|----|------|
| 0 | 001_chips_can | 5 | 006_mustard_bottle |
| 1 | 002_master_chef_can | 6 | 007_tuna_fish_can |
| 2 | 003_cracker_box | 7 | 008_pudding_box |
| 3 | 004_sugar_box | 8 | 009_gelatin_box |
| 4 | 005_tomato_soup_can | 9 | 010_potted_meat_can |

### 果物 (ID: 10-17)
| ID | 名前 | ID | 名前 |
|----|------|----|------|
| 10 | 011_banana | 14 | 015_peach |
| 11 | 012_strawberry | 15 | 016_pear |
| 12 | 013_apple | 16 | 017_orange |
| 13 | 014_lemon | 17 | 018_plum |

### キッチン用品 (ID: 18-31)
019_pitcher_base, 021_bleach_cleanser, 022_windex_bottle, 023_wine_glass, 024_bowl, 025_mug, 026_sponge, 027-skillet, 028_skillet_lid, 029_plate, 030_fork, 031_spoon, 032_knife, 033_spatula

### 工具 (ID: 32-48)
035_power_drill, 036_wood_block, 037_scissors, 038_padlock, 039_key, 040_large_marker, 041_small_marker, 042_adjustable_wrench, 043_phillips_screwdriver, 044_flat_screwdriver, 046_plastic_bolt, 047_plastic_nut, 048_hammer, 049-052_clamps

### スポーツ・おもちゃ (ID: 49-102)
ボール類、チェーン、フォームブロック、サイコロ、ビー玉、カップ、木製ブロック、お
もちゃの飛行機、レゴデュプロ、タイマー、ルービックキューブ

</details>

## 設定ファイル

### データ生成設定 (`scripts/blenderproc/config.yaml`)

```yaml
scene:
  num_images: 12000
  objects_per_scene: [2, 8]    # シーンあたりのYCBオブジェクト数

rendering:
  engine: "CYCLES"
  samples: 32                   # 32=高速(推奨), 128=高品質
  use_denoising: true
  use_gpu: true

camera:
  distance: [0.4, 2.0]
  elevation: [10, 70]
  azimuth: [0, 360]
```

### 学習設定 (`scripts/training/train_config.yaml`)

```yaml
model:
  architecture: yolo26n         # nano / small / medium
  weights: /workspace/weights/yolo26n.pt
  num_classes: 103

training:
  epochs: 100
  batch_size: 16
  imgsz: 640
  optimizer: auto
  lr0: 0.01
  patience: 20

augmentation:
  mosaic: 1.0
  mixup: 0.1
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
```

## 出力ファイル

### 学習結果 (`runs/`)

```
runs/ycb_yolo26/
├── weights/
│   ├── best.pt              # ベストモデル (mAP基準)
│   └── last.pt              # 最終エポックモデル
├── results.csv              # エポックごとのメトリクス
├── results.png              # 学習曲線グラフ
├── confusion_matrix.png     # 混同行列
├── labels.jpg               # ラベル分布
├── train_batch*.jpg         # 訓練バッチサンプル
└── val_batch*_pred.jpg      # 検証予測結果
```

## 生成時間の目安

| 設定 | samples | 速度 | 12,000枚の所要時間 |
|-----|---------|------|-------------------|
| 高速 | 32 | ~45枚/分 | ~4.5時間 |
| 高品質 | 128 | ~10枚/分 | ~20時間 |

## トラブルシューティング

### GPUが認識されない

```bash
# ホストでNVIDIA確認
nvidia-smi

# コンテナ内で確認
docker compose run --rm blenderproc nvidia-smi
```

### メモリ不足エラー

`docker-compose.yml`で`shm_size`を増加:

```yaml
services:
  blenderproc:
    shm_size: '16gb'
  yolo26_train:
    shm_size: '16gb'
```

### NumPy互換性警告

`Dockerfile.yolo26`で`numpy<2`を指定済み。警告が出る場合は再ビルド:

```bash
docker compose build yolo26_train --no-cache
```

### OBJファイルの警告

`Invalid normal index`警告は無害。YCBモデルのメッシュ問題で、レンダリングに影響>なし。

## ライセンス

- YCBモデル: [YCB Object and Model Set License](https://www.ycbbenchmarks.com/)
- CC0テクスチャ: [CC0 1.0 Universal](https://ambientcg.com/)
- BlenderProc: MIT License
- Ultralytics YOLO: AGPL-3.0

## 参考文献

- [BlenderProc](https://github.com/DLR-RM/BlenderProc)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YCB Object Dataset](https://www.ycbbenchmarks.com/)
- [ambientCG Textures](https://ambientcg.com/)


