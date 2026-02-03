# 設定ファイルガイド

このドキュメントでは、YCB SynthForgeの設定ファイルについて説明します。

## データ生成設定

`scripts/blenderproc/config.yaml`

```yaml
# モデルソース設定（複数ソース対応）
model_sources:
  ycb:
    path: "/workspace/models/ycb"
    include: []                   # 空=全オブジェクト使用
    # include:                    # 特定オブジェクトのみ使用する場合
    #   - "002_master_chef_can"
    #   - "005_tomato_soup_can"

  tidbots:                        # カスタムモデルソース
    path: "/workspace/models/tidbots"
    include: []

scene:
  num_images: 30000               # 生成画像数
  objects_per_scene: [2, 8]       # シーンあたりのオブジェクト数 [最小, 最大]

rendering:
  engine: "CYCLES"                # レンダリングエンジン
  samples: 32                     # サンプル数 (32=高速, 128=高品質)
  use_denoising: true             # デノイジング有効
  use_gpu: true                   # GPU使用

camera:
  distance: [0.4, 0.9]            # カメラ距離 [最小, 最大]
  elevation: [35, 65]             # 仰角 [最小, 最大]
  azimuth: [0, 360]               # 方位角 [最小, 最大]

lighting:
  num_lights: [3, 5]              # 光源数 [最小, 最大]
  intensity: [800, 2000]          # 光強度 [最小, 最大]
  ambient: [0.4, 0.7]             # 環境光 [最小, 最大]

placement:
  position:
    x_range: [-0.25, 0.25]        # X方向配置範囲
    y_range: [-0.25, 0.25]        # Y方向配置範囲
  use_physics: false              # 物理シミュレーション (false=グリッド配置)
```

### パラメータ詳細

| パラメータ | 説明 | 推奨値 |
|-----------|------|-------|
| `scene.num_images` | 生成する画像の総数 | 30,000 |
| `scene.objects_per_scene` | 1シーンに配置するオブジェクト数の範囲 | [2, 8] |
| `rendering.samples` | レイトレーシングのサンプル数。高いほど高品質だが遅い | 32 (高速) / 128 (高品質) |
| `camera.distance` | カメラからオブジェクトまでの距離（メートル） | [0.4, 0.9] |
| `camera.elevation` | カメラの仰角（度） | [35, 65] |
| `lighting.num_lights` | シーン内の光源数 | [3, 5] |
| `placement.use_physics` | 物理シミュレーションでオブジェクトを落下配置するか | false |

## 学習設定

`scripts/training/train_config.yaml`

```yaml
model:
  architecture: yolo26n           # モデルアーキテクチャ (nano/small/medium)
  weights: /workspace/weights/yolo26n.pt  # 事前学習重み
  num_classes: 85                 # クラス数

training:
  epochs: 100                     # エポック数
  batch_size: 16                  # バッチサイズ
  imgsz: 640                      # 入力画像サイズ
  optimizer: auto                 # オプティマイザ
  lr0: 0.01                       # 初期学習率
  patience: 20                    # 早期終了の待機エポック数

augmentation:
  mosaic: 1.0                     # モザイク増強の確率
  mixup: 0.1                      # MixUp増強の確率
  hsv_h: 0.015                    # 色相変動
  hsv_s: 0.7                      # 彩度変動
  hsv_v: 0.4                      # 明度変動
```

### パラメータ詳細

| パラメータ | 説明 | 推奨値 |
|-----------|------|-------|
| `model.architecture` | YOLOモデルのサイズ | yolo26n (高速) / yolo26m (バランス) |
| `training.epochs` | 学習のエポック数 | 50-100 |
| `training.batch_size` | バッチサイズ（VRAM依存） | 16 (24GB) / 8 (12GB) |
| `training.imgsz` | 入力画像サイズ | 640 |
| `training.patience` | 改善がない場合に停止するまでのエポック数 | 20 |
| `augmentation.mosaic` | モザイク増強の使用確率 | 1.0 |

## 出力ディレクトリ構造

### 学習結果

```
outputs/trained_models/ycb_yolo26_run/
├── weights/
│   ├── best.pt              # ベストモデル (mAP基準)
│   ├── last.pt              # 最終エポックモデル
│   └── epoch*.pt            # チェックポイント
├── args.yaml                # 学習パラメータ
├── results.csv              # エポックごとのメトリクス
├── labels.jpg               # ラベル分布
└── train_batch*.jpg         # 訓練バッチサンプル
```

### 推論結果

```
outputs/inference_results/predictions/
├── *.jpg                    # バウンディングボックス付き認識結果画像
├── labels/                  # YOLO形式のラベルファイル
└── results.json             # 全検出結果 (JSON)
```

## Docker Compose設定

`docker-compose.yml`

主要なサービス設定:

```yaml
services:
  blenderproc:
    shm_size: '8gb'              # 共有メモリサイズ
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  yolo26_train:
    shm_size: '16gb'             # 学習には大きめの共有メモリが必要
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### メモリ設定

メモリ不足エラーが発生する場合は `shm_size` を増加:

```yaml
services:
  blenderproc:
    shm_size: '16gb'
  yolo26_train:
    shm_size: '16gb'
```

## 関連ドキュメント

- [パイプライン実行](pipeline.md)
- [ドメインランダム化](domain-randomization.md)
- [トラブルシューティング](troubleshooting.md)
