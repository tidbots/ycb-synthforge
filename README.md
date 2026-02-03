# YCB SynthForge

BlenderProcによる合成データ生成とYOLO26によるYCB物体検出パイプライン

## 概要

YCB SynthForgeは、85種類のYCBオブジェクトを検出するためのEnd-to-Endパイプラインです。

- **合成データ生成**: BlenderProcによるフォトリアリスティックなレンダリング
- **ドメインランダム化**: Sim-to-Real転移のための多様なデータ生成
- **YOLO26学習**: COCO事前学習モデルのファインチューニング
- **google_16k + tsdf形式**: オブジェクトごとに最適な形式を自動選択

## クイックスタート

### 1. 環境構築

```bash
# Dockerイメージをビルド
docker compose build

# YOLO26重みをダウンロード
python scripts/download_weights.py

# YCB 3Dモデルをダウンロード
python scripts/download_ycb_models.py --all --format google_16k
python scripts/download_ycb_models.py --all --format berkeley

# tsdf形式のマテリアルを修正
docker compose run --rm fix_tsdf_materials
```

### 2. 合成データ生成

```bash
docker compose run -d blenderproc
```

### 3. YOLO26学習

```bash
# データ形式を変換
docker compose run --rm yolo26_train python \
  scripts/data_processing/coco_to_yolo.py \
  --coco_json data/synthetic/coco/annotations.json \
  --output_dir data/synthetic/yolo

# 学習を実行
docker compose run --rm yolo26_train python3 \
  /workspace/scripts/training/train_yolo26.py \
  --data /workspace/yolo_dataset/dataset.yaml \
  --weights /workspace/weights/yolo26m.pt \
  --epochs 50
```

### 4. 推論

```bash
docker compose run --rm yolo26_inference python3 \
  /workspace/scripts/evaluation/inference.py \
  --model /workspace/outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --source /workspace/yolo_dataset/images/val
```

## 生成画像サンプル

![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000009.png)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000013.png)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000029.png)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/scene_000455.png)

## 認識結果のサンプル

![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/inference_sample1.jpg)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/inference_sample2.jpg)
![](https://github.com/tidbots/ycb_synthforge/blob/main/fig/inference_sample3.jpg)

## プロジェクト構成

```
ycb_synthforge/
├── docker/                       # Dockerファイル
├── docker-compose.yml
├── models/
│   ├── ycb/                      # YCB 3Dモデル (85クラス)
│   └── tidbots/                  # カスタム3Dモデル
├── resources/cctextures/         # CC0テクスチャ
├── weights/                      # YOLO26重み
├── scripts/
│   ├── blenderproc/              # データ生成スクリプト
│   ├── data_processing/          # データ変換
│   ├── training/                 # 学習スクリプト
│   └── inference/                # 推論スクリプト
├── data/synthetic/               # 生成データ
├── outputs/                      # 出力結果
└── docs/                         # 詳細ドキュメント
```

## ドキュメント

| ドキュメント | 説明 |
|-------------|------|
| [セットアップガイド](docs/setup.md) | 環境構築、重み・モデル・テクスチャのダウンロード |
| [パイプライン実行](docs/pipeline.md) | データ生成、学習、評価、推論の手順 |
| [設定ファイル](docs/configuration.md) | config.yamlの詳細設定 |
| [ドメインランダム化](docs/domain-randomization.md) | Sim-to-Real転移のためのランダム化 |
| [カスタムモデル](docs/custom-models.md) | 独自3Dモデルの追加方法 |
| [YCBクラス一覧](docs/ycb-classes.md) | 85種類のYCBオブジェクト |
| [追加学習](docs/incremental-learning.md) | 新しいオブジェクトの追加 |
| [アンサンブル推論](docs/ensemble-inference.md) | 複数モデルの組み合わせ |
| [トラブルシューティング](docs/troubleshooting.md) | 問題解決 |

## 必要環境

- Docker & Docker Compose v2+
- NVIDIA GPU (CUDA対応)
- NVIDIA Container Toolkit
- 推奨: RTX 3090/4090 (VRAM 24GB)

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
