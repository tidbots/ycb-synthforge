# 新規オブジェクト専用学習ガイド

YCBデータセットを使用せず、独自のカスタムオブジェクトのみでYOLO26モデルを学習する方法を説明します。

## 概要

このガイドでは、`models/tidbots/` などのカスタムモデルディレクトリにある独自の3Dモデルのみを使用して、物体検出モデルを学習する手順を説明します。

## 前提条件

- Docker および Docker Compose がインストールされていること
- 3Dスキャンされたオブジェクトモデル（OBJ形式）
- 十分なディスク容量（合成データ用に約10GB推奨）

## Step 1: カスタムモデルの準備

### ディレクトリ構造

カスタムモデルは以下の構造で配置します：

```
models/
└── tidbots/              # カスタムモデルソース名
    ├── object_name_1/    # オブジェクト名（=クラス名）
    │   ├── model.obj     # 3Dモデル（任意のファイル名可）
    │   ├── materials.mtl # マテリアルファイル
    │   └── texture.jpg   # テクスチャ画像
    ├── object_name_2/
    │   └── ...
    └── ...
```

### モデル要件

- **形式**: OBJ形式（`.obj`）
- **テクスチャ**: JPG または PNG 形式
- **スケール**: 任意（自動正規化されます）
- **命名**: フォルダ名がクラス名として使用されます

> **注意**: モデルがミリメートル単位で作成されている場合でも、自動的にメートル単位（約15cm）に正規化されます。

## Step 2: 設定ファイルの編集

`scripts/blenderproc/config.yaml` を編集して、カスタムモデルソースのみを有効にします。

```yaml
# Model sources configuration
model_sources:
  # YCB を無効化（コメントアウト）
  # ycb:
  #   path: "/workspace/models/ycb"
  #   include: []

  # カスタムモデルソースを有効化
  tidbots:
    path: "/workspace/models/tidbots"
    include: []  # 空リスト = 全オブジェクトを使用
```

### 特定のオブジェクトのみを使用する場合

```yaml
  tidbots:
    path: "/workspace/models/tidbots"
    include:
      - "aquarius"
      - "chipstar"
      - "coffee_1"
```

### 設定のポイント

| 設定項目 | 説明 |
|---------|------|
| `path` | Dockerコンテナ内のパス（`/workspace/models/...`） |
| `include: []` | 空リスト = ディレクトリ内の全オブジェクトを使用 |
| `include: [...]` | 指定したオブジェクトのみを使用 |

## Step 3: 合成データの生成

BlenderProcを使用して合成画像を生成します。

```bash
docker compose run --rm blenderproc blenderproc run \
  /workspace/scripts/blenderproc/generate_dataset.py \
  --num_scenes 2000
```

### 生成パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--num_scenes` | 2000 | 生成する画像数 |
| `--output` | `/workspace/data/synthetic/coco` | 出力ディレクトリ |
| `--seed` | 123 | 乱数シード（再現性のため） |

### 出力

```
data/synthetic/coco/
├── images/
│   ├── scene_000000.png
│   ├── scene_000001.png
│   └── ...
└── annotations.json    # COCO形式のアノテーション
```

## Step 4: COCO → YOLO形式変換

生成されたCOCO形式のデータをYOLO形式に変換します。

```bash
docker compose run --rm yolo26_train python3 \
  /workspace/scripts/data_processing/coco_to_yolo.py \
  --coco_json /workspace/data/synthetic/coco/annotations.json \
  --output_dir /workspace/data/synthetic/yolo
```

### 出力

```
data/synthetic/yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml    # YOLO学習用設定ファイル
```

## Step 5: YOLO26モデルの学習

```bash
docker compose run --rm yolo26_train python3 \
  /workspace/scripts/training/train_yolo26.py \
  --data /workspace/data/synthetic/yolo/dataset.yaml \
  --weights /workspace/weights/yolo26m.pt \
  --epochs 50 \
  --name custom_objects
```

### 学習パラメータ

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `--epochs` | 50-100 | エポック数 |
| `--batch` | 16 | バッチサイズ（GPU メモリに応じて調整） |
| `--imgsz` | 640 | 入力画像サイズ |
| `--weights` | `yolo26m.pt` | 事前学習済み重み |

### 出力

```
outputs/trained_models/custom_objects/
├── weights/
│   ├── best.pt     # 最良モデル
│   └── last.pt     # 最終モデル
└── results.csv     # 学習メトリクス
```

## Step 6: 推論テスト

学習済みモデルで推論を実行します。

```bash
docker compose run --rm yolo26_inference python3 \
  /workspace/scripts/evaluation/inference.py \
  --model /workspace/outputs/trained_models/custom_objects/weights/best.pt \
  --source /path/to/test/images
```

## トラブルシューティング

### 画像が真っ白になる

**原因**: モデルのスケールが大きすぎる（ミリメートル単位など）

**解決策**: `generate_dataset.py` の `normalize_object_scale()` 関数が自動的にスケールを正規化します。この機能が正しく動作していることを確認してください。

### テクスチャが表示されない

**原因**: MTLファイルのテクスチャパスが間違っている

**解決策**:
1. MTLファイル内のテクスチャパスが相対パスであることを確認
2. テクスチャファイル名がMTLファイルの記述と一致していることを確認

```
# materials.mtl の例
newmtl material0
map_Kd texture.jpg
```

### アノテーションが生成されない

**原因**: オブジェクトがカメラの視野外にある

**解決策**: `config.yaml` のカメラ設定を調整
```yaml
camera:
  distance: [0.3, 0.8]  # より近くに
  elevation: [30, 60]   # 適切な角度
```

## クラスID の割り当て

カスタムモデルのみを使用する場合、クラスIDは自動的に0から順番に割り当てられます：

| フォルダ名 | クラスID |
|-----------|---------|
| aquarius | 0 |
| chipstar | 1 |
| coffee_1 | 2 |
| coffee_2 | 3 |
| cupnoodle_seafood | 4 |
| redbull | 5 |

> **注意**: クラスIDはフォルダ名のアルファベット順で割り当てられます。

## 関連ドキュメント

- [カスタムモデルの追加](custom-models.md)
- [設定ファイルリファレンス](configuration.md)
- [ドメインランダマイゼーション](domain-randomization.md)
- [トラブルシューティング](troubleshooting.md)
