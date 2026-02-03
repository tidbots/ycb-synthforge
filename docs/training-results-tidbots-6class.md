# tidbots 6クラス学習結果レポート

**実行日:** 2026-02-03
**モデル:** YOLO26m
**データセット:** tidbots 6クラス合成データ

## 対象クラス

| クラスID | クラス名 | 説明 |
|---------|---------|------|
| 0 | aquarius | アクエリアス（飲料） |
| 1 | chipstar | チップスター（菓子） |
| 2 | coffee_1 | コーヒー1（ジョージア） |
| 3 | coffee_2 | コーヒー2 |
| 4 | cupnoodle_seafood | カップヌードルシーフード |
| 5 | redbull | レッドブル（飲料） |

## データセット構成

| 項目 | 数値 |
|------|------|
| 総画像数 | 2,000枚 |
| アノテーション数 | 3,710個 |
| 訓練データ | 1,667枚 |
| 検証データ | 166枚 |
| テストデータ | 167枚 |
| 画像解像度 | 640x640 |

## 学習設定

| パラメータ | 値 |
|-----------|-----|
| ベースモデル | yolo26m.pt |
| エポック数 | 50 |
| バッチサイズ | 16 |
| 画像サイズ | 640 |
| オプティマイザ | AdamW |
| 学習率 | 0.001 |
| GPU | NVIDIA GPU (11.5GB使用) |

## 学習結果

### 最終メトリクス

| メトリクス | 値 |
|-----------|-----|
| **mAP50** | **99.16%** |
| **mAP50-95** | **96.48%** |
| Box Loss | 0.23 |
| Class Loss | 0.08 |
| DFL Loss | 0.003 |

### 学習時間

- **総学習時間:** 約15分 (0.254時間)
- **1エポックあたり:** 約18秒

## 推論テスト結果

| 項目 | 値 |
|------|-----|
| テスト画像数 | 167枚 |
| 総検出数 | 307件 |
| 平均推論時間 | 5-7ms/画像 |
| 信頼度閾値 | 0.5 |

### クラス別検出信頼度（サンプル）

| クラス | 信頼度 |
|--------|--------|
| coffee_1 | 98-99% |
| aquarius | 97% |
| cupnoodle_seafood | 92-97% |
| redbull | 96% |
| chipstar | 95%+ |
| coffee_2 | 95%+ |

## 出力ファイル

```
runs/detect/outputs/trained_models/tidbots_6class/
├── weights/
│   ├── best.pt          # 最良モデル (44MB)
│   ├── last.pt          # 最終モデル (44MB)
│   ├── epoch0.pt        # チェックポイント
│   ├── epoch10.pt
│   ├── epoch20.pt
│   ├── epoch30.pt
│   └── epoch40.pt
├── labels.jpg           # ラベル分布
├── results.csv          # 学習メトリクス
└── args.yaml            # 学習設定
```

## 推論コマンド

```bash
docker compose run --rm yolo26_train python3 /workspace/scripts/evaluation/inference.py \
  --model /workspace/runs/detect/outputs/trained_models/tidbots_6class/weights/best.pt \
  --source /path/to/images \
  --output /workspace/outputs/inference_results
```

## 設定ファイル

### config.yaml (抜粋)

```yaml
model_sources:
  tidbots:
    path: "/workspace/models/tidbots"
    include: []  # 6 classes: aquarius, chipstar, coffee_1, coffee_2, cupnoodle_seafood, redbull

scene:
  num_images: 2000
  objects_per_scene: [1, 3]
```

## 技術的な注意点

### スケール正規化

tidbots モデルはミリメートル単位で作成されているため、`generate_dataset.py` の `normalize_object_scale()` 関数で自動的にメートル単位（約15cm）に正規化されます。

```python
def normalize_object_scale(obj, target_size=0.15):
    """モデルのスケールを正規化"""
    bbox = obj.get_bound_box()
    max_dim = max(dimensions)
    scale_factor = target_size / max_dim
    obj.set_scale([s * scale_factor for s in current_scale])
```

### GPU設定

docker-compose.yml に以下のGPU設定が必要：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 再現手順

1. **データ生成**
   ```bash
   docker compose run --rm blenderproc blenderproc run \
     /workspace/scripts/blenderproc/generate_dataset.py --num_scenes 2000
   ```

2. **COCO→YOLO変換**
   ```bash
   docker compose run --rm yolo26_train python3 \
     /workspace/scripts/data_processing/coco_to_yolo.py \
     --coco_json /workspace/data/synthetic/coco/annotations.json \
     --output_dir /workspace/data/synthetic/yolo
   ```

3. **学習**
   ```bash
   docker compose run --rm yolo26_train python3 \
     /workspace/scripts/training/train_yolo26.py \
     --data /workspace/data/synthetic/yolo/dataset.yaml \
     --weights /workspace/weights/yolo26m.pt \
     --epochs 50 \
     --name tidbots_6class
   ```

4. **推論**
   ```bash
   docker compose run --rm yolo26_train python3 \
     /workspace/scripts/evaluation/inference.py \
     --model /workspace/runs/detect/outputs/trained_models/tidbots_6class/weights/best.pt \
     --source /path/to/test/images \
     --output /workspace/outputs/inference_results
   ```

## 関連ドキュメント

- [新規オブジェクト専用学習ガイド](custom-only-training.md)
- [カスタムモデルの追加](custom-models.md)
- [設定ファイルリファレンス](configuration.md)
