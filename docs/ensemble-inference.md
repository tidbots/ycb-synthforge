# アンサンブル推論

このドキュメントでは、複数のモデルを組み合わせて推論する方法を説明します。

## 概要

アンサンブル推論は、複数の学習済みモデルを組み合わせて推論する手法です。モデルを追加・削除する際に再学習が不要です。

## 追加学習との比較

| 項目 | 追加学習 | アンサンブル推論 |
|------|---------|-----------------|
| 推論速度 | 速い（1モデル） | 遅い（N回推論） |
| メモリ | 少ない | 多い（N倍） |
| 精度維持 | 忘却リスクあり | 各モデル維持 |
| 柔軟性 | 再学習が必要 | モデル追加/削除が容易 |

## 画像推論

```bash
docker compose run --rm ensemble_inference python \
  scripts/inference/ensemble_inference.py \
  --models weights/yolo26n.pt outputs/ycb_best.pt \
  --source data/test_images/ \
  --output outputs/ensemble_results \
  --show-model
```

### パラメータ

| パラメータ | 説明 |
|-----------|------|
| `--models` | 使用するモデルのパス（複数指定可） |
| `--source` | 入力画像またはディレクトリ |
| `--output` | 出力先ディレクトリ |
| `--show-model` | 検出結果にモデル名を表示 |
| `--conf` | 信頼度閾値（デフォルト: 0.3） |
| `--iou` | NMS用IoU閾値（デフォルト: 0.5） |

## リアルタイム推論（Webカメラ）

```bash
docker compose run --rm ensemble_inference python \
  scripts/inference/ensemble_inference.py \
  --models weights/yolo26n.pt outputs/ycb_best.pt \
  --source 0 \
  --realtime
```

## Pythonコードでの使用

```python
from ensemble_inference import EnsembleDetector

# 複数モデルを初期化
detector = EnsembleDetector(
    model_paths=[
        'yolo26n.pt',      # COCO 80クラス (ID: 0-79)
        'ycb_best.pt',     # YCB 85クラス  (ID: 80-164)
        'custom.pt',       # カスタム     (ID: 165+)
    ],
    conf_threshold=0.3,
    iou_threshold=0.5,
)

# 推論
detections = detector.predict(image)

# 結果を描画
result = detector.draw_detections(image, detections, show_model=True)
```

### EnsembleDetector API

```python
class EnsembleDetector:
    def __init__(
        self,
        model_paths: List[str],    # モデルファイルのパス
        conf_threshold: float,      # 信頼度閾値
        iou_threshold: float,       # NMS用IoU閾値
        device: str = 'cuda',       # デバイス ('cuda' or 'cpu')
    )

    def predict(
        self,
        image: np.ndarray,          # 入力画像 (BGR)
    ) -> List[Detection]:           # 検出結果リスト

    def draw_detections(
        self,
        image: np.ndarray,          # 入力画像
        detections: List[Detection], # 検出結果
        show_model: bool = False,   # モデル名を表示するか
    ) -> np.ndarray:                # 描画済み画像
```

### Detection形式

```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float                 # 信頼度
    class_id: int                     # グローバルクラスID
    class_name: str                   # クラス名
    model_name: str                   # 検出元モデル名
```

## クラスIDの管理

アンサンブルでは、各モデルのクラスIDがグローバルIDにマッピングされます:

| モデル | ローカルID | グローバルID |
|--------|-----------|-------------|
| yolo26n.pt (COCO) | 0-79 | 0-79 |
| ycb_best.pt (YCB) | 0-84 | 80-164 |
| custom.pt | 0-N | 165+ |

## 推奨ケース

| ケース | 推奨手法 |
|--------|---------|
| リアルタイム検出が必要 | 追加学習 |
| 精度が最優先 | アンサンブル |
| モデルを頻繁に更新 | アンサンブル |
| エッジデバイス | 追加学習 |
| GPU複数台あり | アンサンブル（並列実行可） |

## パフォーマンス最適化

### バッチ処理

```python
# 複数画像を一括処理
results = []
for image in images:
    detections = detector.predict(image)
    results.append(detections)
```

### 並列推論（マルチGPU）

```python
# 各モデルを別GPUに配置
detector = EnsembleDetector(
    model_paths=['model1.pt', 'model2.pt'],
    devices=['cuda:0', 'cuda:1'],
)
```

## 関連ドキュメント

- [追加学習](incremental-learning.md)
- [パイプライン実行](pipeline.md)
