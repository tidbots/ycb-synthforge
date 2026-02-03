# Ensemble Inference

This document explains how to combine multiple models for inference.

## Overview

Ensemble inference is a technique that combines multiple trained models for inference. No retraining is needed when adding or removing models.

## Comparison with Incremental Learning

| Item | Incremental Learning | Ensemble Inference |
|------|---------------------|-------------------|
| Inference speed | Fast (1 model) | Slow (N inferences) |
| Memory | Low | High (N times) |
| Accuracy retention | Risk of forgetting | Each model maintained |
| Flexibility | Retraining needed | Easy to add/remove models |

## Image Inference

```bash
docker compose run --rm ensemble_inference python \
  scripts/inference/ensemble_inference.py \
  --models weights/yolo26n.pt outputs/ycb_best.pt \
  --source data/test_images/ \
  --output outputs/ensemble_results \
  --show-model
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--models` | Model paths (multiple allowed) |
| `--source` | Input image or directory |
| `--output` | Output directory |
| `--show-model` | Display model name in detection results |
| `--conf` | Confidence threshold (default: 0.3) |
| `--iou` | IoU threshold for NMS (default: 0.5) |

## Real-time Inference (Webcam)

```bash
docker compose run --rm ensemble_inference python \
  scripts/inference/ensemble_inference.py \
  --models weights/yolo26n.pt outputs/ycb_best.pt \
  --source 0 \
  --realtime
```

## Python Code Usage

```python
from ensemble_inference import EnsembleDetector

# Initialize multiple models
detector = EnsembleDetector(
    model_paths=[
        'yolo26n.pt',      # COCO 80 classes (ID: 0-79)
        'ycb_best.pt',     # YCB 85 classes  (ID: 80-164)
        'custom.pt',       # Custom         (ID: 165+)
    ],
    conf_threshold=0.3,
    iou_threshold=0.5,
)

# Inference
detections = detector.predict(image)

# Draw results
result = detector.draw_detections(image, detections, show_model=True)
```

### EnsembleDetector API

```python
class EnsembleDetector:
    def __init__(
        self,
        model_paths: List[str],    # Model file paths
        conf_threshold: float,      # Confidence threshold
        iou_threshold: float,       # IoU threshold for NMS
        device: str = 'cuda',       # Device ('cuda' or 'cpu')
    )

    def predict(
        self,
        image: np.ndarray,          # Input image (BGR)
    ) -> List[Detection]:           # Detection results list

    def draw_detections(
        self,
        image: np.ndarray,          # Input image
        detections: List[Detection], # Detection results
        show_model: bool = False,   # Show model name
    ) -> np.ndarray:                # Image with drawings
```

### Detection Format

```python
@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float                 # Confidence score
    class_id: int                     # Global class ID
    class_name: str                   # Class name
    model_name: str                   # Source model name
```

## Class ID Management

In ensemble, each model's class IDs are mapped to global IDs:

| Model | Local ID | Global ID |
|-------|----------|-----------|
| yolo26n.pt (COCO) | 0-79 | 0-79 |
| ycb_best.pt (YCB) | 0-84 | 80-164 |
| custom.pt | 0-N | 165+ |

## Recommended Cases

| Case | Recommended Method |
|------|-------------------|
| Real-time detection needed | Incremental learning |
| Accuracy is priority | Ensemble |
| Frequent model updates | Ensemble |
| Edge devices | Incremental learning |
| Multiple GPUs available | Ensemble (parallel execution possible) |

## Performance Optimization

### Batch Processing

```python
# Process multiple images at once
results = []
for image in images:
    detections = detector.predict(image)
    results.append(detections)
```

### Parallel Inference (Multi-GPU)

```python
# Place each model on different GPU
detector = EnsembleDetector(
    model_paths=['model1.pt', 'model2.pt'],
    devices=['cuda:0', 'cuda:1'],
)
```

## Related Documentation

- [Incremental Learning](incremental-learning-e.md)
- [Pipeline Execution](pipeline-e.md)
