# Incremental Learning

This document explains how to add new objects to a trained model.

## Overview

Incremental learning is a technique for adding new classes to an existing trained model. It efficiently extends the model without retraining on all data.

## Method Comparison

| Method | Training Time | Accuracy Retention | Implementation Difficulty |
|--------|--------------|-------------------|--------------------------|
| Full data retraining | Long | High | Easy |
| Replay (subset) | Short | Slightly lower | Easy |
| Backbone freezing | Short | Slightly lower | Easy |
| Knowledge distillation | Medium | High | Moderately complex |

## Step 1: Create Subset

Extract representative samples from original data (class-balanced sampling):

```bash
docker compose run --rm yolo26_train python \
  scripts/data_processing/create_subset.py \
  --source /workspace/yolo_dataset \
  --output /workspace/data/ycb_subset \
  --num_samples 5000 \
  --val_samples 500
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--source` | Original dataset path |
| `--output` | Output path |
| `--num_samples` | Number of training samples |
| `--val_samples` | Number of validation samples |

## Step 2: Merge Datasets

Merge subset with new object data:

```bash
docker compose run --rm yolo26_train python \
  scripts/data_processing/merge_for_incremental.py \
  --base /workspace/data/ycb_subset \
  --new /workspace/data/new_objects \
  --output /workspace/data/merged_dataset
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--base` | Base dataset (subset) |
| `--new` | New object dataset |
| `--output` | Merged output destination |

## Step 3: Incremental Training with Backbone Freezing

```bash
docker compose run --rm yolo26_train python \
  scripts/training/train_incremental.py \
  --weights /workspace/outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --data /workspace/data/merged_dataset/dataset.yaml \
  --freeze 10 \
  --epochs 30 \
  --lr0 0.001
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--weights` | Original trained model |
| `--data` | Merged dataset configuration file |
| `--freeze` | Number of layers to freeze (0-22) |
| `--epochs` | Training epochs |
| `--lr0` | Initial learning rate (set lower for incremental training) |

## Parameter Comparison

| Parameter | Normal Training | Incremental Training |
|-----------|-----------------|---------------------|
| Data volume | 30,000 images | 5,500 images |
| freeze | 0 | 10 |
| lr0 | 0.01 | 0.001 |
| epochs | 50-100 | 30-50 |
| **Estimated time** | ~1 hour | ~15 min |

## Recommended Cases

| Case | Recommended Method |
|------|-------------------|
| 1-5 new classes | Backbone freezing |
| Many new classes | Full data retraining |
| Accuracy retention is priority | Knowledge distillation or full retraining |
| Limited time | Backbone freezing |

## Considerations

1. **Forgetting problem**: Incremental learning may reduce accuracy on original classes (catastrophic forgetting)
2. **Subset importance**: Select replay subsets to maintain diversity of original data
3. **Learning rate**: Use low learning rates in incremental training to protect existing knowledge

## Related Documentation

- [Adding Custom Models](custom-models-e.md)
- [Ensemble Inference](ensemble-inference-e.md)
- [Pipeline Execution](pipeline-e.md)
