# Pipeline Execution Guide

This document explains the pipeline execution steps from data generation to inference.

## 1. Synthetic Data Generation

### Basic Execution

```bash
# Generate in background (follows num_images setting in config.yaml)
docker compose run -d blenderproc

# Check container ID
docker ps | grep blenderproc

# Check progress (number of generated images)
ls data/synthetic/coco/images/ | wc -l

# Check real-time logs
docker logs -f <container_id>

# Or check latest logs only
docker logs --tail 30 <container_id>
```

### Configuration

Number of images is set in `scripts/blenderproc/config.yaml` under `scene.num_images` (default: 30,000 images).

For detailed settings, see [Configuration Guide](configuration-e.md).

### Generation Time Estimates

| Setting | Samples | Speed | Time for 30,000 images |
|---------|---------|-------|------------------------|
| Fast | 32 | ~45 images/min | ~11 hours |
| High Quality | 128 | ~10 images/min | ~50 hours |

### Sample Generated Images

![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/scene_000009.png)
![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/scene_000013.png)
![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/scene_000029.png)
![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/scene_000455.png)

## 2. COCO to YOLO Format Conversion

```bash
docker compose run --rm yolo26_train python \
  scripts/data_processing/coco_to_yolo.py \
  --coco_json data/synthetic/coco/annotations.json \
  --output_dir data/synthetic/yolo \
  --train_ratio 0.833 \
  --val_ratio 0.083 \
  --test_ratio 0.083
```

### Dataset Composition

| Split | Images | Ratio | Purpose |
|-------|--------|-------|---------|
| Train | 25,000 | 83.3% | Model training |
| Val | 2,500 | 8.3% | Hyperparameter tuning |
| Test | 2,500 | 8.3% | Final evaluation |

## 3. YOLO26 Training

```bash
# Train with YOLO26m (Medium) model
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

### Training Results Example (YOLO26m, 50 epochs)

| Metric | Value |
|--------|-------|
| **mAP50** | 97.52% |
| **mAP50-95** | 95.30% |
| **Precision** | 97.32% |
| **Recall** | 94.43% |
| Training Time | ~59 min (RTX 4090) |

### Output Files

Trained weights are saved to `outputs/trained_models/ycb_yolo26_run/weights/`:
- `best.pt` - Best accuracy model (recommended for inference)
- `last.pt` - Final epoch model

```
outputs/trained_models/ycb_yolo26_run/
├── weights/
│   ├── best.pt              # Best model (by mAP)
│   ├── last.pt              # Final epoch model
│   └── epoch*.pt            # Checkpoints (every 10 epochs)
├── args.yaml                # Training parameters
├── results.csv              # Metrics per epoch
├── labels.jpg               # Label distribution
└── train_batch*.jpg         # Training batch samples
```

## 4. Evaluation

```bash
docker compose run --rm yolo26_train python \
  scripts/training/evaluate.py \
  --weights outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --data yolo_dataset/dataset.yaml
```

## 5. Inference

### Batch Inference

```bash
# Run inference on validation images
docker compose run --rm \
  -v $(pwd)/yolo_dataset:/workspace/yolo_dataset:ro \
  yolo26_inference python3 /workspace/scripts/evaluation/inference.py \
  --model /workspace/outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --source /workspace/yolo_dataset/images/val \
  --output /workspace/outputs/inference_results \
  --conf 0.5 \
  --device 0
```

### Output Files

Inference results are saved to `outputs/inference_results/predictions/`:
- `*.jpg` - Detection result images with bounding boxes
- `labels/` - YOLO format label files
- `results.json` - All detection results in JSON

### Sample Detection Results

![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/inference_sample1.jpg)
![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/inference_sample2.jpg)
![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/inference_sample3.jpg)

## 6. Real-time Detection (Webcam)

Real-time object detection using USB webcam:

```bash
# Run via Docker
./run_realtime_detection.sh

# With options
./run_realtime_detection.sh --camera 0 --conf 0.5

# Run directly on host (requires: pip install ultralytics opencv-python)
./run_realtime_detection_host.sh
```

### Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save screenshot |
| `c` | Toggle confidence display |

## Utilities

### Mesh Validation

Automatically validate YCB object mesh quality:

```bash
docker compose run --rm mesh_validator
```

Results are saved to `data/mesh_validation_results.json`.

### Thumbnail Generation

Generate thumbnails comparing google_16k/tsdf formats for all objects:

```bash
docker compose run --rm thumbnail_generator
```

Results:
- `data/thumbnails/*.png` - Individual thumbnails
- `data/thumbnails/comparison_grid.png` - Comparison grid

### All Formats Thumbnail Comparison

Compare all 4 formats (clouds/google_16k/poisson/tsdf) for all objects:

```bash
docker compose run --rm thumbnail_all_formats
```

Results:
- `data/thumbnails_all_formats/*.png` - Individual thumbnails
- `data/thumbnails_all_formats/comparison_grid_all.png` - All formats comparison grid

**Note**: clouds format (point cloud) and poisson format do not support textures, so they appear gray.

## Related Documentation

- [Configuration Guide](configuration-e.md)
- [Domain Randomization](domain-randomization-e.md)
- [Incremental Learning](incremental-learning-e.md)
- [Ensemble Inference](ensemble-inference-e.md)
