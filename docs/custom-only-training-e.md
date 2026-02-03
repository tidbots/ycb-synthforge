# Custom Objects Only Training Guide

This guide explains how to train a YOLO26 model using only your custom objects, without the YCB dataset.

## Overview

This guide describes the steps to train an object detection model using only your own 3D models located in custom model directories such as `models/tidbots/`.

## Prerequisites

- Docker and Docker Compose installed
- 3D scanned object models (OBJ format)
- Sufficient disk space (approximately 10GB recommended for synthetic data)

## Step 1: Prepare Custom Models

### Directory Structure

Organize your custom models in the following structure:

```
models/
└── tidbots/              # Custom model source name
    ├── object_name_1/    # Object name (= class name)
    │   ├── model.obj     # 3D model (any filename)
    │   ├── materials.mtl # Material file
    │   └── texture.jpg   # Texture image
    ├── object_name_2/
    │   └── ...
    └── ...
```

### Model Requirements

- **Format**: OBJ format (`.obj`)
- **Texture**: JPG or PNG format
- **Scale**: Any (automatically normalized)
- **Naming**: Folder name is used as the class name

> **Note**: Even if models are created in millimeter units, they will be automatically normalized to meter units (approximately 15cm).

## Step 2: Edit Configuration File

Edit `scripts/blenderproc/config.yaml` to enable only the custom model source.

```yaml
# Model sources configuration
model_sources:
  # Disable YCB (comment out)
  # ycb:
  #   path: "/workspace/models/ycb"
  #   include: []

  # Enable custom model source
  tidbots:
    path: "/workspace/models/tidbots"
    include: []  # Empty list = use all objects
```

### To Use Only Specific Objects

```yaml
  tidbots:
    path: "/workspace/models/tidbots"
    include:
      - "aquarius"
      - "chipstar"
      - "coffee_1"
```

### Configuration Key Points

| Setting | Description |
|---------|-------------|
| `path` | Path inside Docker container (`/workspace/models/...`) |
| `include: []` | Empty list = use all objects in directory |
| `include: [...]` | Use only specified objects |

## Step 3: Generate Synthetic Data

Generate synthetic images using BlenderProc.

```bash
docker compose run --rm blenderproc blenderproc run \
  /workspace/scripts/blenderproc/generate_dataset.py \
  --num_scenes 2000
```

### Generation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num_scenes` | 2000 | Number of images to generate |
| `--output` | `/workspace/data/synthetic/coco` | Output directory |
| `--seed` | 123 | Random seed (for reproducibility) |

### Output

```
data/synthetic/coco/
├── images/
│   ├── scene_000000.png
│   ├── scene_000001.png
│   └── ...
└── annotations.json    # COCO format annotations
```

## Step 4: Convert COCO to YOLO Format

Convert the generated COCO format data to YOLO format.

```bash
docker compose run --rm yolo26_train python3 \
  /workspace/scripts/data_processing/coco_to_yolo.py \
  --coco_json /workspace/data/synthetic/coco/annotations.json \
  --output_dir /workspace/data/synthetic/yolo
```

### Output

```
data/synthetic/yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml    # YOLO training configuration file
```

## Step 5: Train YOLO26 Model

```bash
docker compose run --rm yolo26_train python3 \
  /workspace/scripts/training/train_yolo26.py \
  --data /workspace/data/synthetic/yolo/dataset.yaml \
  --weights /workspace/weights/yolo26m.pt \
  --epochs 50 \
  --name custom_objects
```

### Training Parameters

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--epochs` | 50-100 | Number of epochs |
| `--batch` | 16 | Batch size (adjust based on GPU memory) |
| `--imgsz` | 640 | Input image size |
| `--weights` | `yolo26m.pt` | Pre-trained weights |

### Output

```
outputs/trained_models/custom_objects/
├── weights/
│   ├── best.pt     # Best model
│   └── last.pt     # Final model
└── results.csv     # Training metrics
```

## Step 6: Inference Test

Run inference with the trained model.

```bash
docker compose run --rm yolo26_inference python3 \
  /workspace/scripts/evaluation/inference.py \
  --model /workspace/outputs/trained_models/custom_objects/weights/best.pt \
  --source /path/to/test/images
```

## Troubleshooting

### Images Are Completely White

**Cause**: Model scale is too large (e.g., millimeter units)

**Solution**: The `normalize_object_scale()` function in `generate_dataset.py` automatically normalizes the scale. Verify this feature is working correctly.

### Textures Not Displayed

**Cause**: Texture path in MTL file is incorrect

**Solution**:
1. Ensure texture paths in MTL file are relative paths
2. Verify texture filename matches the MTL file specification

```
# Example materials.mtl
newmtl material0
map_Kd texture.jpg
```

### Annotations Not Generated

**Cause**: Objects are outside the camera's field of view

**Solution**: Adjust camera settings in `config.yaml`
```yaml
camera:
  distance: [0.3, 0.8]  # Closer distance
  elevation: [30, 60]   # Appropriate angle
```

## Class ID Assignment

When using only custom models, class IDs are automatically assigned starting from 0:

| Folder Name | Class ID |
|-------------|----------|
| aquarius | 0 |
| chipstar | 1 |
| coffee_1 | 2 |
| coffee_2 | 3 |
| cupnoodle_seafood | 4 |
| redbull | 5 |

> **Note**: Class IDs are assigned in alphabetical order of folder names.

## Related Documentation

- [Adding Custom Models](custom-models-e.md)
- [Configuration Reference](configuration-e.md)
- [Domain Randomization](domain-randomization-e.md)
- [Troubleshooting](troubleshooting-e.md)
