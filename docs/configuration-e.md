# Configuration Guide

This document explains the configuration files for YCB SynthForge.

## Data Generation Settings

`scripts/blenderproc/config.yaml`

```yaml
# Model source settings (multiple sources supported)
model_sources:
  ycb:
    path: "/workspace/models/ycb"
    include: []                   # Empty = use all objects
    # include:                    # To use specific objects only
    #   - "002_master_chef_can"
    #   - "005_tomato_soup_can"

  tidbots:                        # Custom model source
    path: "/workspace/models/tidbots"
    include: []

scene:
  num_images: 30000               # Number of images to generate
  objects_per_scene: [2, 8]       # Objects per scene [min, max]

rendering:
  engine: "CYCLES"                # Rendering engine
  samples: 32                     # Sample count (32=fast, 128=high quality)
  use_denoising: true             # Enable denoising
  use_gpu: true                   # Use GPU

camera:
  distance: [0.4, 0.9]            # Camera distance [min, max]
  elevation: [35, 65]             # Elevation [min, max]
  azimuth: [0, 360]               # Azimuth [min, max]

lighting:
  num_lights: [3, 5]              # Number of lights [min, max]
  intensity: [800, 2000]          # Light intensity [min, max]
  ambient: [0.4, 0.7]             # Ambient light [min, max]

placement:
  position:
    x_range: [-0.25, 0.25]        # X-direction placement range
    y_range: [-0.25, 0.25]        # Y-direction placement range
  use_physics: false              # Physics simulation (false=grid placement)
```

### Parameter Details

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `scene.num_images` | Total number of images to generate | 30,000 |
| `scene.objects_per_scene` | Range of objects per scene | [2, 8] |
| `rendering.samples` | Ray tracing samples. Higher = better quality but slower | 32 (fast) / 128 (high quality) |
| `camera.distance` | Distance from camera to objects (meters) | [0.4, 0.9] |
| `camera.elevation` | Camera elevation angle (degrees) | [35, 65] |
| `lighting.num_lights` | Number of lights in scene | [3, 5] |
| `placement.use_physics` | Drop objects using physics simulation | false |

## Training Settings

`scripts/training/train_config.yaml`

```yaml
model:
  architecture: yolo26n           # Model architecture (nano/small/medium)
  weights: /workspace/weights/yolo26n.pt  # Pre-trained weights
  num_classes: 85                 # Number of classes

training:
  epochs: 100                     # Number of epochs
  batch_size: 16                  # Batch size
  imgsz: 640                      # Input image size
  optimizer: auto                 # Optimizer
  lr0: 0.01                       # Initial learning rate
  patience: 20                    # Early stopping patience epochs

augmentation:
  mosaic: 1.0                     # Mosaic augmentation probability
  mixup: 0.1                      # MixUp augmentation probability
  hsv_h: 0.015                    # Hue variation
  hsv_s: 0.7                      # Saturation variation
  hsv_v: 0.4                      # Value variation
```

### Parameter Details

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `model.architecture` | YOLO model size | yolo26n (fast) / yolo26m (balanced) |
| `training.epochs` | Number of training epochs | 50-100 |
| `training.batch_size` | Batch size (VRAM dependent) | 16 (24GB) / 8 (12GB) |
| `training.imgsz` | Input image size | 640 |
| `training.patience` | Epochs to wait before early stopping | 20 |
| `augmentation.mosaic` | Mosaic augmentation usage probability | 1.0 |

## Output Directory Structure

### Training Results

```
outputs/trained_models/ycb_yolo26_run/
├── weights/
│   ├── best.pt              # Best model (by mAP)
│   ├── last.pt              # Final epoch model
│   └── epoch*.pt            # Checkpoints
├── args.yaml                # Training parameters
├── results.csv              # Metrics per epoch
├── labels.jpg               # Label distribution
└── train_batch*.jpg         # Training batch samples
```

### Inference Results

```
outputs/inference_results/predictions/
├── *.jpg                    # Detection result images with bounding boxes
├── labels/                  # YOLO format label files
└── results.json             # All detection results (JSON)
```

## Docker Compose Settings

`docker-compose.yml`

Main service settings:

```yaml
services:
  blenderproc:
    shm_size: '8gb'              # Shared memory size
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]

  yolo26_train:
    shm_size: '16gb'             # Training needs larger shared memory
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### Memory Settings

If out of memory errors occur, increase `shm_size`:

```yaml
services:
  blenderproc:
    shm_size: '16gb'
  yolo26_train:
    shm_size: '16gb'
```

## Related Documentation

- [Pipeline Execution](pipeline-e.md)
- [Domain Randomization](domain-randomization-e.md)
- [Troubleshooting](troubleshooting-e.md)
