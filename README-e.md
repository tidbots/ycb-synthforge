# YCB SynthForge

End-to-end pipeline for synthetic data generation with BlenderProc and YCB object detection with YOLO26

## Overview

YCB SynthForge is an end-to-end pipeline for detecting 85 types of YCB objects.

- **Synthetic Data Generation**: Photorealistic rendering with BlenderProc
- **Domain Randomization**: Diverse data generation for Sim-to-Real transfer
- **YOLO26 Training**: Fine-tuning COCO pre-trained models
- **google_16k + tsdf format**: Automatic selection of optimal format per object

## Quick Start

### 1. Setup

```bash
# Build Docker images
docker compose build blenderproc yolo26_train

# Download YOLO26 weights
python scripts/download_weights.py

# Download YCB 3D models
python scripts/download_ycb_models.py --all --format google_16k
python scripts/download_ycb_models.py --all --format berkeley

# Fix tsdf format materials
docker compose run --rm fix_tsdf_materials
```

### 2. Synthetic Data Generation

```bash
docker compose run -d blenderproc
```

### 3. YOLO26 Training

```bash
# Convert data format
docker compose run --rm yolo26_train python \
  scripts/data_processing/coco_to_yolo.py \
  --coco_json data/synthetic/coco/annotations.json \
  --output_dir data/synthetic/yolo

# Run training
docker compose run --rm yolo26_train python3 \
  /workspace/scripts/training/train_yolo26.py \
  --data /workspace/yolo_dataset/dataset.yaml \
  --weights /workspace/weights/yolo26m.pt \
  --epochs 50
```

### 4. Inference

```bash
docker compose run --rm yolo26_inference python3 \
  /workspace/scripts/evaluation/inference.py \
  --model /workspace/outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --source /workspace/yolo_dataset/images/val
```

### 5. Real-time Detection (Webcam)

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/ycb_yolo26_run/weights/best.pt \
  --camera 0 \
  --conf 0.5
```

Stabilized real-time detection features:
- ByteTrack tracking
- Confidence & coordinate smoothing
- Class prediction stabilization
- Hysteresis to prevent flickering
- Trajectory & velocity display

See [Real-time Detection Guide](docs/realtime-detection-e.md) for details.

## Sample Generated Images

![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/scene_000009.png)
![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/scene_000013.png)
![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/scene_000029.png)
![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/scene_000455.png)

## Sample Detection Results

![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/inference_sample1.jpg)
![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/inference_sample2.jpg)
![](https://github.com/tidbots/ycb-synthforge/blob/main/fig/inference_sample3.jpg)

## Project Structure

```
ycb-synthforge/
├── docker/                       # Docker files
├── docker-compose.yml
├── models/
│   ├── ycb/                      # YCB 3D models (85 classes)
│   └── tidbots/                  # Custom 3D models
├── resources/cctextures/         # CC0 textures
├── weights/                      # YOLO26 weights
├── scripts/
│   ├── blenderproc/              # Data generation scripts
│   ├── data_processing/          # Data conversion
│   ├── training/                 # Training scripts
│   └── inference/                # Inference scripts
├── data/synthetic/               # Generated data
├── outputs/                      # Output results
└── docs/                         # Detailed documentation
```

## Documentation

| Document | Description |
|----------|-------------|
| [Setup Guide](docs/setup-e.md) | Environment setup, downloading weights, models, textures |
| [Pipeline Execution](docs/pipeline-e.md) | Data generation, training, evaluation, inference |
| [Configuration](docs/configuration-e.md) | Detailed config.yaml settings |
| [Domain Randomization](docs/domain-randomization-e.md) | Randomization for Sim-to-Real transfer |
| [Custom Models](docs/custom-models-e.md) | Adding custom 3D models |
| [YCB Classes](docs/ycb-classes-e.md) | 85 types of YCB objects |
| [Incremental Learning](docs/incremental-learning-e.md) | Adding new objects |
| [Ensemble Inference](docs/ensemble-inference-e.md) | Combining multiple models |
| [Real-time Detection](docs/realtime-detection-e.md) | Real-time object detection with webcam |
| [Troubleshooting](docs/troubleshooting-e.md) | Problem solving |

## Requirements

- Docker & Docker Compose v2+
- NVIDIA GPU (CUDA compatible)
- NVIDIA Container Toolkit
- Recommended: RTX 3090/4090 (24GB VRAM)

## License

- YCB Models: [YCB Object and Model Set License](https://www.ycbbenchmarks.com/)
- CC0 Textures: [CC0 1.0 Universal](https://ambientcg.com/)
- BlenderProc: MIT License
- Ultralytics YOLO: AGPL-3.0

## References

- [BlenderProc](https://github.com/DLR-RM/BlenderProc)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YCB Object Dataset](https://www.ycbbenchmarks.com/)
- [ambientCG Textures](https://ambientcg.com/)
