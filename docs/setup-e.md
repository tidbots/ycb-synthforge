# Setup Guide

This document explains how to set up the YCB SynthForge environment.

## Requirements

- Docker & Docker Compose v2+ (uses `docker compose` command)
  - Note: Legacy `docker-compose` (v1.x) is not supported
- NVIDIA GPU (CUDA compatible)
- NVIDIA Container Toolkit
- Recommended: RTX 3090/4090 (24GB VRAM)

## Building Docker Images

```bash
# Build all images
docker compose build

# Build individually
docker compose build blenderproc
docker compose build yolo26_train
```

## Downloading YOLO26 Weights

```bash
# List available models
python scripts/download_weights.py --list

# Download default (nano + small)
python scripts/download_weights.py

# Download specific models
python scripts/download_weights.py --models yolo26n yolo26s yolo26m

# Download all models
python scripts/download_weights.py --all

# Force re-download
python scripts/download_weights.py --models yolo26m --force
```

### Model List

| Model | Parameters | Size | Use Case |
|-------|------------|------|----------|
| yolo26n | 2.6M | ~5 MB | Fastest, for edge devices |
| yolo26s | 9.4M | ~19 MB | Balanced |
| yolo26m | 20.1M | ~40 MB | Recommended, general purpose |
| yolo26l | 25.3M | ~49 MB | High accuracy |
| yolo26x | 56.9M | ~109 MB | Highest accuracy |

## Downloading YCB 3D Models

**Important**: This project uses `google_16k` format as the base, with `tsdf` format for some objects.

```bash
# List objects
python scripts/download_ycb_models.py --list

# List categories
python scripts/download_ycb_models.py --list-categories

# Download all objects in google_16k format
python scripts/download_ycb_models.py --all --format google_16k

# Download tsdf format (required for some objects)
python scripts/download_ycb_models.py --all --format berkeley

# Download by category
python scripts/download_ycb_models.py --category food fruit kitchen --format google_16k

# Download specific objects only
python scripts/download_ycb_models.py --objects 003_cracker_box 005_tomato_soup_can --format google_16k

# Force re-download
python scripts/download_ycb_models.py --all --format google_16k --force
```

### Format List

| Format | Description | Usage |
|--------|-------------|-------|
| google_16k | 16k polygons, high-quality textures | Primary format (72 objects) |
| tsdf | TSDF reconstructed mesh | Used for some objects (13 objects) |
| google_64k | 64k polygons, higher resolution | |
| google_512k | 512k polygons, highest resolution | |
| poisson | Poisson reconstruction (not recommended) | Texture corruption issues |

### Fixing tsdf Format Materials

tsdf format OBJ files lack material references, so fixes are needed after initial download:

```bash
# Fix tsdf format OBJ files (add usemtl lines)
docker compose run --rm fix_tsdf_materials
```

**Note**: Original files are automatically backed up as `.obj.backup`.

## Downloading CC0 Textures

Download PBR textures from [ambientCG](https://ambientcg.com/) (CC0 license).

```bash
# List categories
python scripts/download_cctextures.py --list-categories

# Download default 100 textures
python scripts/download_cctextures.py

# Download by category
python scripts/download_cctextures.py --category floor wall table

# Download by prefix (Wood*, Metal* 20 each)
python scripts/download_cctextures.py --prefix Wood Metal --limit 20

# Download specific textures
python scripts/download_cctextures.py --textures Wood001 Metal002 Tiles005

# Download high resolution (1K/2K/4K/8K)
python scripts/download_cctextures.py --resolution 4K

# Search online
python scripts/download_cctextures.py --search Marble --limit 30
```

### Category List

| Category | Use | Prefixes |
|----------|-----|----------|
| floor | Floors | Wood, WoodFloor, Tiles, Marble, Concrete |
| wall | Walls | Bricks, PaintedPlaster, Wallpaper, Facade |
| table | Tables | Wood, Metal, Plastic, Marble, Granite |
| metal | Metals | Metal, MetalPlates, DiamondPlate, Rust |
| fabric | Fabrics | Fabric, Leather, Carpet, Wicker |
| natural | Natural | Ground, Grass, Rock, Gravel, Sand |
| industrial | Industrial | Asphalt, Concrete, CorrugatedSteel |

## Next Steps

Once setup is complete, proceed to [Pipeline Execution](pipeline-e.md).
