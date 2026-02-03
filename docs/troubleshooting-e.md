# Troubleshooting

This document explains potential issues and solutions for YCB SynthForge.

## GPU Related

### GPU Not Recognized

```bash
# Check NVIDIA on host
nvidia-smi

# Check inside container
docker compose run --rm yolo26_train nvidia-smi
```

**Solution:**
- Verify NVIDIA Container Toolkit is installed
- Restart Docker daemon: `sudo systemctl restart docker`

### Out of Memory Error

Increase `shm_size` in `docker-compose.yml`:

```yaml
services:
  blenderproc:
    shm_size: '16gb'
  yolo26_train:
    shm_size: '16gb'
```

### CUDA Out of Memory

Reduce batch size during training:

```bash
--batch 8   # Reduce from 16 to 8
```

## YCB Model Related

### YCB Model Not Found

google_16k or tsdf format models are required:

```
models/ycb/{object_name}/google_16k/textured.obj
models/ycb/{object_name}/tsdf/textured.obj
```

**Solution:**
```bash
# google_16k format
python scripts/download_ycb_models.py --all --format google_16k

# tsdf format (required for some objects)
python scripts/download_ycb_models.py --all --format berkeley
```

### tsdf Format Textures Not Displaying

tsdf format OBJ files lack material references (usemtl).

**Solution:**
```bash
docker compose run --rm fix_tsdf_materials
```

### Corrupted Texture Display

You may be using `poisson` format.

**Solution:**
Use `google_16k` or `tsdf` format instead.

### Specific Object Mesh Distortion

**Verification:**
```bash
# Generate thumbnails (google_16k/tsdf comparison for all objects)
docker compose run --rm thumbnail_generator

# Check results
xdg-open data/thumbnails/comparison_grid.png
```

**Solution:**
If you find problematic objects, configure in `generate_dataset.py`:

```python
# scripts/blenderproc/generate_dataset.py

# Objects to completely exclude
EXCLUDED_OBJECTS = {
    "072-b_toy_airplane",
    "problematic_object_name",  # Add
}

# Objects to use tsdf format (when google_16k has issues)
USE_TSDF_FORMAT = {
    "001_chips_can",
    "problematic_object_name",  # Add
}
```

### Automatic Mesh Quality Validation

```bash
# Automatic mesh validation (checks Non-manifold edges, etc.)
docker compose run --rm mesh_validator

# Check results
cat data/mesh_validation_results.json
```

## Docker Related

### docker-compose Error

Legacy `docker-compose` (v1.x) doesn't support this Compose file format:

```bash
# Error example
The Compose file is invalid because: Unsupported config option for services

# Solution: Use docker compose (v2+)
docker compose run -d blenderproc  # ○ Correct
docker-compose run -d blenderproc  # × Legacy not supported
```

### Container Crashes

Check logs:
```bash
docker logs <container_id>
```

If shared memory is insufficient, increase `shm_size`.

## Python/Dependencies

### NumPy Compatibility Warning

`numpy<2` is specified in `Dockerfile.yolo26`. If warnings appear, rebuild:

```bash
docker compose build yolo26_train --no-cache
```

### Module Not Found

```bash
# Rebuild container
docker compose build --no-cache
```

## Data Generation Related

### Objects Not Appearing in Generated Images

**Possible causes:**
1. Object scale is too large/small
2. Camera distance is inappropriate
3. Objects are placed outside the scene

**Verification:**
```bash
# Check object sizes
python3 << 'EOF'
from pathlib import Path

def get_obj_size(obj_path):
    min_c, max_c = [float('inf')]*3, [float('-inf')]*3
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                for i in range(3):
                    v = float(parts[i+1])
                    min_c[i], max_c[i] = min(min_c[i], v), max(max_c[i], v)
    return [max_c[i] - min_c[i] for i in range(3)]

for obj in Path('models').glob('**/textured.obj'):
    size = get_obj_size(obj)
    print(f"{obj}: {size[0]*100:.1f} x {size[1]*100:.1f} x {size[2]*100:.1f} cm")
EOF
```

### No Annotations

Check if COCO format `annotations.json` is generated:
```bash
ls -la data/synthetic/coco/
cat data/synthetic/coco/annotations.json | head -100
```

## Training Related

### Training Not Converging

**Possible causes:**
1. Learning rate is too high
2. Insufficient data
3. Class imbalance

**Solutions:**
- Reduce learning rate: `--lr0 0.001`
- Increase data
- Check class balance

### Low mAP

**Possible causes:**
1. Insufficient data diversity
2. Inadequate domain randomization
3. Model size is too small

**Solutions:**
- Use larger model: `yolo26m` → `yolo26l`
- Review data generation settings
- Increase number of epochs

## Related Documentation

- [Setup Guide](setup-e.md)
- [Configuration Guide](configuration-e.md)
- [Pipeline Execution](pipeline-e.md)
