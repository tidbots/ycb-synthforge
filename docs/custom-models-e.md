# Adding Custom Models

This document explains how to add custom 3D models beyond YCB to include in training data.

## Directory Structure

```
models/
├── ycb/                          # YCB objects (Class ID: 0-102)
│   └── 002_master_chef_can/
│       └── google_16k/
│           └── textured.obj
└── tidbots/                      # Custom objects (Class ID: 103-)
    ├── my_bottle/
    │   └── google_16k/
    │       ├── textured.obj
    │       └── textured.png
    └── my_gripper/
        └── google_16k/
            ├── textured.obj
            └── textured.png
```

## Configuration File

Set model sources in `scripts/blenderproc/config.yaml`:

```yaml
model_sources:
  # YCB models
  ycb:
    path: "/workspace/models/ycb"
    include:                      # Use specific objects only
      - "002_master_chef_can"
      - "005_tomato_soup_can"
      - "006_mustard_bottle"

  # Custom models
  tidbots:
    path: "/workspace/models/tidbots"
    include: []                   # Empty = use all objects
```

## Class ID Assignment

| Source | Class ID Range | Description |
|--------|----------------|-------------|
| ycb | 0-102 | Maintains existing YCB IDs |
| tidbots | 103- | Auto-assigned sequentially |
| (additional sources) | Continues from last | Assigned in source order |

## Supported Model Formats

The following formats are auto-detected (in priority order):

1. `object_name/google_16k/textured.obj` (YCB format)
2. `object_name/tsdf/textured.obj`
3. `object_name/textured.obj` (simple format)
4. `object_name/*.obj` (any OBJ)

## 3D Model Conversion

Scripts are provided to convert downloaded 3D models to OBJ format.

### Blender format (.blend) → OBJ

```bash
docker compose run --rm -v /tmp/mymodel:/tmp/mymodel blenderproc \
  blenderproc run /workspace/scripts/convert_blend_to_obj.py \
  /tmp/mymodel/model.blend \
  /tmp/mymodel/output \
  /tmp/mymodel/textures  # Texture directory (optional)
```

### COLLADA format (.dae) → OBJ

```bash
docker compose run --rm -v /tmp/mymodel:/tmp/mymodel blenderproc \
  blenderproc run /workspace/scripts/convert_dae_to_obj.py \
  /tmp/mymodel/model.dae \
  /tmp/mymodel/output
```

### FBX format (.fbx) → OBJ

```bash
docker compose run --rm -v /tmp/mymodel:/tmp/mymodel blenderproc \
  blenderproc run /workspace/scripts/convert_fbx_to_obj.py \
  /tmp/mymodel/model.fbx \
  /tmp/mymodel/output
```

### Copy After Conversion

```bash
mkdir -p models/tidbots/my_object/google_16k
cp /tmp/mymodel/output/* models/tidbots/my_object/google_16k/
```

## Checking/Fixing Model Scale

Downloaded 3D models often have inconsistent scales. If objects don't appear in generated images, check the scale.

### Check Scale

```bash
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

for obj in Path('models/tidbots').glob('*/google_16k/textured.obj'):
    size = get_obj_size(obj)
    print(f"{obj.parent.parent.name}: {size[0]*100:.1f} x {size[1]*100:.1f} x {size[2]*100:.1f} cm")
EOF
```

### Fix Scale (example: scale down to 0.03x)

```bash
python3 << 'EOF'
from pathlib import Path
import shutil

def scale_obj(obj_path, scale):
    shutil.copy(obj_path, str(obj_path) + '.backup')
    lines = []
    with open(obj_path) as f:
        for line in f:
            if line.startswith('v '):
                p = line.split()
                lines.append(f"v {float(p[1])*scale:.6f} {float(p[2])*scale:.6f} {float(p[3])*scale:.6f}\n")
            else:
                lines.append(line)
    with open(obj_path, 'w') as f:
        f.writelines(lines)

# Example: scale coke_zero to 0.03x
scale_obj(Path('models/tidbots/coke_zero/google_16k/textured.obj'), 0.03)
EOF
```

### Typical Object Sizes

| Object | Actual Size |
|--------|-------------|
| Can (350ml) | 6-7 × 12-13 cm |
| Plastic bottle | 6-8 × 20-25 cm |
| Apple | 7-8 × 7-8 cm |

## Training with Custom Models Only

To train with only custom models without YCB:

```yaml
# scripts/blenderproc/config.yaml
model_sources:
  # YCB disabled
  # ycb:
  #   path: "/workspace/models/ycb"
  #   include: []

  # Use custom models only
  tidbots:
    path: "/workspace/models/tidbots"
    include: []  # Empty = use all objects

scene:
  num_images: 2000          # 2000 images sufficient for few classes
  objects_per_scene: [1, 5]  # Adjust based on number of classes
```

### Recommended Data Volume

| Classes | Recommended Images | Per Class |
|---------|-------------------|-----------|
| 5 | 2,000 | 400 |
| 10 | 3,000 | 300 |
| 20 | 5,000 | 250 |
| 50+ | 10,000+ | 200+ |

## Using Specific Objects Only

To use only specific objects instead of all:

```yaml
model_sources:
  ycb:
    path: "/workspace/models/ycb"
    include:
      - "002_master_chef_can"     # Can
      - "003_cracker_box"         # Box
      - "006_mustard_bottle"      # Bottle
      - "024_bowl"                # Dish
      - "025_mug"                 # Mug
```

This allows you to narrow down training targets for efficient model creation.

## Related Documentation

- [Configuration Guide](configuration-e.md)
- [YCB Classes](ycb-classes-e.md)
- [Incremental Learning](incremental-learning-e.md)
