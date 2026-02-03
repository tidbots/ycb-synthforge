# Domain Randomization

This document explains domain randomization techniques for reducing the Sim-to-Real gap.

## Overview

Domain randomization is a technique that increases the diversity of synthetic data to reduce the gap between simulation and real environments (Sim-to-Real gap) when applying models trained in simulation to real-world scenarios.

YCB SynthForge randomizes the following elements.

## Randomization Elements

### Background

| Element | Range |
|---------|-------|
| Floor texture | Wood, Concrete, Tiles, Marble, Metal, Fabric |
| Wall texture | Concrete, Plaster, Brick, Paint, Wallpaper |
| Table material | Wood, Metal, Plastic |

### Lighting

| Element | Range |
|---------|-------|
| Number of lights | 3-5 |
| Color temperature | 2700K-6500K |
| Intensity | 800-2000W equivalent |
| Shadow softness | 0.3-0.9 |

### Camera

| Element | Range |
|---------|-------|
| Distance | 0.4-2.0m |
| Elevation | 10-70° |
| Azimuth | 0-360° |
| Exposure | EV -1.5 to +1.5 |
| ISO | 100-3200 |
| Depth of field | f/1.8-11.0 |

### Material

| Element | Range |
|---------|-------|
| Metallic | 0.8-1.0 (metal objects) |
| Roughness | 0.05-0.6 |
| Hue shift | ±10° |

### Object

| Element | Range |
|---------|-------|
| Position | X,Y: ±0.3m |
| Rotation | 0-360° (each axis) |
| Scale | ±5% |

## Configuration

Domain randomization settings are configured in `scripts/blenderproc/config.yaml`.

```yaml
camera:
  distance: [0.4, 0.9]          # [min, max]
  elevation: [35, 65]           # Elevation range
  azimuth: [0, 360]             # Azimuth range

lighting:
  num_lights: [3, 5]            # Number of lights range
  intensity: [800, 2000]        # Intensity range
  ambient: [0.4, 0.7]           # Ambient light range

placement:
  position:
    x_range: [-0.25, 0.25]      # X-direction placement range
    y_range: [-0.25, 0.25]      # Y-direction placement range
```

## Effects

Domain randomization provides the following benefits:

1. **Robustness to lighting conditions**: Stable detection under various lighting environments
2. **Viewpoint diversity**: Improved object recognition from different angles
3. **Background generalization**: Adaptability to various background textures
4. **Object position generalization**: Detection of objects at arbitrary positions in images

## Related Documentation

- [Configuration Guide](configuration-e.md)
- [Pipeline Execution](pipeline-e.md)
