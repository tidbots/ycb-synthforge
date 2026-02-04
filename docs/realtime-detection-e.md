# Real-time Object Detection Guide

This document describes how to use real-time object detection and tracking with a webcam.

## Overview

`scripts/evaluation/realtime_detection.py` provides the following features:

- **Object tracking with ByteTrack** - Reduces detection flickering
- **Moving average coordinates** - Smoothed over 1 second of history
- **Confidence smoothing** - Exponential moving average suppresses sudden score changes
- **Class prediction stabilization** - Majority voting fixes class for same track ID
- **Hysteresis** - Appearance/disappearance thresholds prevent flickering
- **Trajectory visualization** - Shows movement path over the past 1 second
- **Velocity display** - Shows movement speed in px/sec
- **Variable frame rate** - Target 30Hz, dynamically adjustable

## Basic Usage

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/tidbots_6class/weights/best.pt \
  --camera 0 \
  --conf 0.5
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `outputs/trained_models/ycb_yolo26_run/weights/best.pt` | Path to model weights |
| `--camera` | `0` | Camera device ID |
| `--conf` | `0.5` | Confidence threshold |
| `--iou` | `0.45` | IoU threshold for NMS |
| `--imgsz` | `640` | Input image size |
| `--device` | `0` | GPU device |
| `--save-video` | `False` | Enable video recording |
| `--output` | `outputs/realtime_detection.mp4` | Output video path |
| `--target-fps` | `30.0` | Target frame rate (Hz) |

## Keyboard Controls

| Key | Function |
|-----|----------|
| `q` | Quit |
| `s` | Save screenshot |
| `c` | Toggle confidence display |
| `p` | Toggle coordinate display |
| `t` | Toggle trajectory display |
| `v` | Toggle velocity display |
| `+` / `=` | Increase target FPS by 5 (max 60) |
| `-` | Decrease target FPS by 5 (min 5) |

## Stabilization Features

### 1. Confidence Smoothing

Uses Exponential Moving Average (EMA) to suppress sudden score fluctuations.

```
smoothed_conf = α × current_conf + (1 - α) × previous_smoothed_conf
α = 0.3 (default)
```

### 2. Class Prediction Stabilization

For each track ID, the class is fixed by majority vote over the first 5 frames.
This prevents flickering between similar object classes.

### 3. Hysteresis

- **Appearance threshold**: Display starts after 3 consecutive frames detected
- **Disappearance threshold**: Display ends after 5 consecutive frames missing

This prevents flickering from momentary false detections or detection failures.

### 4. Variable Frame Rate

- Default 30Hz
- Dynamically adjustable with `+`/`-` keys (5-60Hz)
- Adapts to GPU load

## Screen Display

```
+------------------------------------------+
| FPS: 28.5/30 | Visible: 2                |
|                                          |
|   +---------------+                      |
|   | cracker_box   |                      |
|   |    0.85       |  <- Smoothed confidence
|   +----~~~~-------+  <- Trajectory (yellow)
|   ID:1 (230,250)     <- Smoothed coords (cyan)
|   45 px/s            <- Velocity (green) |
|                                          |
|          +--------+                      |
|          | banana |  <- Fixed class      |
|          |  0.72  |                      |
|          +--------+                      |
|          ID:2 (450,280)                  |
|          12 px/s                         |
+------------------------------------------+
```

### Display Elements

| Element | Color | Description |
|---------|-------|-------------|
| Bounding box | Track ID dependent | Unique color per track |
| Trajectory | Yellow `(0,255,255)` | Movement path of center point over past 1 second |
| Coordinates | Cyan `(255,255,0)` | Center coordinates smoothed by moving average |
| Velocity | Green `(0,255,0)` | Speed in px/sec calculated from 1-second movement |

## Stabilization Parameters

The following parameters can be adjusted in the source code:

```python
APPEAR_THRESHOLD = 3      # Consecutive frames required to appear
DISAPPEAR_THRESHOLD = 5   # Consecutive frames required to disappear
CONF_SMOOTHING_ALPHA = 0.3  # EMA alpha (lower = smoother)
CLASS_VOTE_FRAMES = 5     # Frames used for class voting
```

## Usage Examples

### Recording Video

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/tidbots_6class/weights/best.pt \
  --camera 0 \
  --conf 0.5 \
  --save-video \
  --output outputs/detection_demo.mp4
```

### High Frame Rate

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/tidbots_6class/weights/best.pt \
  --camera 0 \
  --target-fps 60 \
  --conf 0.5
```

### CPU Execution

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/tidbots_6class/weights/best.pt \
  --camera 0 \
  --device cpu \
  --conf 0.5
```

## Troubleshooting

### Camera won't open

```
Error: Cannot open camera 0
```

- Check if camera is connected: `ls /dev/video*`
- Ensure no other application is using the camera
- Try a different camera ID: `--camera 1`

### FPS doesn't reach target

- Reduce `--imgsz` (e.g., 320, 416)
- Increase `--conf` to reduce detection count
- Verify GPU usage: `--device 0`
- Lower `--target-fps`

### Tracking IDs change frequently

- Lower `--conf` to stabilize detections
- Adjust `--iou` threshold
- Increase hysteresis thresholds (in source code)

### Objects appear slowly

Reduce `APPEAR_THRESHOLD` (default: 3)

## Related Documentation

- [Pipeline Execution Guide](pipeline-e.md)
- [Troubleshooting](troubleshooting-e.md)
