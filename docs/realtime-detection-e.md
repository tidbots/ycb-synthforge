# Real-time Object Detection Guide

This document describes how to use real-time object detection and tracking with a webcam.

## Overview

`scripts/evaluation/realtime_detection.py` provides the following features:

- **Object tracking with ByteTrack** - Reduces detection flickering
- **Moving average coordinates** - Smoothed over 1 second of history
- **Trajectory visualization** - Shows movement path over the past 1 second
- **Velocity display** - Shows movement speed in px/sec

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

## Keyboard Controls

| Key | Function |
|-----|----------|
| `q` | Quit |
| `s` | Save screenshot |
| `c` | Toggle confidence display |
| `p` | Toggle coordinate display |
| `t` | Toggle trajectory display |
| `v` | Toggle velocity display |

## Screen Display

```
+------------------------------------------+
| FPS: 30.0 | Detections: 2                |
|                                          |
|   +---------------+                      |
|   | cracker_box   |                      |
|   |    0.85       |                      |
|   +----~~~~-------+  <- Trajectory (yellow)
|   ID:1 (230,250)     <- Smoothed coords (cyan)
|   45 px/s            <- Velocity (green) |
|                                          |
|          +--------+                      |
|          | banana |                      |
|          |  0.72  |                      |
|          +--------+                      |
|          ID:2 (450,280)                  |
|          12 px/s                         |
+------------------------------------------+
```

### Display Elements

| Element | Color | Description |
|---------|-------|-------------|
| Bounding box | Class-dependent | Default YOLO annotation |
| Trajectory | Yellow `(0,255,255)` | Movement path of center point over past 1 second |
| Coordinates | Cyan `(255,255,0)` | Center coordinates smoothed by moving average |
| Velocity | Green `(0,255,0)` | Speed in px/sec calculated from 1-second movement |

## Tracking Features

### ByteTrack

This script uses the ByteTrack algorithm for object tracking.

- Assigns consistent IDs to the same object
- Handles temporary detection failures
- Uses `persist=True` to maintain state across frames

### Moving Average Smoothing

To reduce coordinate flickering, the system maintains 1 second (~30 frames) of history and calculates the moving average.

```
Smoothed coordinates = mean(center coordinates over past 1 second)
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

### High Resolution (may impact performance)

```bash
python scripts/evaluation/realtime_detection.py \
  --model outputs/trained_models/tidbots_6class/weights/best.pt \
  --camera 0 \
  --imgsz 1280 \
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

### Low FPS

- Reduce `--imgsz` (e.g., 320, 416)
- Increase `--conf` to reduce detection count
- Verify GPU usage: `--device 0`

### Tracking IDs change frequently

- Lower `--conf` to stabilize detections
- Adjust `--iou` threshold

## Related Documentation

- [Pipeline Execution Guide](pipeline-e.md)
- [Troubleshooting](troubleshooting-e.md)
