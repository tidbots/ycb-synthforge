#!/bin/bash
# Real-time YCB Object Detection (Host execution)
# ホストで直接実行する場合

set -e

MODEL_PATH="${1:-outputs/trained_models/ycb_yolo26_run/weights/best.pt}"
CAMERA_ID="${2:-0}"

echo "=============================================="
echo "YCB Real-time Object Detection (Host)"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Camera: /dev/video$CAMERA_ID"
echo "=============================================="

python3 scripts/evaluation/realtime_detection.py \
    --model "$MODEL_PATH" \
    --camera "$CAMERA_ID" \
    --conf 0.5 \
    --device 0
