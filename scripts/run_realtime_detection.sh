#!/bin/bash
# Real-time YCB Object Detection Script
# USB Webカメラを使用したリアルタイム物体検出

set -e

# Default values
MODEL_PATH="outputs/trained_models/ycb_yolo26_run/weights/best.pt"
CAMERA_ID=0
CONF=0.5
DEVICE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --camera)
            CAMERA_ID="$2"
            shift 2
            ;;
        --conf)
            CONF="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --save-video)
            SAVE_VIDEO="--save-video"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH    Model weights path (default: outputs/trained_models/ycb_yolo26_run/weights/best.pt)"
            echo "  --camera ID     Camera device ID (default: 0)"
            echo "  --conf VALUE    Confidence threshold (default: 0.5)"
            echo "  --device ID     GPU device (default: 0)"
            echo "  --save-video    Save output video"
            echo "  --help          Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "YCB Real-time Object Detection"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Camera: /dev/video$CAMERA_ID"
echo "Confidence: $CONF"
echo "GPU Device: $DEVICE"
echo "=============================================="

# Run with Docker (GUI support)
docker run --rm \
    --gpus all \
    -e DISPLAY=$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority:ro \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/scripts:/workspace/scripts:ro \
    --device /dev/video$CAMERA_ID:/dev/video0 \
    --shm-size=8gb \
    ycb_synthforge-yolo26_inference \
    python3 /workspace/scripts/evaluation/realtime_detection.py \
    --model /workspace/$MODEL_PATH \
    --camera 0 \
    --conf $CONF \
    --device $DEVICE \
    $SAVE_VIDEO
