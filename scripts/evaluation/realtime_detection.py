#!/usr/bin/env python3
"""
Real-time YCB Object Detection with Webcam
USB Webカメラを使用したリアルタイム物体検出

Features:
- ByteTrack object tracking
- Coordinate smoothing (moving average)
- Confidence smoothing (exponential moving average)
- Class stabilization (voting-based)
- Hysteresis (appear/disappear thresholds)
- Trajectory and velocity display
- Adaptive frame rate targeting 30Hz
"""

import argparse
import cv2
import time
import numpy as np
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class TrackState:
    """State for each tracked object."""
    # Position history: (cx, cy, timestamp)
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    # Class voting history
    class_votes: list = field(default_factory=list)
    fixed_class: Optional[int] = None
    # Confidence smoothing
    smoothed_conf: float = 0.0
    # Hysteresis counters
    appear_count: int = 0
    disappear_count: int = 0
    is_visible: bool = False
    # Last seen bbox for drawing during disappear grace period
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    last_conf: float = 0.0


def run_realtime_detection(
    model_path: str,
    camera_id: int = 0,
    conf: float = 0.5,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "0",
    save_video: bool = False,
    output_path: str = "outputs/realtime_detection.mp4",
    target_fps: float = 30.0,
):
    """
    Run real-time object detection using webcam.

    Args:
        model_path: Path to trained model weights
        camera_id: Camera device ID (default: 0)
        conf: Confidence threshold
        iou: IoU threshold for NMS
        imgsz: Input image size
        device: GPU device
        save_video: Save output video
        output_path: Output video path
        target_fps: Target processing rate (default: 30.0)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed")
        print("Run: pip install ultralytics")
        return

    # Stabilization parameters
    APPEAR_THRESHOLD = 3      # Frames to confirm appearance
    DISAPPEAR_THRESHOLD = 5   # Frames to confirm disappearance
    CONF_SMOOTHING_ALPHA = 0.3  # EMA alpha (lower = smoother)
    CLASS_VOTE_FRAMES = 5     # Frames for class voting

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    class_names = model.names

    # Open camera
    print(f"Opening camera: /dev/video{camera_id}")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"Camera resolution: {width}x{height} @ {camera_fps:.1f} FPS")
    print(f"Target processing rate: {target_fps:.1f} Hz")

    # Video writer
    writer = None
    if save_video:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
        print(f"Saving video to: {output_path}")

    print("\n" + "=" * 50)
    print("Real-time YCB Object Detection (Stabilized)")
    print("=" * 50)
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    print("Press 'c' to toggle confidence display")
    print("Press 'p' to toggle coordinate display")
    print("Press 't' to toggle trajectory display")
    print("Press 'v' to toggle velocity display")
    print("Press '+'/'-' to adjust target FPS")
    print("=" * 50 + "\n")

    show_conf = True
    show_coords = True
    show_trajectory = True
    show_velocity = True

    # Track states
    track_states: Dict[int, TrackState] = defaultdict(TrackState)
    history_length = int(target_fps)  # 1 second of history

    frame_count = 0
    start_time = time.time()
    screenshot_count = 0
    last_frame_time = time.time()
    frame_interval = 1.0 / target_fps

    # For actual FPS calculation
    fps_history = deque(maxlen=30)

    try:
        while True:
            loop_start = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            # Run inference with tracking (use lower conf for tracking, filter later)
            results = model.track(
                source=frame,
                conf=conf * 0.5,  # Lower threshold for tracking stability
                iou=iou,
                imgsz=imgsz,
                device=device,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            # Process detections
            current_ids = set()
            detections = []  # (track_id, bbox, conf, cls_id)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes
                for i, box in enumerate(boxes):
                    track_id = int(box.id[i].item())
                    x1, y1, x2, y2 = box.xyxy[i].tolist()
                    conf_val = box.conf[i].item()
                    cls_id = int(box.cls[i].item())

                    current_ids.add(track_id)
                    detections.append((track_id, (x1, y1, x2, y2), conf_val, cls_id))

            # Update track states
            for track_id, bbox, conf_val, cls_id in detections:
                state = track_states[track_id]
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                # Update position history
                state.position_history.append((cx, cy, time.time()))

                # Update last bbox
                state.last_bbox = bbox
                state.last_conf = conf_val

                # Confidence smoothing (EMA)
                if state.smoothed_conf == 0:
                    state.smoothed_conf = conf_val
                else:
                    state.smoothed_conf = (
                        CONF_SMOOTHING_ALPHA * conf_val +
                        (1 - CONF_SMOOTHING_ALPHA) * state.smoothed_conf
                    )

                # Class voting
                if state.fixed_class is None:
                    state.class_votes.append(cls_id)
                    if len(state.class_votes) >= CLASS_VOTE_FRAMES:
                        # Majority vote
                        vote_counter = Counter(state.class_votes)
                        state.fixed_class = vote_counter.most_common(1)[0][0]

                # Hysteresis: appearance
                state.disappear_count = 0
                if not state.is_visible:
                    state.appear_count += 1
                    if state.appear_count >= APPEAR_THRESHOLD:
                        state.is_visible = True

            # Handle disappeared tracks
            for track_id in list(track_states.keys()):
                if track_id not in current_ids:
                    state = track_states[track_id]
                    state.appear_count = 0
                    state.disappear_count += 1

                    # Hysteresis: disappearance
                    if state.disappear_count >= DISAPPEAR_THRESHOLD:
                        state.is_visible = False
                        # Clean up after grace period
                        if state.disappear_count >= DISAPPEAR_THRESHOLD * 2:
                            del track_states[track_id]

            # Draw frame
            annotated_frame = frame.copy()

            # Draw visible tracks
            for track_id, state in track_states.items():
                if not state.is_visible or state.last_bbox is None:
                    continue

                # Apply confidence threshold to smoothed confidence
                if state.smoothed_conf < conf:
                    continue

                x1, y1, x2, y2 = state.last_bbox
                cls_id = state.fixed_class if state.fixed_class is not None else -1
                cls_name = class_names.get(cls_id, "unknown") if cls_id >= 0 else "unknown"

                # Get color based on class
                color = get_color(track_id)

                # Draw bounding box
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                )

                # Draw label with smoothed confidence
                if show_conf:
                    label = f"{cls_name} {state.smoothed_conf:.2f}"
                else:
                    label = cls_name

                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    annotated_frame,
                    (int(x1), int(y1) - label_size[1] - 10),
                    (int(x1) + label_size[0], int(y1)),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                # Draw trajectory
                if show_trajectory and len(state.position_history) > 1:
                    points = np.array(
                        [(int(p[0]), int(p[1])) for p in state.position_history],
                        dtype=np.int32,
                    )
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(0, 255, 255),
                        thickness=2,
                    )

                # Calculate smoothed coordinates
                history = state.position_history
                if len(history) > 0:
                    avg_cx = np.mean([p[0] for p in history])
                    avg_cy = np.mean([p[1] for p in history])
                else:
                    avg_cx, avg_cy = (x1 + x2) / 2, (y1 + y2) / 2

                # Calculate velocity
                velocity = 0.0
                if len(history) >= 2:
                    p_old = history[0]
                    p_new = history[-1]
                    dt = p_new[2] - p_old[2]
                    if dt > 0:
                        dx = p_new[0] - p_old[0]
                        dy = p_new[1] - p_old[1]
                        velocity = np.sqrt(dx**2 + dy**2) / dt

                # Draw coordinate overlay
                text_y = int(y2) + 20
                if show_coords:
                    coord_text = f"ID:{track_id} ({avg_cx:.0f},{avg_cy:.0f})"
                    cv2.putText(
                        annotated_frame,
                        coord_text,
                        (int(x1), text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
                    text_y += 20

                # Draw velocity
                if show_velocity:
                    vel_text = f"{velocity:.0f} px/s"
                    cv2.putText(
                        annotated_frame,
                        vel_text,
                        (int(x1), text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            # Calculate actual FPS
            frame_count += 1
            current_time = time.time()
            frame_duration = current_time - last_frame_time
            last_frame_time = current_time

            if frame_duration > 0:
                fps_history.append(1.0 / frame_duration)
            actual_fps = np.mean(fps_history) if fps_history else 0

            # Draw info
            visible_count = sum(1 for s in track_states.values() if s.is_visible and s.smoothed_conf >= conf)
            info_text = f"FPS: {actual_fps:.1f}/{target_fps:.0f} | Visible: {visible_count}"
            cv2.putText(
                annotated_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            # Display
            cv2.imshow("YCB Object Detection", annotated_frame)

            # Save video frame
            if writer:
                writer.write(annotated_frame)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("s"):
                screenshot_path = f"outputs/screenshot_{screenshot_count:04d}.jpg"
                Path(screenshot_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(screenshot_path, annotated_frame)
                print(f"Screenshot saved: {screenshot_path}")
                screenshot_count += 1
            elif key == ord("c"):
                show_conf = not show_conf
                print(f"Confidence display: {'ON' if show_conf else 'OFF'}")
            elif key == ord("p"):
                show_coords = not show_coords
                print(f"Coordinate display: {'ON' if show_coords else 'OFF'}")
            elif key == ord("t"):
                show_trajectory = not show_trajectory
                print(f"Trajectory display: {'ON' if show_trajectory else 'OFF'}")
            elif key == ord("v"):
                show_velocity = not show_velocity
                print(f"Velocity display: {'ON' if show_velocity else 'OFF'}")
            elif key == ord("+") or key == ord("="):
                target_fps = min(60, target_fps + 5)
                frame_interval = 1.0 / target_fps
                print(f"Target FPS: {target_fps:.0f}")
            elif key == ord("-"):
                target_fps = max(5, target_fps - 5)
                frame_interval = 1.0 / target_fps
                print(f"Target FPS: {target_fps:.0f}")

            # Frame rate control
            loop_duration = time.time() - loop_start
            sleep_time = frame_interval - loop_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        # Print summary
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print("\n" + "=" * 50)
        print("Summary")
        print("=" * 50)
        print(f"Total frames: {frame_count}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Average FPS: {avg_fps:.1f}")
        print("=" * 50)


def get_color(track_id: int) -> Tuple[int, int, int]:
    """Get consistent color for track ID."""
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 255),  # Orange
        (255, 128, 0),  # Light blue
        (0, 128, 255),  # Orange-red
        (128, 255, 0),  # Light green
    ]
    return colors[track_id % len(colors)]


def main():
    parser = argparse.ArgumentParser(
        description="Real-time YCB object detection with webcam (stabilized)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/trained_models/ycb_yolo26_run/weights/best.pt",
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold (default: 0.45)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device (default: 0)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save output video",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/realtime_detection.mp4",
        help="Output video path",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="Target processing rate in Hz (default: 30.0)",
    )

    args = parser.parse_args()

    run_realtime_detection(
        model_path=args.model,
        camera_id=args.camera,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save_video=args.save_video,
        output_path=args.output,
        target_fps=args.target_fps,
    )


if __name__ == "__main__":
    main()
