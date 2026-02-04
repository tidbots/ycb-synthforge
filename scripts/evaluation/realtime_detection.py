#!/usr/bin/env python3
"""
Real-time YCB Object Detection with Webcam
USB Webカメラを使用したリアルタイム物体検出
"""

import argparse
import cv2
import time
import numpy as np
from collections import defaultdict, deque
from pathlib import Path


def run_realtime_detection(
    model_path: str,
    camera_id: int = 0,
    conf: float = 0.5,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "0",
    save_video: bool = False,
    output_path: str = "outputs/realtime_detection.mp4",
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
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics package not installed")
        print("Run: pip install ultralytics")
        return

    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)

    # Open camera
    print(f"Opening camera: /dev/video{camera_id}")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Cannot open camera {camera_id}")
        return

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"Camera resolution: {width}x{height} @ {fps:.1f} FPS")

    # Video writer
    writer = None
    if save_video:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving video to: {output_path}")

    print("\n" + "=" * 50)
    print("Real-time YCB Object Detection (with Tracking)")
    print("=" * 50)
    print("Press 'q' to quit")
    print("Press 's' to save screenshot")
    print("Press 'c' to toggle confidence display")
    print("Press 'p' to toggle coordinate display")
    print("Press 't' to toggle trajectory display")
    print("Press 'v' to toggle velocity display")
    print("=" * 50 + "\n")

    show_conf = True
    show_coords = True
    show_trajectory = True
    show_velocity = True

    # Track history for smoothing, trajectory, and velocity (1 second = ~30 frames)
    history_length = int(fps)  # 1 second of history
    track_history = defaultdict(lambda: deque(maxlen=history_length))
    frame_count = 0
    start_time = time.time()
    screenshot_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            # Run inference with tracking
            results = model.track(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )

            # Draw results
            annotated_frame = results[0].plot(
                conf=show_conf,
                line_width=2,
                font_size=0.6,
            )

            # Process detections: smoothing, trajectory, velocity
            if results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    track_id = int(box.id[0].item()) if box.id is not None else -1

                    if track_id == -1:
                        continue

                    # Calculate center point
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

                    # Store in history
                    track_history[track_id].append((cx, cy, time.time()))

                    # Get smoothed coordinates (moving average)
                    history = track_history[track_id]
                    if len(history) > 0:
                        avg_cx = np.mean([p[0] for p in history])
                        avg_cy = np.mean([p[1] for p in history])
                    else:
                        avg_cx, avg_cy = cx, cy

                    # Draw trajectory (F)
                    if show_trajectory and len(history) > 1:
                        points = np.array([(int(p[0]), int(p[1])) for p in history], dtype=np.int32)
                        cv2.polylines(
                            annotated_frame,
                            [points],
                            isClosed=False,
                            color=(0, 255, 255),  # Yellow
                            thickness=2,
                        )

                    # Calculate velocity (E)
                    velocity = 0.0
                    if len(history) >= 2:
                        # Use first and last points for velocity calculation
                        p_old = history[0]
                        p_new = history[-1]
                        dt = p_new[2] - p_old[2]
                        if dt > 0:
                            dx = p_new[0] - p_old[0]
                            dy = p_new[1] - p_old[1]
                            distance = np.sqrt(dx**2 + dy**2)
                            velocity = distance / dt  # pixels per second

                    # Draw coordinate overlay (D - smoothed)
                    if show_coords:
                        coord_text = f"ID:{track_id} ({avg_cx:.0f},{avg_cy:.0f})"
                        text_y = int(y2) + 20
                        cv2.putText(
                            annotated_frame,
                            coord_text,
                            (int(x1), text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),  # Cyan
                            1,
                        )

                    # Draw velocity (E)
                    if show_velocity:
                        vel_text = f"{velocity:.0f} px/s"
                        text_y = int(y2) + 40 if show_coords else int(y2) + 20
                        cv2.putText(
                            annotated_frame,
                            vel_text,
                            (int(x1), text_y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),  # Green
                            1,
                        )

            # Clean up old tracks (not seen in current frame)
            current_ids = set()
            if results[0].boxes is not None and results[0].boxes.id is not None:
                current_ids = set(int(id.item()) for id in results[0].boxes.id)
            stale_ids = [tid for tid in track_history.keys() if tid not in current_ids]
            for tid in stale_ids:
                del track_history[tid]

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            # Draw FPS and detection count
            num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
            info_text = f"FPS: {current_fps:.1f} | Detections: {num_detections}"
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


def main():
    parser = argparse.ArgumentParser(
        description="Real-time YCB object detection with webcam"
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
    )


if __name__ == "__main__":
    main()
