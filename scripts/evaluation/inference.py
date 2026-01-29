#!/usr/bin/env python3
"""
YOLO26 Inference Script
Run inference on images using trained model.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_inference(
    model_path: str,
    source: str,
    output_dir: str,
    conf: float = 0.5,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "0",
    save_images: bool = True,
    save_txt: bool = True,
    save_json: bool = True,
    show: bool = False,
    classes: Optional[List[int]] = None,
    max_det: int = 300,
) -> List[Dict[str, Any]]:
    """
    Run inference on images.

    Args:
        model_path: Path to trained model weights
        source: Source images (file, directory, URL, webcam)
        output_dir: Output directory for results
        conf: Confidence threshold
        iou: IoU threshold for NMS
        imgsz: Input image size
        device: GPU device
        save_images: Save annotated images
        save_txt: Save results in YOLO txt format
        save_json: Save results in JSON format
        show: Display results (requires display)
        classes: Filter by class index
        max_det: Maximum detections per image

    Returns:
        List of detection results
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics package not installed. Run: pip install ultralytics")
        return []

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("YOLO26 Inference")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Source: {source}")
    logger.info(f"Confidence threshold: {conf}")
    logger.info(f"IoU threshold: {iou}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)

    # Run inference
    logger.info("Running inference...")
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=save_images,
        save_txt=save_txt,
        show=show,
        classes=classes,
        max_det=max_det,
        project=str(output_path),
        name="predictions",
        exist_ok=True,
    )

    # Process results
    all_detections = []

    for i, result in enumerate(results):
        image_path = result.path
        image_name = Path(image_path).name if image_path else f"image_{i}"

        detections = {
            "image": image_name,
            "image_path": str(image_path) if image_path else None,
            "detections": [],
        }

        if result.boxes is not None:
            boxes = result.boxes

            for j in range(len(boxes)):
                box = boxes[j]

                detection = {
                    "class_id": int(box.cls.item()),
                    "class_name": result.names[int(box.cls.item())],
                    "confidence": float(box.conf.item()),
                    "bbox": {
                        "x1": float(box.xyxy[0][0].item()),
                        "y1": float(box.xyxy[0][1].item()),
                        "x2": float(box.xyxy[0][2].item()),
                        "y2": float(box.xyxy[0][3].item()),
                    },
                    "bbox_normalized": {
                        "x_center": float(box.xywhn[0][0].item()),
                        "y_center": float(box.xywhn[0][1].item()),
                        "width": float(box.xywhn[0][2].item()),
                        "height": float(box.xywhn[0][3].item()),
                    },
                }

                detections["detections"].append(detection)

        all_detections.append(detections)

        # Log summary
        num_dets = len(detections["detections"])
        if num_dets > 0:
            logger.info(f"  {image_name}: {num_dets} detections")

    # Save JSON results
    if save_json:
        json_path = output_path / "predictions" / "results.json"
        with open(json_path, "w") as f:
            json.dump(all_detections, f, indent=2)
        logger.info(f"Results saved to {json_path}")

    # Summary
    total_images = len(all_detections)
    total_detections = sum(len(d["detections"]) for d in all_detections)
    logger.info("=" * 60)
    logger.info(f"Processed {total_images} images")
    logger.info(f"Total detections: {total_detections}")
    logger.info("=" * 60)

    return all_detections


def run_video_inference(
    model_path: str,
    source: str,
    output_dir: str,
    conf: float = 0.5,
    iou: float = 0.45,
    imgsz: int = 640,
    device: str = "0",
) -> None:
    """
    Run inference on video.

    Args:
        model_path: Path to trained model weights
        source: Video source (file or webcam index)
        output_dir: Output directory
        conf: Confidence threshold
        iou: IoU threshold
        imgsz: Input image size
        device: GPU device
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics package not installed")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)

    logger.info(f"Running video inference on {source}")
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=True,
        project=str(output_path),
        name="video_predictions",
        exist_ok=True,
        stream=True,
    )

    # Process streaming results
    for result in results:
        # Results are yielded frame by frame
        pass

    logger.info(f"Video inference complete. Output saved to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run YOLO26 inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source (image, directory, video, webcam)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/predictions",
        help="Output directory",
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
        "--no-save-images",
        action="store_true",
        help="Don't save annotated images",
    )
    parser.add_argument(
        "--no-save-txt",
        action="store_true",
        help="Don't save txt labels",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=None,
        help="Filter by class indices",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Video inference mode",
    )

    args = parser.parse_args()

    if args.video:
        run_video_inference(
            model_path=args.model,
            source=args.source,
            output_dir=args.output,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
        )
    else:
        run_inference(
            model_path=args.model,
            source=args.source,
            output_dir=args.output,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            save_images=not args.no_save_images,
            save_txt=not args.no_save_txt,
            show=args.show,
            classes=args.classes,
        )


if __name__ == "__main__":
    main()
