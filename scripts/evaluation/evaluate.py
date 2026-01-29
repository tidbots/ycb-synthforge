#!/usr/bin/env python3
"""
YOLO26 Model Evaluation Script
Evaluates trained model on validation/test data.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_model(
    model_path: str,
    data_yaml: str,
    output_dir: str,
    batch: int = 16,
    imgsz: int = 640,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = "0",
    workers: int = 8,
    split: str = "val",
    save_json: bool = True,
    plots: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate YOLO26 model.

    Args:
        model_path: Path to trained model weights
        data_yaml: Path to dataset YAML file
        output_dir: Output directory for results
        batch: Batch size
        imgsz: Input image size
        conf: Confidence threshold for NMS
        iou: IoU threshold for NMS
        device: GPU device
        workers: Number of dataloader workers
        split: Dataset split to evaluate on
        save_json: Save results in JSON format
        plots: Generate plots

    Returns:
        Dictionary of evaluation metrics
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics package not installed. Run: pip install ultralytics")
        return {}

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("YOLO26 Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Model: {model_path}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Split: {split}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = YOLO(model_path)

    # Run validation
    logger.info("Running evaluation...")
    results = model.val(
        data=data_yaml,
        batch=batch,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        workers=workers,
        split=split,
        plots=plots,
        save_json=save_json,
        project=str(output_path),
        name="eval",
        exist_ok=True,
    )

    # Extract metrics
    metrics = {}

    if hasattr(results, "box"):
        box_metrics = results.box
        metrics["mAP50"] = float(box_metrics.map50)
        metrics["mAP50-95"] = float(box_metrics.map)
        metrics["precision"] = float(box_metrics.mp)
        metrics["recall"] = float(box_metrics.mr)

        # Per-class metrics
        if hasattr(box_metrics, "ap_class_index") and hasattr(box_metrics, "ap"):
            metrics["per_class_ap50"] = {}
            for i, class_idx in enumerate(box_metrics.ap_class_index):
                if hasattr(box_metrics, "names") and class_idx in box_metrics.names:
                    class_name = box_metrics.names[class_idx]
                else:
                    class_name = f"class_{class_idx}"
                metrics["per_class_ap50"][class_name] = float(box_metrics.ap50[i])

    # Log metrics
    logger.info("=" * 60)
    logger.info("Evaluation Results:")
    logger.info("=" * 60)
    logger.info(f"  mAP@0.5: {metrics.get('mAP50', 'N/A'):.4f}")
    logger.info(f"  mAP@0.5:0.95: {metrics.get('mAP50-95', 'N/A'):.4f}")
    logger.info(f"  Precision: {metrics.get('precision', 'N/A'):.4f}")
    logger.info(f"  Recall: {metrics.get('recall', 'N/A'):.4f}")
    logger.info("=" * 60)

    # Save metrics to JSON
    if save_json:
        metrics_path = output_path / "eval" / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")

    return metrics


def compare_models(
    model_paths: list,
    data_yaml: str,
    output_dir: str,
    **kwargs,
) -> None:
    """
    Compare multiple models.

    Args:
        model_paths: List of model paths
        data_yaml: Path to dataset YAML
        output_dir: Output directory
        **kwargs: Additional arguments for evaluate_model
    """
    results = {}

    for model_path in model_paths:
        model_name = Path(model_path).stem
        logger.info(f"\nEvaluating: {model_name}")

        metrics = evaluate_model(
            model_path=model_path,
            data_yaml=data_yaml,
            output_dir=os.path.join(output_dir, model_name),
            **kwargs,
        )
        results[model_name] = metrics

    # Print comparison table
    logger.info("\n" + "=" * 80)
    logger.info("Model Comparison")
    logger.info("=" * 80)
    logger.info(f"{'Model':<30} {'mAP50':<12} {'mAP50-95':<12} {'Precision':<12} {'Recall':<12}")
    logger.info("-" * 80)

    for model_name, metrics in results.items():
        logger.info(
            f"{model_name:<30} "
            f"{metrics.get('mAP50', 0):<12.4f} "
            f"{metrics.get('mAP50-95', 0):<12.4f} "
            f"{metrics.get('precision', 0):<12.4f} "
            f"{metrics.get('recall', 0):<12.4f}"
        )

    logger.info("=" * 80)

    # Save comparison
    comparison_path = Path(output_dir) / "model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Comparison saved to {comparison_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate YOLO26 model")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/metrics",
        help="Output directory for results",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.6,
        help="IoU threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation",
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        output_dir=args.output,
        batch=args.batch,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        split=args.split,
        plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
