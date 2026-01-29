#!/usr/bin/env python3
"""
YOLO26 Training Script
Fine-tunes YOLO26 on YCB synthetic dataset.
"""

import argparse
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


def load_training_config(config_path: str) -> Dict[str, Any]:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_yolo26(
    data_yaml: str,
    weights: str,
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    project: str = "outputs/trained_models",
    name: str = "ycb_yolo26",
    device: str = "0",
    workers: int = 8,
    freeze: Optional[int] = None,
    lr0: float = 0.01,
    optimizer: str = "auto",
    patience: int = 50,
    save_period: int = 10,
    resume: bool = False,
    augment_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Train YOLO26 model.

    Args:
        data_yaml: Path to dataset YAML file
        weights: Path to pretrained weights
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Input image size
        project: Project directory for outputs
        name: Experiment name
        device: GPU device(s) to use
        workers: Number of dataloader workers
        freeze: Number of layers to freeze (for fine-tuning)
        lr0: Initial learning rate
        optimizer: Optimizer (auto, SGD, Adam, etc.)
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
        resume: Resume from last checkpoint
        augment_config: Augmentation configuration
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics package not installed. Run: pip install ultralytics")
        return

    logger.info("=" * 60)
    logger.info("YOLO26 Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Weights: {weights}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch}")
    logger.info(f"Image size: {imgsz}")
    logger.info(f"Freeze layers: {freeze}")
    logger.info(f"Learning rate: {lr0}")
    logger.info("=" * 60)

    # Load model
    logger.info(f"Loading model from {weights}")
    model = YOLO(weights)

    # Prepare training arguments
    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "project": project,
        "name": name,
        "device": device,
        "workers": workers,
        "patience": patience,
        "save_period": save_period,
        "exist_ok": True,
        "pretrained": True,
        "lr0": lr0,
        "verbose": True,
    }

    # Add freeze if specified
    if freeze is not None and freeze > 0:
        train_args["freeze"] = freeze
        logger.info(f"Freezing first {freeze} layers")

    # Add optimizer if not auto
    if optimizer != "auto":
        train_args["optimizer"] = optimizer

    # Add resume flag
    if resume:
        train_args["resume"] = True

    # Add augmentation settings if provided
    if augment_config:
        for key, value in augment_config.items():
            train_args[key] = value

    # Start training
    logger.info("Starting training...")
    results = model.train(**train_args)

    logger.info("Training complete!")
    logger.info(f"Results saved to: {project}/{name}")

    # Log final metrics
    if hasattr(results, "results_dict"):
        logger.info("Final metrics:")
        for key, value in results.results_dict.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train YOLO26 on YCB dataset")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo26m.pt",
        help="Path to pretrained weights (default: yolo26m.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size (default: 640)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="outputs/trained_models",
        help="Project directory (default: outputs/trained_models)",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (default: auto-generated)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="GPU device (default: 0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of dataloader workers (default: 8)",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Number of layers to freeze for fine-tuning",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        help="Optimizer (auto, SGD, Adam, AdamW)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (default: 50)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (overrides other args)",
    )

    args = parser.parse_args()

    # Generate default name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"ycb_yolo26_{timestamp}"

    # Load config file if provided
    augment_config = None
    if args.config:
        config = load_training_config(args.config)

        # Override args with config values
        for key, value in config.items():
            if key == "augmentation":
                augment_config = value
            elif hasattr(args, key) and getattr(args, key) is None:
                setattr(args, key, value)

    # Run training
    train_yolo26(
        data_yaml=args.data,
        weights=args.weights,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        freeze=args.freeze,
        lr0=args.lr0,
        optimizer=args.optimizer,
        patience=args.patience,
        resume=args.resume,
        augment_config=augment_config,
    )


if __name__ == "__main__":
    main()
