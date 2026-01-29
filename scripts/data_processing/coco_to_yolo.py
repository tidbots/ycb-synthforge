#!/usr/bin/env python3
"""
COCO to YOLO Format Converter
Converts COCO format annotations to YOLO format and splits into train/val/test sets.
"""

import argparse
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_coco_annotations(coco_json_path: str) -> Dict[str, Any]:
    """
    Load COCO format annotations.

    Args:
        coco_json_path: Path to COCO annotations JSON file

    Returns:
        COCO annotations dictionary
    """
    logger.info(f"Loading COCO annotations from {coco_json_path}")
    with open(coco_json_path, "r") as f:
        return json.load(f)


def coco_to_yolo_bbox(
    bbox: List[float],
    img_width: int,
    img_height: int,
) -> Tuple[float, float, float, float]:
    """
    Convert COCO bbox [x, y, width, height] to YOLO format [x_center, y_center, width, height].
    All values are normalized to [0, 1].

    Args:
        bbox: COCO format bbox [x, y, width, height]
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        YOLO format bbox (x_center, y_center, width, height) normalized
    """
    x, y, w, h = bbox

    # Calculate center coordinates
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height

    # Normalize width and height
    width = w / img_width
    height = h / img_height

    # Clip to [0, 1]
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))

    return x_center, y_center, width, height


def convert_annotations(
    coco_data: Dict[str, Any],
    output_dir: Path,
    images_dir: Path,
    split: str,
    image_ids: List[int],
) -> int:
    """
    Convert COCO annotations to YOLO format for specified images.

    Args:
        coco_data: COCO annotations dictionary
        output_dir: Output directory for YOLO labels
        images_dir: Source images directory
        split: Split name ("train" or "val")
        image_ids: List of image IDs to process

    Returns:
        Number of images processed
    """
    # Create output directories
    images_output = output_dir / "images" / split
    labels_output = output_dir / "labels" / split
    images_output.mkdir(parents=True, exist_ok=True)
    labels_output.mkdir(parents=True, exist_ok=True)

    # Build image ID to info mapping
    image_info = {img["id"]: img for img in coco_data["images"]}

    # Build image ID to annotations mapping
    image_annotations: Dict[int, List[Dict]] = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if img_id not in image_annotations:
            image_annotations[img_id] = []
        image_annotations[img_id].append(ann)

    processed = 0

    for img_id in image_ids:
        if img_id not in image_info:
            logger.warning(f"Image ID {img_id} not found in annotations")
            continue

        img_data = image_info[img_id]
        filename = img_data["file_name"]
        img_width = img_data["width"]
        img_height = img_data["height"]

        # Source image path
        src_image = images_dir / filename
        if not src_image.exists():
            logger.warning(f"Image file not found: {src_image}")
            continue

        # Copy image to output directory
        dst_image = images_output / filename
        shutil.copy2(src_image, dst_image)

        # Create YOLO label file
        label_filename = Path(filename).stem + ".txt"
        label_path = labels_output / label_filename

        # Get annotations for this image
        annotations = image_annotations.get(img_id, [])

        # Write YOLO format labels
        with open(label_path, "w") as f:
            for ann in annotations:
                category_id = ann["category_id"]
                bbox = ann["bbox"]

                # Skip invalid bboxes
                if bbox[2] <= 0 or bbox[3] <= 0:
                    continue

                # Convert to YOLO format
                x_center, y_center, width, height = coco_to_yolo_bbox(
                    bbox, img_width, img_height
                )

                # Write line: class_id x_center y_center width height
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        processed += 1

    return processed


def create_dataset_yaml(
    output_dir: Path,
    categories: List[Dict[str, Any]],
    num_classes: int,
    has_test: bool = False,
) -> None:
    """
    Create dataset.yaml file for YOLO training.

    Args:
        output_dir: Output directory
        categories: List of category dictionaries
        num_classes: Number of classes
        has_test: Whether test split exists
    """
    # Build class names dictionary
    names = {}
    for cat in categories:
        names[cat["id"]] = cat["name"]

    # Ensure all IDs from 0 to num_classes-1 are present
    for i in range(num_classes):
        if i not in names:
            names[i] = f"class_{i}"

    dataset_config = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": num_classes,
        "names": names,
    }

    if has_test:
        dataset_config["test"] = "images/test"

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Created dataset.yaml at {yaml_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert COCO annotations to YOLO format"
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        required=True,
        help="Path to COCO annotations JSON file",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Path to images directory (default: same as annotations)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for YOLO dataset",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.833,
        help="Ratio of training data (default: 0.833 for 10k/12k)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.083,
        help="Ratio of validation data (default: 0.083 for 1k/12k)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.083,
        help="Ratio of test data (default: 0.083 for 1k/12k, set to 0 to skip)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Load COCO annotations
    coco_data = load_coco_annotations(args.coco_json)

    # Determine images directory
    coco_json_path = Path(args.coco_json)
    if args.images_dir:
        images_dir = Path(args.images_dir)
    else:
        images_dir = coco_json_path.parent / "images"

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image IDs
    all_image_ids = [img["id"] for img in coco_data["images"]]
    random.shuffle(all_image_ids)

    # Normalize ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    train_ratio = args.train_ratio / total_ratio
    val_ratio = args.val_ratio / total_ratio
    test_ratio = args.test_ratio / total_ratio

    # Split into train/val/test
    n_total = len(all_image_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = all_image_ids[:n_train]
    val_ids = all_image_ids[n_train:n_train + n_val]
    test_ids = all_image_ids[n_train + n_val:] if test_ratio > 0 else []

    logger.info(f"Total images: {n_total}")
    logger.info(f"Training images: {len(train_ids)}")
    logger.info(f"Validation images: {len(val_ids)}")
    if test_ids:
        logger.info(f"Test images: {len(test_ids)}")

    # Convert training data
    logger.info("Converting training data...")
    train_processed = convert_annotations(
        coco_data, output_dir, images_dir, "train", train_ids
    )
    logger.info(f"Processed {train_processed} training images")

    # Convert validation data
    logger.info("Converting validation data...")
    val_processed = convert_annotations(
        coco_data, output_dir, images_dir, "val", val_ids
    )
    logger.info(f"Processed {val_processed} validation images")

    # Convert test data
    has_test = False
    if test_ids:
        logger.info("Converting test data...")
        test_processed = convert_annotations(
            coco_data, output_dir, images_dir, "test", test_ids
        )
        logger.info(f"Processed {test_processed} test images")
        has_test = True

    # Determine number of classes
    if coco_data.get("categories"):
        num_classes = max(cat["id"] for cat in coco_data["categories"]) + 1
    else:
        # Infer from annotations
        all_category_ids = set()
        for ann in coco_data["annotations"]:
            all_category_ids.add(ann["category_id"])
        num_classes = max(all_category_ids) + 1 if all_category_ids else 1

    # Create dataset.yaml
    create_dataset_yaml(
        output_dir,
        coco_data.get("categories", []),
        num_classes,
        has_test=has_test,
    )

    logger.info("Conversion complete!")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
