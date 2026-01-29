#!/usr/bin/env python3
"""
Dataset Merger
Merges synthetic and real image datasets for training.
"""

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_dataset_files(dataset_dir: Path, split: str) -> Tuple[List[Path], List[Path]]:
    """
    Get image and label files from a dataset.

    Args:
        dataset_dir: Dataset root directory
        split: Split name ("train" or "val")

    Returns:
        Tuple of (image_paths, label_paths)
    """
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    images = []
    labels = []

    if not images_dir.exists():
        logger.warning(f"Images directory not found: {images_dir}")
        return images, labels

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            label_path = labels_dir / (img_path.stem + ".txt")
            if label_path.exists():
                images.append(img_path)
                labels.append(label_path)
            else:
                logger.warning(f"Label not found for {img_path.name}")

    return images, labels


def merge_datasets(
    synthetic_dir: Path,
    real_dir: Path,
    output_dir: Path,
    real_ratio: float = 0.05,
    seed: int = 42,
) -> None:
    """
    Merge synthetic and real datasets.

    Args:
        synthetic_dir: Path to synthetic dataset
        real_dir: Path to real image dataset
        output_dir: Output directory for merged dataset
        real_ratio: Ratio of real images in final dataset
        seed: Random seed
    """
    random.seed(seed)

    # Create output directories
    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for split in ["train", "val"]:
        logger.info(f"Processing {split} split...")

        # Get synthetic data
        syn_images, syn_labels = get_dataset_files(synthetic_dir, split)
        logger.info(f"  Synthetic: {len(syn_images)} images")

        # Get real data
        real_images, real_labels = get_dataset_files(real_dir, split)
        logger.info(f"  Real: {len(real_images)} images")

        # Calculate how many real images to include
        total_synthetic = len(syn_images)
        if total_synthetic > 0 and len(real_images) > 0:
            # Target: real_ratio of total should be real images
            # total = synthetic + real_to_use
            # real_ratio = real_to_use / total
            # real_to_use = real_ratio * total = real_ratio * (synthetic + real_to_use)
            # real_to_use * (1 - real_ratio) = real_ratio * synthetic
            # real_to_use = real_ratio * synthetic / (1 - real_ratio)
            target_real = int(real_ratio * total_synthetic / (1 - real_ratio))
            real_to_use = min(target_real, len(real_images))
        else:
            real_to_use = len(real_images)

        # Sample real images if needed
        if real_to_use < len(real_images):
            indices = random.sample(range(len(real_images)), real_to_use)
            real_images = [real_images[i] for i in indices]
            real_labels = [real_labels[i] for i in indices]

        logger.info(f"  Using {len(real_images)} real images")

        # Copy synthetic data
        file_counter = 0
        for img_path, label_path in zip(syn_images, syn_labels):
            new_name = f"syn_{file_counter:06d}"
            new_img = output_dir / "images" / split / f"{new_name}{img_path.suffix}"
            new_label = output_dir / "labels" / split / f"{new_name}.txt"

            shutil.copy2(img_path, new_img)
            shutil.copy2(label_path, new_label)
            file_counter += 1

        # Copy real data
        for img_path, label_path in zip(real_images, real_labels):
            new_name = f"real_{file_counter:06d}"
            new_img = output_dir / "images" / split / f"{new_name}{img_path.suffix}"
            new_label = output_dir / "labels" / split / f"{new_name}.txt"

            shutil.copy2(img_path, new_img)
            shutil.copy2(label_path, new_label)
            file_counter += 1

        logger.info(f"  Total: {file_counter} images in {split}")


def create_merged_dataset_yaml(
    output_dir: Path,
    source_yaml: Path,
) -> None:
    """
    Create dataset.yaml for merged dataset.

    Args:
        output_dir: Output directory
        source_yaml: Source dataset.yaml to copy class names from
    """
    # Load source yaml
    with open(source_yaml, "r") as f:
        source_config = yaml.safe_load(f)

    # Create new config
    merged_config = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "nc": source_config.get("nc", 103),
        "names": source_config.get("names", {}),
    }

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(merged_config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Created dataset.yaml at {yaml_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Merge synthetic and real datasets")
    parser.add_argument(
        "--synthetic",
        type=str,
        required=True,
        help="Path to synthetic dataset directory",
    )
    parser.add_argument(
        "--real",
        type=str,
        required=True,
        help="Path to real image dataset directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for merged dataset",
    )
    parser.add_argument(
        "--real_ratio",
        type=float,
        default=0.05,
        help="Ratio of real images in final dataset (default: 0.05)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    synthetic_dir = Path(args.synthetic)
    real_dir = Path(args.real)
    output_dir = Path(args.output)

    # Merge datasets
    merge_datasets(
        synthetic_dir,
        real_dir,
        output_dir,
        real_ratio=args.real_ratio,
        seed=args.seed,
    )

    # Create dataset.yaml
    synthetic_yaml = synthetic_dir / "dataset.yaml"
    if synthetic_yaml.exists():
        create_merged_dataset_yaml(output_dir, synthetic_yaml)
    else:
        logger.warning("Source dataset.yaml not found, skipping yaml creation")

    logger.info("Dataset merge complete!")


if __name__ == "__main__":
    main()
