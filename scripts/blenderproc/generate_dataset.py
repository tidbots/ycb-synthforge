#!/usr/bin/env python3
# BlenderProc must be imported first before any other imports
import blenderproc as bproc

"""
YCB Object Dataset Generator using BlenderProc
Generates photorealistic synthetic data with domain randomization for YOLO training.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from camera import CameraRandomizer
from lighting import LightingRandomizer
from materials import MaterialRandomizer
from scene_setup import SceneSetup
from ycb_classes import YCB_CLASSES, YCB_NAME_TO_ID, get_material_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


# Objects to completely exclude due to poor mesh quality in both formats
EXCLUDED_OBJECTS = {
    "072-b_toy_airplane",
    "072-c_toy_airplane",
    "072-d_toy_airplane",
    "072-e_toy_airplane",
    "072-h_toy_airplane",
    "072-k_toy_airplane",
}

# Objects where tsdf format is preferred over google_16k
# (google_16k has mesh/texture issues for these objects)
USE_TSDF_FORMAT = {
    "001_chips_can",
    "041_small_marker",
    "049_small_clamp",
    "058_golf_ball",
    "062_dice",
    "073-g_lego_duplo",
    "073-h_lego_duplo",
    "073-i_lego_duplo",
    "073-j_lego_duplo",
    "073-k_lego_duplo",
    "073-l_lego_duplo",
    "073-m_lego_duplo",
    "076_timer",
}


class ModelSourceManager:
    """Manages multiple model sources and class ID assignments."""

    def __init__(self):
        self.model_paths: Dict[str, str] = {}  # object_name -> file_path
        self.class_mapping: Dict[str, int] = {}  # object_name -> class_id
        self.id_to_name: Dict[int, str] = {}  # class_id -> object_name
        self.source_info: Dict[str, str] = {}  # object_name -> source_name
        self._next_class_id = 0

    def load_source(
        self,
        source_name: str,
        source_path: str,
        include_objects: Optional[List[str]] = None,
        use_ycb_classes: bool = False,
    ) -> int:
        """
        Load models from a source directory.

        Args:
            source_name: Name of the source (e.g., "ycb", "tidbots")
            source_path: Path to the source directory
            include_objects: Optional list of objects to include
            use_ycb_classes: If True, use predefined YCB class IDs

        Returns:
            Number of models loaded from this source
        """
        source_dir = Path(source_path)
        if not source_dir.exists():
            logger.warning(f"Source directory not found: {source_path}")
            return 0

        # Get available objects in this source
        available_objects = []
        for obj_dir in sorted(source_dir.iterdir()):
            if obj_dir.is_dir() and not obj_dir.name.startswith('.'):
                available_objects.append(obj_dir.name)

        # Filter by include list if specified
        if include_objects:
            target_objects = [o for o in include_objects if o in available_objects]
            if len(target_objects) < len(include_objects):
                missing = set(include_objects) - set(available_objects)
                logger.warning(f"Objects not found in {source_name}: {missing}")
        else:
            target_objects = available_objects

        loaded_count = 0
        for obj_name in target_objects:
            # Skip excluded objects
            if obj_name in EXCLUDED_OBJECTS:
                logger.info(f"Excluding {obj_name}: poor mesh quality")
                continue

            # Find model file
            model_file = self._find_model_file(source_dir / obj_name)
            if model_file is None:
                logger.warning(f"No model file found for {obj_name}")
                continue

            # Assign class ID
            if use_ycb_classes and obj_name in YCB_NAME_TO_ID:
                class_id = YCB_NAME_TO_ID[obj_name]
                # Update next_class_id to avoid conflicts
                self._next_class_id = max(self._next_class_id, class_id + 1)
            else:
                class_id = self._next_class_id
                self._next_class_id += 1

            # Register the model
            self.model_paths[obj_name] = str(model_file)
            self.class_mapping[obj_name] = class_id
            self.id_to_name[class_id] = obj_name
            self.source_info[obj_name] = source_name
            loaded_count += 1

            logger.debug(f"Loaded {obj_name} (class_id={class_id}) from {source_name}")

        logger.info(f"Loaded {loaded_count} models from {source_name}")
        return loaded_count

    def _find_model_file(self, obj_dir: Path) -> Optional[str]:
        """Find the best model file for an object."""
        obj_name = obj_dir.name

        # Check for YCB-style subdirectories
        google_path = obj_dir / "google_16k" / "textured.obj"
        tsdf_path = obj_dir / "tsdf" / "textured.obj"

        # For objects where tsdf is preferred
        if obj_name in USE_TSDF_FORMAT:
            if tsdf_path.exists():
                logger.debug(f"Using tsdf format for {obj_name}")
                return str(tsdf_path)
            elif google_path.exists():
                return str(google_path)

        # Prefer google_16k, fallback to tsdf
        if google_path.exists():
            return str(google_path)
        elif tsdf_path.exists():
            logger.debug(f"Using tsdf format for {obj_name}")
            return str(tsdf_path)

        # Check for direct OBJ file
        direct_obj = obj_dir / "textured.obj"
        if direct_obj.exists():
            return str(direct_obj)

        # Check for any OBJ file
        obj_files = list(obj_dir.glob("*.obj"))
        if obj_files:
            return str(obj_files[0])

        return None

    def get_class_id(self, obj_name: str) -> int:
        """Get class ID for an object name."""
        return self.class_mapping.get(obj_name, -1)

    def get_class_name(self, class_id: int) -> str:
        """Get object name for a class ID."""
        return self.id_to_name.get(class_id, "unknown")

    @property
    def num_classes(self) -> int:
        """Get total number of classes."""
        return len(self.class_mapping)

    def get_categories_for_coco(self) -> List[Dict]:
        """Get category list for COCO format annotations."""
        return [
            {
                "id": class_id,
                "name": obj_name,
                "supercategory": self.source_info.get(obj_name, "object"),
            }
            for obj_name, class_id in sorted(
                self.class_mapping.items(), key=lambda x: x[1]
            )
        ]


def load_model_sources(config: Dict[str, Any]) -> ModelSourceManager:
    """
    Load all model sources from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        ModelSourceManager with all models loaded
    """
    manager = ModelSourceManager()

    model_sources = config.get("model_sources", {})

    if not model_sources:
        # Fallback to legacy config format
        ycb_path = config.get("paths", {}).get("ycb_models")
        include_objects = config.get("objects", {}).get("include", [])
        if ycb_path:
            manager.load_source(
                "ycb",
                ycb_path,
                include_objects if include_objects else None,
                use_ycb_classes=True,
            )
        return manager

    # Load each configured source
    for source_name, source_config in model_sources.items():
        source_path = source_config.get("path", "")
        include_objects = source_config.get("include", [])

        # YCB source uses predefined class IDs for compatibility
        use_ycb_classes = source_name.lower() == "ycb"

        manager.load_source(
            source_name,
            source_path,
            include_objects if include_objects else None,
            use_ycb_classes=use_ycb_classes,
        )

    return manager


def normalize_object_scale(obj: bproc.types.MeshObject, target_size: float = 0.15) -> None:
    """
    Normalize object scale so that its largest dimension equals target_size.
    This handles models in different units (mm vs m).

    Args:
        obj: The mesh object to normalize
        target_size: Target size for the largest dimension in meters
    """
    # Get bounding box
    bbox = obj.get_bound_box()
    bbox_array = np.array(bbox)

    # Calculate dimensions
    min_coords = bbox_array.min(axis=0)
    max_coords = bbox_array.max(axis=0)
    dimensions = max_coords - min_coords

    # Get largest dimension
    max_dim = max(dimensions)

    if max_dim > 0:
        # Calculate scale factor
        scale_factor = target_size / max_dim

        # Apply scale
        current_scale = obj.get_scale()
        new_scale = [s * scale_factor for s in current_scale]
        obj.set_scale(new_scale)

        logger.debug(f"Normalized {obj.get_name()}: max_dim={max_dim:.3f}m -> {target_size}m (scale={scale_factor:.4f})")


def load_objects(
    model_manager: ModelSourceManager,
    selected_objects: List[str],
    material_randomizer: MaterialRandomizer,
    use_physics: bool = False,
) -> List[bproc.types.MeshObject]:
    """
    Load and prepare objects for the scene.

    Args:
        model_manager: ModelSourceManager instance
        selected_objects: List of object names to load
        material_randomizer: Material randomizer instance
        use_physics: Whether to enable physics simulation

    Returns:
        List of loaded mesh objects
    """
    loaded_objects = []

    for obj_name in selected_objects:
        if obj_name not in model_manager.model_paths:
            logger.warning(f"Skipping {obj_name}: model path not found")
            continue

        try:
            # Load the object
            objs = bproc.loader.load_obj(model_manager.model_paths[obj_name])

            for obj in objs:
                # Set object name for identification
                obj.set_name(obj_name)

                # Set custom property for class ID
                class_id = model_manager.get_class_id(obj_name)
                obj.set_cp("category_id", class_id)
                obj.set_cp("class_name", obj_name)

                # Normalize object scale (handles mm vs m unit differences)
                # Target size ~15cm for typical product objects
                normalize_object_scale(obj, target_size=0.15)

                # Skip material randomization to preserve original textures
                # TODO: Fix material randomization to not break textures
                # material_type = get_material_type(obj_name)
                # material_randomizer.randomize_object_material(obj, material_type)

                # Enable physics only if physics simulation will be used
                if use_physics:
                    obj.enable_rigidbody(
                        active=True,
                        collision_shape="CONVEX_HULL",
                        mass=0.1,
                    )

                loaded_objects.append(obj)

        except Exception as e:
            logger.error(f"Error loading {obj_name}: {e}")
            continue

    return loaded_objects


def generate_scene(
    config: Dict[str, Any],
    model_manager: ModelSourceManager,
    scene_setup: SceneSetup,
    lighting_randomizer: LightingRandomizer,
    camera_randomizer: CameraRandomizer,
    material_randomizer: MaterialRandomizer,
    scene_idx: int,
) -> Optional[Dict[str, Any]]:
    """
    Generate a single scene with domain randomization.

    Args:
        config: Configuration dictionary
        model_manager: ModelSourceManager instance
        scene_setup: Scene setup instance
        lighting_randomizer: Lighting randomizer instance
        camera_randomizer: Camera randomizer instance
        material_randomizer: Material randomizer instance
        scene_idx: Current scene index

    Returns:
        Scene metadata or None if generation failed
    """
    try:
        # Clear previous scene (keep Blender initialized)
        bproc.clean_up(clean_up_camera=True)

        # Setup background (floor, walls, table)
        room_result = scene_setup.create_room()
        surface_height = room_result.get("surface_height", 0)

        # Get placement config early to check physics setting
        placement_config = config.get("placement", {})
        use_physics = placement_config.get("use_physics", False)

        # Randomly select objects for this scene
        num_objects = random.randint(
            config["scene"]["objects_per_scene"][0],
            config["scene"]["objects_per_scene"][1],
        )
        available_objects = list(model_manager.model_paths.keys())
        selected_objects = random.sample(
            available_objects,
            min(num_objects, len(available_objects)),
        )

        # Load objects
        ycb_objects = load_objects(
            model_manager,
            selected_objects,
            material_randomizer,
            use_physics=use_physics,
        )

        if not ycb_objects:
            logger.warning(f"Scene {scene_idx}: No objects loaded, skipping")
            return None

        # Position objects with grid-based spacing to avoid overlap
        pos_cfg = placement_config.get("position", {})
        x_range = pos_cfg.get("x_range", [-0.2, 0.2])
        y_range = pos_cfg.get("y_range", [-0.2, 0.2])

        # Calculate grid spacing based on number of objects
        num_objs = len(ycb_objects)
        grid_size = int(np.ceil(np.sqrt(num_objs)))
        x_step = (x_range[1] - x_range[0]) / max(grid_size, 1)
        y_step = (y_range[1] - y_range[0]) / max(grid_size, 1)

        # Place objects in a grid with random offset
        for i, obj in enumerate(ycb_objects):
            grid_x = i % grid_size
            grid_y = i // grid_size

            # Base position from grid
            base_x = x_range[0] + (grid_x + 0.5) * x_step
            base_y = y_range[0] + (grid_y + 0.5) * y_step

            # Add small random offset (within cell)
            offset_x = random.uniform(-x_step * 0.3, x_step * 0.3)
            offset_y = random.uniform(-y_step * 0.3, y_step * 0.3)

            x = base_x + offset_x
            y = base_y + offset_y
            z = surface_height + 0.05

            obj.set_location([x, y, z])

            # Rotation - upright with random z rotation
            rx = np.radians(random.uniform(-10, 10))
            ry = np.radians(random.uniform(-10, 10))
            rz = np.radians(random.uniform(0, 360))
            obj.set_rotation_euler([rx, ry, rz])

            # Apply scale variation while preserving normalized scale
            scale_var = placement_config.get("scale_variation", [-0.05, 0.05])
            variation = 1.0 + random.uniform(scale_var[0], scale_var[1])
            current_scale = obj.get_scale()
            obj.set_scale([s * variation for s in current_scale])

        # Run physics simulation to settle objects
        if use_physics:
            bproc.object.simulate_physics_and_fix_final_poses(
                min_simulation_time=2,
                max_simulation_time=6,
                check_object_interval=1,
                substeps_per_frame=20,
            )

        # Setup lighting
        lighting_randomizer.setup_random_lighting()

        # Setup camera
        camera_randomizer.setup_camera(ycb_objects)

        # Render settings
        render_cfg = config.get("rendering", {})
        bproc.renderer.set_max_amount_of_samples(render_cfg.get("samples", 128))

        if render_cfg.get("use_denoising", True):
            # Enable denoiser via Blender API
            try:
                import bpy
                bpy.context.scene.cycles.use_denoising = True
            except Exception as e:
                logger.debug(f"Could not enable denoiser: {e}")

        # Enable COCO annotations
        for obj in ycb_objects:
            obj.set_cp("category_id", obj.get_cp("category_id"))

        # Render the scene
        data = bproc.renderer.render()

        # Generate COCO annotations
        # Get 2D bounding boxes
        # Use default_value to handle background objects
        seg_data = bproc.renderer.render_segmap(
            map_by=["category_id", "instance", "name"],
            default_values={"category_id": -1}
        )

        # Collect scene metadata
        scene_metadata = {
            "scene_idx": scene_idx,
            "objects": [
                {
                    "name": obj.get_name(),
                    "class_id": obj.get_cp("category_id"),
                    "location": obj.get_location().tolist(),
                    "rotation": obj.get_rotation_euler().tolist(),
                }
                for obj in ycb_objects
            ],
        }

        return {
            "data": data,
            "seg_data": seg_data,
            "metadata": scene_metadata,
            "objects": ycb_objects,
        }

    except Exception as e:
        logger.error(f"Error generating scene {scene_idx}: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_coco_annotations(
    output_dir: str,
    all_annotations: List[Dict],
    all_images: List[Dict],
    model_manager: ModelSourceManager,
) -> None:
    """
    Save annotations in COCO format.

    Args:
        output_dir: Output directory path
        all_annotations: List of annotation dictionaries
        all_images: List of image metadata dictionaries
        model_manager: ModelSourceManager instance for category info
    """
    # Build category list from model manager
    categories = model_manager.get_categories_for_coco()

    coco_format = {
        "info": {
            "description": "Synthetic Object Dataset",
            "version": "1.0",
            "year": 2026,
            "contributor": "BlenderProc",
        },
        "licenses": [],
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories,
    }

    output_path = Path(output_dir) / "annotations.json"
    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=2)

    logger.info(f"Saved COCO annotations to {output_path}")


def main():
    """Main entry point for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic YCB dataset with domain randomization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/workspace/scripts/blenderproc/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/workspace/data/synthetic/coco",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=None,
        help="Number of scenes to generate (overrides config)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="Starting index for scene numbering",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Setup random seeds
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    setup_random_seeds(seed)

    # Create output directories
    output_dir = Path(args.output)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load model sources
    model_manager = load_model_sources(config)

    if model_manager.num_classes == 0:
        logger.error("No models found. Exiting.")
        sys.exit(1)

    logger.info(f"Total classes: {model_manager.num_classes}")

    # Initialize randomizers
    textures_dir = config["paths"]["textures"]

    scene_setup = SceneSetup(config, textures_dir)
    lighting_randomizer = LightingRandomizer(config)
    camera_randomizer = CameraRandomizer(config)
    material_randomizer = MaterialRandomizer(config)

    # Initialize BlenderProc once
    bproc.init()

    # Determine number of scenes
    num_scenes = args.num_scenes if args.num_scenes is not None else config["scene"]["num_images"]

    logger.info(f"Generating {num_scenes} scenes")
    logger.info(f"Output directory: {output_dir}")

    # Track all annotations and images for COCO format
    all_annotations = []
    all_images = []
    annotation_id = 1

    # Generate scenes
    for scene_idx in range(args.start_idx, args.start_idx + num_scenes):
        logger.info(f"Generating scene {scene_idx + 1}/{args.start_idx + num_scenes}")

        result = generate_scene(
            config,
            model_manager,
            scene_setup,
            lighting_randomizer,
            camera_randomizer,
            material_randomizer,
            scene_idx,
        )

        if result is None:
            continue

        # Save rendered image
        image_filename = f"scene_{scene_idx:06d}.png"
        image_path = images_dir / image_filename

        # Get the rendered image
        colors = result["data"]["colors"][0]

        # Save using BlenderProc's writer or manually
        from PIL import Image
        img = Image.fromarray((colors * 255).astype(np.uint8) if colors.max() <= 1 else colors.astype(np.uint8))
        img.save(image_path)

        # Get image dimensions
        height, width = colors.shape[:2]

        # Add image metadata
        image_info = {
            "id": scene_idx,
            "file_name": image_filename,
            "width": width,
            "height": height,
        }
        all_images.append(image_info)

        # Generate bounding box annotations from segmentation
        seg_data = result["seg_data"]

        # Get instance segmentation map
        if "instance_segmaps" not in seg_data:
            logger.warning(f"Scene {scene_idx}: No instance segmentation map")
            continue

        instance_map = seg_data["instance_segmaps"][0]

        # Get instance attribute maps for category_id lookup
        instance_attrs_list = seg_data.get("instance_attribute_maps", [])
        instance_attrs = instance_attrs_list[0] if instance_attrs_list else []

        # Build a mapping from instance idx to attributes
        idx_to_attrs = {}
        if isinstance(instance_attrs, list):
            for attr_dict in instance_attrs:
                if isinstance(attr_dict, dict) and 'idx' in attr_dict:
                    idx_to_attrs[attr_dict['idx']] = attr_dict

        # Build a mapping from object name to category_id from our loaded objects
        obj_name_to_category = {}
        for obj in result.get("objects", []):
            try:
                obj_name = obj.get_name()
                cat_id = obj.get_cp("category_id")
                if cat_id is not None and cat_id >= 0:
                    obj_name_to_category[obj_name] = cat_id
            except Exception:
                pass

        # Get unique instances
        unique_instances = np.unique(instance_map)

        for inst_id in unique_instances:
            if inst_id == 0:  # Skip background
                continue

            # Get mask for this instance
            mask = instance_map == inst_id

            # Get category ID from instance attributes
            category_id = -1

            if inst_id in idx_to_attrs:
                attrs = idx_to_attrs[inst_id]
                # Get category_id directly
                category_id = attrs.get("category_id", -1)

                # If category_id is -1, try to match by name
                if category_id < 0:
                    obj_name = attrs.get("name", "")
                    if obj_name in obj_name_to_category:
                        category_id = obj_name_to_category[obj_name]

            if category_id < 0 or category_id not in model_manager.id_to_name:
                continue

            # Calculate bounding box
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)

            if not np.any(rows) or not np.any(cols):
                continue

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # COCO format: [x, y, width, height]
            bbox = [
                int(cmin),
                int(rmin),
                int(cmax - cmin + 1),
                int(rmax - rmin + 1),
            ]

            # Calculate area
            area = int(np.sum(mask))

            # Create annotation
            annotation = {
                "id": annotation_id,
                "image_id": scene_idx,
                "category_id": category_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
            all_annotations.append(annotation)
            annotation_id += 1

        # Log progress
        if (scene_idx + 1) % 100 == 0:
            logger.info(f"Progress: {scene_idx + 1}/{args.start_idx + num_scenes} scenes completed")

    # Save COCO annotations
    save_coco_annotations(str(output_dir), all_annotations, all_images, model_manager)

    logger.info("Dataset generation complete!")
    logger.info(f"Generated {len(all_images)} images with {len(all_annotations)} annotations")


if __name__ == "__main__":
    main()
