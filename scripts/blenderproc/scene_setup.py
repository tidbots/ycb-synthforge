"""
Scene Setup Module for BlenderProc
Creates room environments with randomized textures for domain randomization.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import blenderproc as bproc
import numpy as np

logger = logging.getLogger(__name__)


class SceneSetup:
    """Handles scene creation and background randomization."""

    def __init__(self, config: Dict[str, Any], textures_dir: str):
        """
        Initialize scene setup.

        Args:
            config: Configuration dictionary
            textures_dir: Path to textures directory
        """
        self.config = config
        self.textures_dir = Path(textures_dir)
        self.background_config = config.get("background", {})

        # Cache texture paths by category
        self._texture_cache: Dict[str, List[Path]] = {}
        self._load_texture_paths()

    def _load_texture_paths(self) -> None:
        """Load and categorize available texture paths."""
        if not self.textures_dir.exists():
            logger.warning(f"Textures directory not found: {self.textures_dir}")
            return

        # Define texture category patterns
        category_patterns = {
            "Wood": ["Wood", "Planks", "WoodFloor", "WoodSiding", "Bamboo"],
            "Concrete": ["Concrete", "Asphalt", "Road"],
            "Tiles": ["Tiles", "PavingStones", "Terrazzo"],
            "Marble": ["Marble", "Granite", "Travertine", "Onyx"],
            "Metal": ["Metal", "DiamondPlate", "CorrugatedSteel", "SheetMetal", "Rust"],
            "Fabric": ["Fabric", "Carpet", "Leather", "Wicker"],
            "Plaster": ["Plaster", "Stucco"],
            "Brick": ["Brick", "Facade"],
            "Paint": ["Paint", "Wall"],
            "Wallpaper": ["Wallpaper", "Pattern"],
            "Plastic": ["Plastic", "Rubber", "Foam", "Styrofoam"],
            "Ground": ["Ground", "Gravel", "Grass", "Rock", "Rocks"],
        }

        # Scan textures directory
        for texture_dir in self.textures_dir.iterdir():
            if not texture_dir.is_dir():
                continue

            texture_name = texture_dir.name

            # Categorize texture
            for category, patterns in category_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in texture_name.lower():
                        if category not in self._texture_cache:
                            self._texture_cache[category] = []
                        self._texture_cache[category].append(texture_dir)
                        break

        # Log loaded textures
        for category, paths in self._texture_cache.items():
            logger.debug(f"Loaded {len(paths)} textures for category: {category}")

    def get_random_texture(self, categories: List[str]) -> Optional[Path]:
        """
        Get a random texture path from specified categories.

        Args:
            categories: List of texture categories to sample from

        Returns:
            Path to texture directory or None if not found
        """
        available = []
        for category in categories:
            if category in self._texture_cache:
                available.extend(self._texture_cache[category])

        if not available:
            # Fallback to any available texture
            all_textures = []
            for paths in self._texture_cache.values():
                all_textures.extend(paths)
            if all_textures:
                return random.choice(all_textures)
            return None

        return random.choice(available)

    def load_cc_texture(self, texture_path: Path) -> Optional[bproc.types.Material]:
        """
        Load a CC0 texture as a material.

        Args:
            texture_path: Path to texture directory

        Returns:
            BlenderProc material or None if loading failed
        """
        try:
            # Find color/diffuse texture
            color_files = list(texture_path.glob("*_Color.*")) + \
                         list(texture_path.glob("*_Diffuse.*")) + \
                         list(texture_path.glob("*_BaseColor.*"))

            if not color_files:
                # Try to find any image file
                color_files = list(texture_path.glob("*.jpg")) + \
                             list(texture_path.glob("*.png"))

            if not color_files:
                logger.warning(f"No color texture found in {texture_path}")
                return None

            # Load material using BlenderProc
            materials = bproc.loader.load_ccmaterials(
                folder_path=str(texture_path.parent),
                used_assets=[texture_path.name],
            )

            if materials:
                return materials[0]
            return None

        except Exception as e:
            logger.error(f"Error loading texture {texture_path}: {e}")
            return None

    def create_floor(
        self,
        size: Tuple[float, float] = (3.0, 3.0),
        location: Tuple[float, float, float] = (0, 0, 0),
    ) -> bproc.types.MeshObject:
        """
        Create a floor plane with randomized texture.

        Args:
            size: Floor dimensions (width, depth)
            location: Floor center location

        Returns:
            Floor mesh object
        """
        # Create floor plane
        floor = bproc.object.create_primitive(
            shape="PLANE",
            size=max(size),
            location=location,
        )
        floor.set_name("floor")

        # Mark as background (not a target object)
        floor.set_cp("category_id", -1)
        floor.set_cp("is_background", True)

        # Scale to desired size
        floor.set_scale([size[0] / 2, size[1] / 2, 1])

        # Get texture categories from config
        floor_config = self.background_config.get("floor", {})
        categories = floor_config.get("categories", ["Wood", "Concrete", "Tiles"])

        # Apply random texture
        texture_path = self.get_random_texture(categories)
        if texture_path:
            material = self.load_cc_texture(texture_path)
            if material:
                # Apply texture scale randomization
                scale_range = floor_config.get("texture_scale", [0.5, 2.0])
                tex_scale = random.uniform(scale_range[0], scale_range[1])

                # Apply material to floor
                floor.replace_materials(material)

                # Randomize material properties
                self._randomize_material_properties(material, floor_config)

        # Make floor static for physics
        floor.enable_rigidbody(
            active=False,
            collision_shape="BOX",
        )

        return floor

    def create_walls(
        self,
        room_size: Tuple[float, float, float] = (3.0, 3.0, 2.5),
    ) -> List[bproc.types.MeshObject]:
        """
        Create walls with randomized textures.

        Args:
            room_size: Room dimensions (width, depth, height)

        Returns:
            List of wall mesh objects
        """
        walls = []
        width, depth, height = room_size

        # Wall configurations: (location, rotation, size)
        wall_configs = [
            # Back wall
            ((0, depth / 2, height / 2), (np.pi / 2, 0, 0), (width, height)),
            # Left wall
            ((-width / 2, 0, height / 2), (np.pi / 2, 0, np.pi / 2), (depth, height)),
            # Right wall
            ((width / 2, 0, height / 2), (np.pi / 2, 0, -np.pi / 2), (depth, height)),
        ]

        wall_config = self.background_config.get("wall", {})
        categories = wall_config.get("categories", ["Concrete", "Plaster", "Brick"])

        for i, (loc, rot, size) in enumerate(wall_configs):
            wall = bproc.object.create_primitive(
                shape="PLANE",
                size=1,
                location=loc,
            )
            wall.set_name(f"wall_{i}")

            # Mark as background
            wall.set_cp("category_id", -1)
            wall.set_cp("is_background", True)

            wall.set_rotation_euler(rot)
            wall.set_scale([size[0] / 2, size[1] / 2, 1])

            # Apply random texture
            texture_path = self.get_random_texture(categories)
            if texture_path:
                material = self.load_cc_texture(texture_path)
                if material:
                    wall.replace_materials(material)
                    self._randomize_material_properties(material, wall_config)

            # Make wall static
            wall.enable_rigidbody(active=False, collision_shape="BOX")

            walls.append(wall)

        return walls

    def create_table(
        self,
        size: Tuple[float, float, float] = (1.0, 0.6, 0.75),
        location: Tuple[float, float, float] = (0, 0, 0),
    ) -> bproc.types.MeshObject:
        """
        Create a table/shelf surface with randomized texture.

        Args:
            size: Table dimensions (width, depth, height)
            location: Table center location

        Returns:
            Table mesh object
        """
        width, depth, height = size

        # Create table top
        table = bproc.object.create_primitive(
            shape="CUBE",
            size=1,
            location=(location[0], location[1], location[2] + height / 2),
        )
        table.set_name("table")

        # Mark as background
        table.set_cp("category_id", -1)
        table.set_cp("is_background", True)

        table.set_scale([width / 2, depth / 2, 0.02])  # Thin surface

        # Get texture categories from config
        surface_config = self.background_config.get("surface", {})
        categories = surface_config.get("categories", ["Wood", "Metal", "Plastic"])

        # Apply random texture
        texture_path = self.get_random_texture(categories)
        if texture_path:
            material = self.load_cc_texture(texture_path)
            if material:
                table.replace_materials(material)

        # Make table static
        table.enable_rigidbody(active=False, collision_shape="BOX")

        return table

    def _randomize_material_properties(
        self,
        material: bproc.types.Material,
        config: Dict[str, Any],
    ) -> None:
        """
        Randomize material properties for domain randomization.

        Args:
            material: Material to randomize
            config: Configuration for randomization ranges
        """
        try:
            # Hue shift
            hue_range = config.get("hue_shift", [-30, 30])
            # Note: Hue shifting would require shader node manipulation
            # This is a simplified version

            # Brightness adjustment
            brightness_range = config.get("brightness", [0.7, 1.3])
            brightness = random.uniform(brightness_range[0], brightness_range[1])

            # Apply brightness by adjusting base color
            # This requires access to the material's shader nodes
            # Simplified implementation here

        except Exception as e:
            logger.debug(f"Could not randomize material properties: {e}")

    def create_room(
        self,
        room_size: Tuple[float, float, float] = (3.0, 3.0, 2.5),
        include_walls: bool = True,
        include_table: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a complete room environment.

        Args:
            room_size: Room dimensions (width, depth, height)
            include_walls: Whether to create walls
            include_table: Whether to create a table

        Returns:
            Dictionary containing created objects
        """
        result = {}

        # Create floor
        floor = self.create_floor(
            size=(room_size[0], room_size[1]),
            location=(0, 0, 0),
        )
        result["floor"] = floor

        # Create walls (optional, based on random choice or config)
        if include_walls and random.random() < 0.7:  # 70% chance of walls
            walls = self.create_walls(room_size)
            result["walls"] = walls

        # Create table (optional)
        if include_table and random.random() < 0.8:  # 80% chance of table
            table_height = random.uniform(0.5, 0.9)
            table = self.create_table(
                size=(random.uniform(0.8, 1.5), random.uniform(0.5, 1.0), table_height),
                location=(0, 0, 0),
            )
            result["table"] = table
            result["surface_height"] = table_height
        else:
            result["surface_height"] = 0  # Objects on floor

        return result

    def add_clutter_objects(
        self,
        num_objects: int,
        surface_height: float = 0.0,
    ) -> List[bproc.types.MeshObject]:
        """
        Add clutter objects to the scene (unlabeled distractors).

        Args:
            num_objects: Number of clutter objects to add
            surface_height: Height of the surface to place objects on

        Returns:
            List of clutter mesh objects
        """
        clutter_objects = []

        # Simple primitive shapes for clutter
        shapes = ["CUBE", "SPHERE", "CYLINDER", "CONE"]

        for i in range(num_objects):
            shape = random.choice(shapes)
            size = random.uniform(0.02, 0.1)

            obj = bproc.object.create_primitive(
                shape=shape,
                size=size,
            )
            obj.set_name(f"clutter_{i}")

            # Random position
            x = random.uniform(-0.4, 0.4)
            y = random.uniform(-0.4, 0.4)
            z = surface_height + size / 2 + random.uniform(0.05, 0.3)
            obj.set_location([x, y, z])

            # Random rotation
            obj.set_rotation_euler([
                random.uniform(0, 2 * np.pi),
                random.uniform(0, 2 * np.pi),
                random.uniform(0, 2 * np.pi),
            ])

            # Random color
            material = bproc.material.create("clutter_material")
            material.set_principled_shader_value(
                "Base Color",
                [random.random(), random.random(), random.random(), 1.0],
            )
            obj.replace_materials(material)

            # Enable physics
            obj.enable_rigidbody(
                active=True,
                collision_shape="CONVEX_HULL",
                mass=0.05,
            )

            # Mark as clutter (not labeled)
            obj.set_cp("category_id", -1)
            obj.set_cp("is_clutter", True)

            clutter_objects.append(obj)

        return clutter_objects
