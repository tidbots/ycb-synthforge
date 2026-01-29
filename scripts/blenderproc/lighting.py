"""
Lighting Module for BlenderProc
Implements lighting randomization for domain randomization.
"""

import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import blenderproc as bproc
import numpy as np

logger = logging.getLogger(__name__)


def kelvin_to_rgb(kelvin: float) -> Tuple[float, float, float]:
    """
    Convert color temperature in Kelvin to RGB values.

    Args:
        kelvin: Color temperature in Kelvin (1000-40000)

    Returns:
        Tuple of (R, G, B) values in range [0, 1]
    """
    # Algorithm based on Tanner Helland's work
    temp = kelvin / 100.0

    # Calculate red
    if temp <= 66:
        red = 255
    else:
        red = temp - 60
        red = 329.698727446 * (red ** -0.1332047592)
        red = max(0, min(255, red))

    # Calculate green
    if temp <= 66:
        green = temp
        green = 99.4708025861 * math.log(green) - 161.1195681661
        green = max(0, min(255, green))
    else:
        green = temp - 60
        green = 288.1221695283 * (green ** -0.0755148492)
        green = max(0, min(255, green))

    # Calculate blue
    if temp >= 66:
        blue = 255
    elif temp <= 19:
        blue = 0
    else:
        blue = temp - 10
        blue = 138.5177312231 * math.log(blue) - 305.0447927307
        blue = max(0, min(255, blue))

    return (red / 255.0, green / 255.0, blue / 255.0)


class LightingRandomizer:
    """Handles lighting setup and randomization."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize lighting randomizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.lighting_config = config.get("lighting", {})
        self.hdri_dir = config.get("paths", {}).get("hdri", None)

    def create_point_light(
        self,
        location: Tuple[float, float, float],
        energy: float = 500.0,
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> bproc.types.Light:
        """
        Create a point light.

        Args:
            location: Light position
            energy: Light energy in watts
            color: Light color (RGB)

        Returns:
            Light object
        """
        light = bproc.types.Light()
        light.set_type("POINT")
        light.set_location(location)
        light.set_energy(energy)
        light.set_color(color)
        return light

    def create_area_light(
        self,
        location: Tuple[float, float, float],
        rotation: Tuple[float, float, float],
        energy: float = 500.0,
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        size: float = 1.0,
    ) -> bproc.types.Light:
        """
        Create an area light (soft shadows).

        Args:
            location: Light position
            rotation: Light rotation (Euler angles)
            energy: Light energy in watts
            color: Light color (RGB)
            size: Light size for soft shadows

        Returns:
            Light object
        """
        light = bproc.types.Light()
        light.set_type("AREA")
        light.set_location(location)
        light.set_rotation_euler(rotation)
        light.set_energy(energy)
        light.set_color(color)
        # Area light size would need to be set via Blender API directly
        return light

    def create_sun_light(
        self,
        rotation: Tuple[float, float, float],
        energy: float = 5.0,
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> bproc.types.Light:
        """
        Create a sun light (directional).

        Args:
            rotation: Light direction (Euler angles)
            energy: Light energy
            color: Light color (RGB)

        Returns:
            Light object
        """
        light = bproc.types.Light()
        light.set_type("SUN")
        light.set_rotation_euler(rotation)
        light.set_energy(energy)
        light.set_color(color)
        return light

    def setup_random_lighting(self) -> List[bproc.types.Light]:
        """
        Setup random lighting configuration.

        Returns:
            List of created light objects
        """
        lights = []

        # Get configuration
        num_lights_range = self.lighting_config.get("num_lights", [1, 4])
        num_lights = random.randint(num_lights_range[0], num_lights_range[1])

        temp_range = self.lighting_config.get("color_temperature", [2700, 6500])
        intensity_range = self.lighting_config.get("intensity", [100, 1000])
        elevation_range = self.lighting_config.get("elevation", [30, 90])
        azimuth_range = self.lighting_config.get("azimuth", [0, 360])

        logger.debug(f"Creating {num_lights} lights")

        for i in range(num_lights):
            # Random color temperature
            kelvin = random.uniform(temp_range[0], temp_range[1])
            color = kelvin_to_rgb(kelvin)

            # Random intensity
            energy = random.uniform(intensity_range[0], intensity_range[1])

            # Random position (spherical coordinates)
            elevation = np.radians(random.uniform(elevation_range[0], elevation_range[1]))
            azimuth = np.radians(random.uniform(azimuth_range[0], azimuth_range[1]))
            distance = random.uniform(1.5, 3.0)

            # Convert to Cartesian coordinates
            x = distance * math.cos(elevation) * math.cos(azimuth)
            y = distance * math.cos(elevation) * math.sin(azimuth)
            z = distance * math.sin(elevation)

            # Randomly choose light type
            light_type = random.choice(["POINT", "AREA", "SPOT"])

            if light_type == "POINT":
                light = self.create_point_light(
                    location=(x, y, z),
                    energy=energy,
                    color=color,
                )
            elif light_type == "AREA":
                # Area light needs to point toward scene center
                rot_x = -elevation + np.pi / 2
                rot_z = azimuth
                light = self.create_area_light(
                    location=(x, y, z),
                    rotation=(rot_x, 0, rot_z),
                    energy=energy * 2,  # Area lights need more energy
                    color=color,
                    size=random.uniform(0.5, 2.0),
                )
            else:  # SPOT
                light = bproc.types.Light()
                light.set_type("SPOT")
                light.set_location([x, y, z])
                light.set_energy(energy * 1.5)
                light.set_color(color)
                # Point toward scene center
                direction = np.array([0, 0, 0]) - np.array([x, y, z])
                direction = direction / np.linalg.norm(direction)
                # Calculate rotation to point toward origin
                light.set_rotation_euler([
                    math.acos(direction[2]),
                    0,
                    math.atan2(direction[1], direction[0]),
                ])

            lights.append(light)

        # Add ambient lighting
        self._setup_ambient_lighting()

        # Optionally add HDRI environment
        if self.lighting_config.get("use_hdri", True) and self.hdri_dir:
            self._setup_hdri_lighting()

        return lights

    def _setup_ambient_lighting(self) -> None:
        """Setup ambient/environment lighting."""
        ambient_range = self.lighting_config.get("ambient", [0.1, 0.5])
        ambient_strength = random.uniform(ambient_range[0], ambient_range[1])

        try:
            # Set a simple background color with ambient strength
            import bpy
            world = bpy.context.scene.world
            if world is None:
                world = bpy.data.worlds.new("World")
                bpy.context.scene.world = world
            world.use_nodes = True
            bg_node = world.node_tree.nodes.get("Background")
            if bg_node:
                # Set gray background with ambient strength
                gray = ambient_strength * 0.5
                bg_node.inputs["Color"].default_value = (gray, gray, gray, 1.0)
                bg_node.inputs["Strength"].default_value = ambient_strength
        except Exception as e:
            logger.debug(f"Could not setup ambient lighting: {e}")

    def _setup_hdri_lighting(self) -> None:
        """Setup HDRI environment lighting."""
        if not self.hdri_dir:
            return

        hdri_path = Path(self.hdri_dir)
        if not hdri_path.exists():
            logger.debug(f"HDRI directory not found: {hdri_path}")
            return

        # Find HDRI files
        hdri_files = list(hdri_path.glob("*.hdr")) + list(hdri_path.glob("*.exr"))
        if not hdri_files:
            logger.debug("No HDRI files found")
            return

        # Random HDRI selection
        hdri_file = random.choice(hdri_files)

        try:
            # Set HDRI as world background
            rotation = np.radians(random.uniform(0, 360))
            bproc.world.set_world_background_hdr_img(
                path=str(hdri_file),
                rotation_z=rotation,
            )
            logger.debug(f"Applied HDRI: {hdri_file.name}")
        except Exception as e:
            logger.debug(f"Could not load HDRI: {e}")

    def create_three_point_lighting(
        self,
        target_location: Tuple[float, float, float] = (0, 0, 0.5),
        key_intensity: float = 800.0,
    ) -> List[bproc.types.Light]:
        """
        Create classic three-point lighting setup.

        Args:
            target_location: Point to illuminate
            key_intensity: Intensity of key light

        Returns:
            List of three lights (key, fill, back)
        """
        lights = []

        # Random color temperature for key light
        temp_range = self.lighting_config.get("color_temperature", [2700, 6500])
        kelvin = random.uniform(temp_range[0], temp_range[1])
        key_color = kelvin_to_rgb(kelvin)

        # Key light (main light, 45 degrees to side and above)
        key_azimuth = np.radians(random.uniform(30, 60))
        key_elevation = np.radians(random.uniform(30, 50))
        key_distance = 2.0

        key_x = key_distance * math.cos(key_elevation) * math.cos(key_azimuth)
        key_y = key_distance * math.cos(key_elevation) * math.sin(key_azimuth)
        key_z = target_location[2] + key_distance * math.sin(key_elevation)

        key_light = self.create_point_light(
            location=(key_x, key_y, key_z),
            energy=key_intensity,
            color=key_color,
        )
        lights.append(key_light)

        # Fill light (opposite side, softer)
        fill_azimuth = np.radians(random.uniform(-60, -30))
        fill_elevation = np.radians(random.uniform(20, 40))
        fill_distance = 2.5

        fill_x = fill_distance * math.cos(fill_elevation) * math.cos(fill_azimuth)
        fill_y = fill_distance * math.cos(fill_elevation) * math.sin(fill_azimuth)
        fill_z = target_location[2] + fill_distance * math.sin(fill_elevation)

        fill_light = self.create_area_light(
            location=(fill_x, fill_y, fill_z),
            rotation=(0, 0, 0),
            energy=key_intensity * 0.5,  # Fill is softer
            color=(1.0, 1.0, 1.0),  # Neutral fill
            size=1.5,
        )
        lights.append(fill_light)

        # Back light (rim light, behind subject)
        back_azimuth = np.radians(random.uniform(150, 210))
        back_elevation = np.radians(random.uniform(40, 60))
        back_distance = 2.0

        back_x = back_distance * math.cos(back_elevation) * math.cos(back_azimuth)
        back_y = back_distance * math.cos(back_elevation) * math.sin(back_azimuth)
        back_z = target_location[2] + back_distance * math.sin(back_elevation)

        back_light = self.create_point_light(
            location=(back_x, back_y, back_z),
            energy=key_intensity * 0.7,
            color=key_color,
        )
        lights.append(back_light)

        return lights

    def create_overhead_lighting(
        self,
        num_lights: int = 4,
        height: float = 2.5,
        area_size: Tuple[float, float] = (2.0, 2.0),
    ) -> List[bproc.types.Light]:
        """
        Create overhead ceiling-style lighting.

        Args:
            num_lights: Number of overhead lights
            height: Height of lights
            area_size: Area to distribute lights over

        Returns:
            List of overhead lights
        """
        lights = []

        # Random color temperature (typically office/warehouse lighting)
        temp_range = [3500, 5500]  # Neutral white range
        kelvin = random.uniform(temp_range[0], temp_range[1])
        color = kelvin_to_rgb(kelvin)

        # Grid distribution
        grid_size = int(math.ceil(math.sqrt(num_lights)))
        x_step = area_size[0] / (grid_size + 1)
        y_step = area_size[1] / (grid_size + 1)

        intensity = random.uniform(200, 500)

        for i in range(num_lights):
            row = i // grid_size
            col = i % grid_size

            x = -area_size[0] / 2 + (col + 1) * x_step + random.uniform(-0.1, 0.1)
            y = -area_size[1] / 2 + (row + 1) * y_step + random.uniform(-0.1, 0.1)
            z = height

            # Area lights for soft ceiling lighting
            light = self.create_area_light(
                location=(x, y, z),
                rotation=(0, 0, 0),  # Pointing down
                energy=intensity,
                color=color,
                size=0.5,
            )
            lights.append(light)

        return lights
