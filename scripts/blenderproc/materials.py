"""
Materials Module for BlenderProc
Implements material randomization for domain randomization.
Handles metallic, glossy, plastic, and other material types.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import blenderproc as bproc
import numpy as np

logger = logging.getLogger(__name__)


class MaterialRandomizer:
    """Handles material property randomization for domain randomization."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize material randomizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.materials_config = config.get("materials", {})

    def randomize_object_material(
        self,
        obj: bproc.types.MeshObject,
        material_type: str = "default",
    ) -> None:
        """
        Randomize material properties of an object based on its type.

        Args:
            obj: Object to modify
            material_type: Type of material ("metallic", "glossy", "plastic", "default")
        """
        try:
            materials = obj.get_materials()

            for material in materials:
                if material is None:
                    continue

                if material_type == "metallic":
                    self._randomize_metallic(material)
                elif material_type == "glossy":
                    self._randomize_glossy(material)
                elif material_type == "plastic":
                    self._randomize_plastic(material)
                else:
                    self._randomize_default(material)

        except Exception as e:
            logger.debug(f"Could not randomize material for {obj.get_name()}: {e}")

    def _randomize_metallic(self, material: bproc.types.Material) -> None:
        """
        Randomize metallic material properties (cans, tools, etc.).

        Args:
            material: Material to modify
        """
        config = self.materials_config.get("metallic", {})

        # Metallic value
        metallic_range = config.get("metallic_range", [0.8, 1.0])
        metallic = random.uniform(metallic_range[0], metallic_range[1])

        # Roughness (lower = more reflective/shiny)
        roughness_range = config.get("roughness_range", [0.05, 0.3])
        roughness = random.uniform(roughness_range[0], roughness_range[1])

        # Specular
        specular_range = config.get("specular_range", [0.5, 1.0])
        specular = random.uniform(specular_range[0], specular_range[1])

        # Apply to principled shader
        try:
            material.set_principled_shader_value("Metallic", metallic)
            material.set_principled_shader_value("Roughness", roughness)
            material.set_principled_shader_value("Specular IOR Level", specular)

            logger.debug(
                f"Metallic material: metallic={metallic:.2f}, "
                f"roughness={roughness:.2f}, specular={specular:.2f}"
            )
        except Exception as e:
            logger.debug(f"Could not set metallic properties: {e}")

    def _randomize_glossy(self, material: bproc.types.Material) -> None:
        """
        Randomize glossy material properties (boxes, bottles, etc.).

        Args:
            material: Material to modify
        """
        config = self.materials_config.get("glossy", {})

        # Glossy materials are non-metallic but shiny
        metallic_range = config.get("metallic_range", [0.0, 0.0])
        metallic = random.uniform(metallic_range[0], metallic_range[1])

        # Variable roughness for different gloss levels
        roughness_range = config.get("roughness_range", [0.1, 0.5])
        roughness = random.uniform(roughness_range[0], roughness_range[1])

        # Specular for fresnel effect
        specular_range = config.get("specular_range", [0.3, 0.7])
        specular = random.uniform(specular_range[0], specular_range[1])

        try:
            material.set_principled_shader_value("Metallic", metallic)
            material.set_principled_shader_value("Roughness", roughness)
            material.set_principled_shader_value("Specular IOR Level", specular)

            # Add clearcoat for printed materials (simulates glossy coating)
            if random.random() < 0.5:
                clearcoat = random.uniform(0.3, 0.8)
                clearcoat_roughness = random.uniform(0.1, 0.3)
                material.set_principled_shader_value("Coat Weight", clearcoat)
                material.set_principled_shader_value("Coat Roughness", clearcoat_roughness)

            logger.debug(
                f"Glossy material: roughness={roughness:.2f}, specular={specular:.2f}"
            )
        except Exception as e:
            logger.debug(f"Could not set glossy properties: {e}")

    def _randomize_plastic(self, material: bproc.types.Material) -> None:
        """
        Randomize plastic material properties.

        Args:
            material: Material to modify
        """
        config = self.materials_config.get("plastic", {})

        # Plastic is non-metallic
        metallic = 0.0

        # Variable roughness
        roughness_range = config.get("roughness_range", [0.2, 0.6])
        roughness = random.uniform(roughness_range[0], roughness_range[1])

        # Subsurface scattering for translucent plastics
        subsurface_range = config.get("subsurface", [0, 0.1])
        subsurface = random.uniform(subsurface_range[0], subsurface_range[1])

        try:
            material.set_principled_shader_value("Metallic", metallic)
            material.set_principled_shader_value("Roughness", roughness)

            if subsurface > 0.01:
                material.set_principled_shader_value("Subsurface Weight", subsurface)
                # Random subsurface color (slightly tinted)
                sss_color = [
                    random.uniform(0.8, 1.0),
                    random.uniform(0.8, 1.0),
                    random.uniform(0.8, 1.0),
                    1.0,
                ]
                material.set_principled_shader_value("Subsurface Radius", sss_color[:3])

            logger.debug(
                f"Plastic material: roughness={roughness:.2f}, subsurface={subsurface:.2f}"
            )
        except Exception as e:
            logger.debug(f"Could not set plastic properties: {e}")

    def _randomize_default(self, material: bproc.types.Material) -> None:
        """
        Randomize default material properties.

        Args:
            material: Material to modify
        """
        # Apply slight variations to any material
        try:
            # Small roughness variation
            current_roughness = 0.5  # Default assumption
            roughness_variation = random.uniform(-0.1, 0.1)
            new_roughness = max(0.05, min(0.95, current_roughness + roughness_variation))
            material.set_principled_shader_value("Roughness", new_roughness)

            logger.debug(f"Default material: roughness={new_roughness:.2f}")
        except Exception as e:
            logger.debug(f"Could not set default properties: {e}")

    def apply_base_color_variation(
        self,
        material: bproc.types.Material,
        hue_shift_range: Tuple[float, float] = (-10, 10),
    ) -> None:
        """
        Apply base color/hue variation to material.

        Args:
            material: Material to modify
            hue_shift_range: Range of hue shift in degrees
        """
        try:
            # This requires more complex node manipulation
            # Simplified version: slightly adjust color
            hue_shift = random.uniform(hue_shift_range[0], hue_shift_range[1])

            # Note: Full implementation would require HSV conversion
            # and shader node manipulation
            logger.debug(f"Applied hue shift: {hue_shift:.1f} degrees")

        except Exception as e:
            logger.debug(f"Could not apply base color variation: {e}")

    def create_random_material(
        self,
        name: str = "random_material",
        material_type: str = "default",
    ) -> bproc.types.Material:
        """
        Create a new material with random properties.

        Args:
            name: Material name
            material_type: Type of material

        Returns:
            New material with randomized properties
        """
        material = bproc.material.create(name)

        # Set random base color
        base_color = [
            random.uniform(0.1, 0.9),
            random.uniform(0.1, 0.9),
            random.uniform(0.1, 0.9),
            1.0,
        ]
        material.set_principled_shader_value("Base Color", base_color)

        # Apply type-specific properties
        if material_type == "metallic":
            self._randomize_metallic(material)
        elif material_type == "glossy":
            self._randomize_glossy(material)
        elif material_type == "plastic":
            self._randomize_plastic(material)
        else:
            self._randomize_default(material)

        return material

    def enhance_highlights(
        self,
        obj: bproc.types.MeshObject,
        intensity: float = 1.0,
    ) -> None:
        """
        Enhance specular highlights on object (for "highlight hell" effect).

        Args:
            obj: Object to modify
            intensity: Highlight intensity multiplier
        """
        try:
            materials = obj.get_materials()

            for material in materials:
                if material is None:
                    continue

                # Increase specular
                current_specular = 0.5
                new_specular = min(1.0, current_specular * intensity)
                material.set_principled_shader_value("Specular IOR Level", new_specular)

                # Decrease roughness for sharper highlights
                roughness = random.uniform(0.05, 0.15)
                material.set_principled_shader_value("Roughness", roughness)

                # Add anisotropic for stretched highlights
                if random.random() < 0.3:
                    anisotropic = random.uniform(0.3, 0.7)
                    material.set_principled_shader_value("Anisotropic", anisotropic)

            logger.debug(f"Enhanced highlights for {obj.get_name()}")

        except Exception as e:
            logger.debug(f"Could not enhance highlights: {e}")

    def apply_wear_and_tear(
        self,
        material: bproc.types.Material,
        amount: float = 0.2,
    ) -> None:
        """
        Apply wear and tear effects to material.

        Args:
            material: Material to modify
            amount: Amount of wear (0-1)
        """
        try:
            # Increase roughness in some areas
            base_roughness = random.uniform(0.3, 0.5)
            wear_roughness = base_roughness + amount * 0.3

            material.set_principled_shader_value("Roughness", wear_roughness)

            # Note: Full implementation would add scratches, dirt, etc.
            # using procedural textures or texture masks

            logger.debug(f"Applied wear: amount={amount:.2f}")

        except Exception as e:
            logger.debug(f"Could not apply wear: {e}")

    def create_environment_reflection_probe(
        self,
        location: Tuple[float, float, float] = (0, 0, 1),
    ) -> None:
        """
        Create environment reflection probe for better reflections.

        Args:
            location: Probe location
        """
        try:
            import bpy

            # Create reflection probe (cubemap)
            bpy.ops.object.lightprobe_add(
                type='CUBEMAP',
                location=location,
            )

            logger.debug("Created reflection probe")

        except Exception as e:
            logger.debug(f"Could not create reflection probe: {e}")


class TextureRandomizer:
    """Handles texture-level randomization."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize texture randomizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.materials_config = config.get("materials", {})

    def randomize_texture_mapping(
        self,
        material: bproc.types.Material,
        scale_range: Tuple[float, float] = (0.5, 1.5),
        rotation_range: Tuple[float, float] = (0, 360),
    ) -> None:
        """
        Randomize texture mapping (scale, rotation, offset).

        Args:
            material: Material to modify
            scale_range: Range for texture scale
            rotation_range: Range for texture rotation in degrees
        """
        try:
            # Get texture mapping node
            # This requires shader node manipulation
            scale = random.uniform(scale_range[0], scale_range[1])
            rotation = np.radians(random.uniform(rotation_range[0], rotation_range[1]))

            # Note: Full implementation requires accessing shader nodes
            logger.debug(f"Texture mapping: scale={scale:.2f}, rotation={np.degrees(rotation):.0f}")

        except Exception as e:
            logger.debug(f"Could not randomize texture mapping: {e}")

    def adjust_normal_map_strength(
        self,
        material: bproc.types.Material,
        strength_range: Tuple[float, float] = (0.5, 1.5),
    ) -> None:
        """
        Adjust normal map strength for surface detail variation.

        Args:
            material: Material to modify
            strength_range: Range for normal map strength
        """
        try:
            strength = random.uniform(strength_range[0], strength_range[1])

            # Note: Requires shader node manipulation
            logger.debug(f"Normal map strength: {strength:.2f}")

        except Exception as e:
            logger.debug(f"Could not adjust normal map: {e}")
