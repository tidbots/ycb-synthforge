"""
Camera Module for BlenderProc
Implements camera setup and effects randomization for domain randomization.
"""

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import blenderproc as bproc
import numpy as np

logger = logging.getLogger(__name__)


class CameraRandomizer:
    """Handles camera setup and effects randomization."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize camera randomizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.camera_config = config.get("camera", {})
        self.output_config = config.get("output", {})

    def setup_camera(
        self,
        target_objects: List[bproc.types.MeshObject],
        num_views: int = 1,
    ) -> None:
        """
        Setup camera with randomized parameters.

        Args:
            target_objects: Objects to focus on
            num_views: Number of camera views to create
        """
        if not target_objects:
            logger.warning("No target objects for camera setup")
            return

        # Calculate scene center from objects
        positions = [obj.get_location() for obj in target_objects]
        scene_center = np.mean(positions, axis=0)

        # Get configuration
        resolution = self.output_config.get("resolution", [640, 640])
        distance_range = self.camera_config.get("distance", [0.4, 2.0])
        elevation_range = self.camera_config.get("elevation", [10, 70])
        azimuth_range = self.camera_config.get("azimuth", [0, 360])

        # Set resolution
        bproc.camera.set_resolution(resolution[0], resolution[1])

        for _ in range(num_views):
            # Random camera position (spherical coordinates)
            distance = random.uniform(distance_range[0], distance_range[1])
            elevation = np.radians(random.uniform(elevation_range[0], elevation_range[1]))
            azimuth = np.radians(random.uniform(azimuth_range[0], azimuth_range[1]))

            # Convert to Cartesian coordinates
            cam_x = scene_center[0] + distance * math.cos(elevation) * math.cos(azimuth)
            cam_y = scene_center[1] + distance * math.cos(elevation) * math.sin(azimuth)
            cam_z = scene_center[2] + distance * math.sin(elevation)

            # Create camera pose looking at scene center
            cam_location = np.array([cam_x, cam_y, cam_z])

            # Calculate rotation to look at target
            rotation_matrix = bproc.camera.rotation_from_forward_vec(
                scene_center - cam_location,
                inplane_rot=random.uniform(-0.1, 0.1),  # Small random roll
            )

            # Create camera-to-world transformation matrix
            cam2world_matrix = bproc.math.build_transformation_mat(
                cam_location,
                rotation_matrix,
            )

            # Add camera pose
            bproc.camera.add_camera_pose(cam2world_matrix)

        # Apply camera effects
        self._apply_camera_effects()

    def _apply_camera_effects(self) -> None:
        """Apply randomized camera effects for domain randomization."""
        # Exposure
        self._apply_exposure()

        # Depth of field
        self._apply_depth_of_field()

        # Motion blur
        self._apply_motion_blur()

        # Lens distortion (applied in post-processing or compositor)
        # Note: Full lens distortion requires compositor setup

    def _apply_exposure(self) -> None:
        """Apply exposure randomization."""
        exposure_config = self.camera_config.get("exposure", {})
        ev_range = exposure_config.get("ev_range", [-1.5, 1.5])

        # Random exposure value
        ev = random.uniform(ev_range[0], ev_range[1])

        try:
            # Set exposure through Blender's color management
            # BlenderProc may not have direct exposure control
            # This would need to be done via Blender API
            import bpy
            bpy.context.scene.view_settings.exposure = ev
            logger.debug(f"Applied exposure: {ev:.2f} EV")
        except Exception as e:
            logger.debug(f"Could not apply exposure: {e}")

    def _apply_depth_of_field(self) -> None:
        """Apply depth of field randomization."""
        dof_config = self.camera_config.get("blur", {}).get("depth_of_field", {})

        if not dof_config:
            return

        # Random f-stop
        fstop_range = dof_config.get("fstop", [1.8, 11.0])
        fstop = random.uniform(fstop_range[0], fstop_range[1])

        # Only apply DOF sometimes (adds render time)
        if random.random() < 0.3:  # 30% chance
            try:
                import bpy
                camera = bpy.context.scene.camera
                if camera:
                    camera.data.dof.use_dof = True
                    camera.data.dof.aperture_fstop = fstop

                    # Focus distance (slightly random around objects)
                    focus_offset_range = dof_config.get("focus_offset", [-0.2, 0.2])
                    # Would need to calculate based on object distance
                    camera.data.dof.focus_distance = 1.0 + random.uniform(
                        focus_offset_range[0],
                        focus_offset_range[1],
                    )
                    logger.debug(f"Applied DOF: f/{fstop:.1f}")
            except Exception as e:
                logger.debug(f"Could not apply DOF: {e}")

    def _apply_motion_blur(self) -> None:
        """Apply motion blur randomization."""
        blur_config = self.camera_config.get("blur", {})
        motion_blur_range = blur_config.get("motion_blur", [0, 0.05])

        # Motion blur intensity
        blur_amount = random.uniform(motion_blur_range[0], motion_blur_range[1])

        # Only apply motion blur sometimes
        if blur_amount > 0.01 and random.random() < 0.2:  # 20% chance
            try:
                import bpy
                bpy.context.scene.render.use_motion_blur = True
                bpy.context.scene.render.motion_blur_shutter = blur_amount
                logger.debug(f"Applied motion blur: {blur_amount:.3f}")
            except Exception as e:
                logger.debug(f"Could not apply motion blur: {e}")

    def apply_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply camera sensor noise to rendered image.

        Args:
            image: Input image array (H, W, C) in [0, 255] or [0, 1]

        Returns:
            Image with added noise
        """
        noise_config = self.camera_config.get("noise", {})
        iso_values = noise_config.get("iso_values", [100, 200, 400, 800, 1600, 3200])

        # Random ISO selection
        iso = random.choice(iso_values)

        # Calculate noise level based on ISO
        # Higher ISO = more noise
        noise_level = (iso / 100.0) * 0.02  # Base noise at ISO 100

        # Normalize image to [0, 1] if needed
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
            was_uint8 = True
        else:
            image = image.astype(np.float32)
            was_uint8 = False

        # Add Gaussian noise
        noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
        noisy_image = image + noise

        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 1)

        # Convert back if needed
        if was_uint8:
            noisy_image = (noisy_image * 255).astype(np.uint8)

        logger.debug(f"Applied noise: ISO {iso}, level {noise_level:.4f}")
        return noisy_image

    def apply_white_balance_shift(self, image: np.ndarray) -> np.ndarray:
        """
        Apply white balance shift to simulate camera WB errors.

        Args:
            image: Input image array

        Returns:
            Image with white balance shift
        """
        wb_config = self.camera_config.get("white_balance", {})
        temp_offset_range = wb_config.get("temp_offset", [-1000, 1000])

        # Random temperature offset
        temp_offset = random.uniform(temp_offset_range[0], temp_offset_range[1])

        # Normalize offset to color multiplier
        # Warmer (negative offset) = more red/yellow
        # Cooler (positive offset) = more blue
        offset_normalized = temp_offset / 2000.0  # Normalize to [-0.5, 0.5]

        # Normalize image
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
            was_uint8 = True
        else:
            image = image.astype(np.float32)
            was_uint8 = False

        # Apply color shift
        if offset_normalized < 0:  # Warmer
            image[:, :, 0] *= (1 - offset_normalized * 0.3)  # Boost red
            image[:, :, 2] *= (1 + offset_normalized * 0.3)  # Reduce blue
        else:  # Cooler
            image[:, :, 0] *= (1 - offset_normalized * 0.3)  # Reduce red
            image[:, :, 2] *= (1 + offset_normalized * 0.3)  # Boost blue

        # Clip
        image = np.clip(image, 0, 1)

        if was_uint8:
            image = (image * 255).astype(np.uint8)

        logger.debug(f"Applied WB shift: {temp_offset:.0f}K")
        return image

    def apply_lens_distortion(self, image: np.ndarray) -> np.ndarray:
        """
        Apply lens distortion (barrel/pincushion).

        Args:
            image: Input image array

        Returns:
            Image with lens distortion
        """
        distortion_config = self.camera_config.get("lens_distortion", {})
        barrel_range = distortion_config.get("barrel", [-0.1, 0])
        pincushion_range = distortion_config.get("pincushion", [0, 0.1])

        # Random distortion coefficient
        if random.random() < 0.5:
            k1 = random.uniform(barrel_range[0], barrel_range[1])
        else:
            k1 = random.uniform(pincushion_range[0], pincushion_range[1])

        if abs(k1) < 0.01:
            return image  # Skip if distortion is minimal

        try:
            import cv2

            h, w = image.shape[:2]

            # Camera matrix (assuming image center as principal point)
            fx = fy = w  # Focal length approximation
            cx, cy = w / 2, h / 2
            camera_matrix = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ], dtype=np.float32)

            # Distortion coefficients [k1, k2, p1, p2, k3]
            dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)

            # Apply distortion
            distorted = cv2.undistort(image, camera_matrix, dist_coeffs)

            logger.debug(f"Applied lens distortion: k1={k1:.3f}")
            return distorted

        except ImportError:
            logger.debug("OpenCV not available for lens distortion")
            return image
        except Exception as e:
            logger.debug(f"Could not apply lens distortion: {e}")
            return image

    def apply_vignette(self, image: np.ndarray) -> np.ndarray:
        """
        Apply vignette effect (darkening at edges).

        Args:
            image: Input image array

        Returns:
            Image with vignette
        """
        vignette_config = self.camera_config.get("lens_distortion", {})
        vignette_range = vignette_config.get("vignette", [0, 0.3])

        # Random vignette strength
        strength = random.uniform(vignette_range[0], vignette_range[1])

        if strength < 0.05:
            return image  # Skip if minimal

        # Normalize image
        if image.max() > 1:
            image = image.astype(np.float32) / 255.0
            was_uint8 = True
        else:
            image = image.astype(np.float32)
            was_uint8 = False

        h, w = image.shape[:2]

        # Create vignette mask
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h / 2, w / 2

        # Distance from center (normalized)
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        dist_normalized = dist / max_dist

        # Vignette falloff
        vignette_mask = 1 - (dist_normalized ** 2) * strength
        vignette_mask = np.clip(vignette_mask, 0, 1)

        # Apply to all channels
        if len(image.shape) == 3:
            vignette_mask = vignette_mask[:, :, np.newaxis]

        image = image * vignette_mask

        if was_uint8:
            image = (image * 255).astype(np.uint8)

        logger.debug(f"Applied vignette: strength={strength:.2f}")
        return image

    def apply_all_post_effects(self, image: np.ndarray) -> np.ndarray:
        """
        Apply all post-processing camera effects.

        Args:
            image: Input rendered image

        Returns:
            Image with all effects applied
        """
        # Apply effects in order
        if random.random() < 0.5:
            image = self.apply_white_balance_shift(image)

        if random.random() < 0.3:
            image = self.apply_noise(image)

        if random.random() < 0.2:
            image = self.apply_lens_distortion(image)

        if random.random() < 0.4:
            image = self.apply_vignette(image)

        return image
