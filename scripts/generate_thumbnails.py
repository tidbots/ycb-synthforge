#!/usr/bin/env python3
# BlenderProc must be imported first before any other imports
import blenderproc as bproc

"""
YCB Object Thumbnail Generator
Generates thumbnail images for all YCB objects to visually inspect mesh quality.
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_simple_scene():
    """Setup a simple scene with ground plane and lighting."""
    # Create ground plane
    ground = bproc.object.create_primitive("PLANE", scale=[2, 2, 1])
    ground.set_location([0, 0, 0])

    # Set ground material (light gray)
    mat = bproc.material.create("ground_material")
    mat.set_principled_shader_value("Base Color", [0.8, 0.8, 0.8, 1.0])
    ground.replace_materials(mat)

    return ground


def setup_lighting():
    """Setup studio-style lighting for thumbnails."""
    # Key light (main)
    key_light = bproc.types.Light()
    key_light.set_type("AREA")
    key_light.set_location([1.5, -1.5, 2.0])
    key_light.set_rotation_euler([math.radians(45), 0, math.radians(45)])
    key_light.set_energy(200)

    # Fill light (softer)
    fill_light = bproc.types.Light()
    fill_light.set_type("AREA")
    fill_light.set_location([-1.5, -1.0, 1.5])
    fill_light.set_rotation_euler([math.radians(50), 0, math.radians(-30)])
    fill_light.set_energy(100)

    # Rim light (back)
    rim_light = bproc.types.Light()
    rim_light.set_type("AREA")
    rim_light.set_location([0, 2.0, 1.5])
    rim_light.set_rotation_euler([math.radians(-60), 0, math.radians(180)])
    rim_light.set_energy(80)

    return [key_light, fill_light, rim_light]


def setup_camera_for_object(obj):
    """Setup camera to frame the object nicely."""
    # Get object bounding box
    bbox = obj.get_bound_box()
    bbox_array = np.array(bbox)

    # Calculate object center and size
    center = bbox_array.mean(axis=0)
    size = bbox_array.max(axis=0) - bbox_array.min(axis=0)
    max_dim = max(size)

    # Camera distance based on object size
    distance = max_dim * 2.5

    # Camera position (45 degree angle from front-top)
    cam_x = center[0] + distance * 0.7
    cam_y = center[1] - distance * 0.7
    cam_z = center[2] + distance * 0.5

    # Set camera
    cam_pose = bproc.math.build_transformation_mat(
        [cam_x, cam_y, cam_z],
        bproc.camera.rotation_from_forward_vec(
            center - np.array([cam_x, cam_y, cam_z])
        )
    )
    bproc.camera.add_camera_pose(cam_pose)


def render_object_thumbnail(obj_path: Path, output_path: Path, format_name: str):
    """Render a single object thumbnail."""
    try:
        # Clean up previous scene
        bproc.clean_up(clean_up_camera=True)

        # Setup scene
        ground = setup_simple_scene()
        lights = setup_lighting()

        # Load object
        objs = bproc.loader.load_obj(str(obj_path))

        if not objs:
            logger.error(f"Failed to load object from {obj_path}")
            return False

        obj = objs[0]

        # Center object on ground
        bbox = obj.get_bound_box()
        bbox_array = np.array(bbox)
        min_z = bbox_array[:, 2].min()

        # Move object so it sits on ground
        current_loc = obj.get_location()
        obj.set_location([0, 0, -min_z + 0.01])

        # Setup camera
        setup_camera_for_object(obj)

        # Render settings
        bproc.renderer.set_max_amount_of_samples(64)
        bproc.renderer.set_output_format(file_format="PNG")

        # Enable denoiser
        try:
            import bpy
            bpy.context.scene.cycles.use_denoising = True
        except Exception:
            pass

        # Render
        data = bproc.renderer.render()

        # Save image
        from PIL import Image
        colors = data["colors"][0]
        if colors.max() <= 1:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

        img = Image.fromarray(colors)
        img.save(output_path)

        return True

    except Exception as e:
        logger.error(f"Error rendering {obj_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_all_thumbnails(ycb_dir: Path, output_dir: Path):
    """Generate thumbnails for all YCB objects."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    results = []

    for obj_dir in sorted(ycb_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        obj_name = obj_dir.name

        # Check both formats
        for format_name in ['google_16k', 'tsdf']:
            obj_path = obj_dir / format_name / 'textured.obj'

            if not obj_path.exists():
                continue

            output_filename = f"{obj_name}_{format_name}.png"
            output_path = output_dir / output_filename

            logger.info(f"Rendering: {obj_name}/{format_name}...")

            success = render_object_thumbnail(obj_path, output_path, format_name)

            results.append({
                'object_name': obj_name,
                'format': format_name,
                'success': success,
                'output_path': str(output_path) if success else None
            })

            if success:
                logger.info(f"  -> Saved to {output_path}")
            else:
                logger.error(f"  -> FAILED")

    return results


def create_comparison_grid(output_dir: Path, ycb_dir: Path):
    """Create a comparison grid showing google_16k vs tsdf side by side."""
    from PIL import Image, ImageDraw, ImageFont

    # Collect all thumbnails
    thumbnails = {}
    for img_path in output_dir.glob("*.png"):
        name = img_path.stem
        if "_google_16k" in name:
            obj_name = name.replace("_google_16k", "")
            if obj_name not in thumbnails:
                thumbnails[obj_name] = {}
            thumbnails[obj_name]['google_16k'] = img_path
        elif "_tsdf" in name:
            obj_name = name.replace("_tsdf", "")
            if obj_name not in thumbnails:
                thumbnails[obj_name] = {}
            thumbnails[obj_name]['tsdf'] = img_path

    if not thumbnails:
        logger.warning("No thumbnails found for comparison grid")
        return

    # Settings
    thumb_size = (256, 256)
    padding = 10
    label_height = 30

    # Calculate grid size
    num_objects = len(thumbnails)
    cols = 2  # google_16k and tsdf
    rows = num_objects

    grid_width = cols * (thumb_size[0] + padding) + padding + 200  # Extra for labels
    grid_height = rows * (thumb_size[1] + padding + label_height) + padding + 50

    # Create grid image
    grid = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Try to load a font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_small = font

    # Header
    draw.text((210 + padding, 10), "google_16k", fill=(0, 0, 0), font=font)
    draw.text((210 + thumb_size[0] + 2*padding, 10), "tsdf", fill=(0, 0, 0), font=font)

    y = 50
    for obj_name in sorted(thumbnails.keys()):
        obj_thumbs = thumbnails[obj_name]

        # Object name label
        draw.text((padding, y + thumb_size[1]//2), obj_name[:25], fill=(0, 0, 0), font=font_small)

        x = 200 + padding

        # google_16k thumbnail
        if 'google_16k' in obj_thumbs:
            try:
                img = Image.open(obj_thumbs['google_16k'])
                img.thumbnail(thumb_size)
                grid.paste(img, (x, y))
            except Exception as e:
                draw.rectangle([x, y, x+thumb_size[0], y+thumb_size[1]], outline=(255, 0, 0))
                draw.text((x+10, y+thumb_size[1]//2), "ERROR", fill=(255, 0, 0), font=font_small)
        else:
            draw.rectangle([x, y, x+thumb_size[0], y+thumb_size[1]], outline=(200, 200, 200))
            draw.text((x+10, y+thumb_size[1]//2), "N/A", fill=(150, 150, 150), font=font_small)

        x += thumb_size[0] + padding

        # tsdf thumbnail
        if 'tsdf' in obj_thumbs:
            try:
                img = Image.open(obj_thumbs['tsdf'])
                img.thumbnail(thumb_size)
                grid.paste(img, (x, y))
            except Exception as e:
                draw.rectangle([x, y, x+thumb_size[0], y+thumb_size[1]], outline=(255, 0, 0))
                draw.text((x+10, y+thumb_size[1]//2), "ERROR", fill=(255, 0, 0), font=font_small)
        else:
            draw.rectangle([x, y, x+thumb_size[0], y+thumb_size[1]], outline=(200, 200, 200))
            draw.text((x+10, y+thumb_size[1]//2), "N/A", fill=(150, 150, 150), font=font_small)

        y += thumb_size[1] + padding + label_height

    # Save grid
    grid_path = output_dir / "comparison_grid.png"
    grid.save(grid_path)
    logger.info(f"Comparison grid saved to {grid_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate YCB object thumbnails')
    parser.add_argument('--ycb-dir', type=str,
                        default='/workspace/models/ycb',
                        help='Path to YCB models directory')
    parser.add_argument('--output', type=str,
                        default='/workspace/data/thumbnails',
                        help='Output directory for thumbnails')
    parser.add_argument('--comparison-grid', action='store_true',
                        help='Generate comparison grid after thumbnails')

    args = parser.parse_args()

    ycb_dir = Path(args.ycb_dir)
    output_dir = Path(args.output)

    if not ycb_dir.exists():
        logger.error(f"YCB directory not found: {ycb_dir}")
        sys.exit(1)

    logger.info(f"Generating thumbnails for YCB objects in: {ycb_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Initialize BlenderProc
    bproc.init()

    # Generate thumbnails
    results = generate_all_thumbnails(ycb_dir, output_dir)

    # Summary
    successful = sum(1 for r in results if r['success'])
    failed = sum(1 for r in results if not r['success'])

    logger.info(f"\nSummary: {successful} successful, {failed} failed")

    if failed > 0:
        logger.info("Failed objects:")
        for r in results:
            if not r['success']:
                logger.info(f"  - {r['object_name']}/{r['format']}")

    # Generate comparison grid
    if args.comparison_grid:
        logger.info("\nGenerating comparison grid...")
        create_comparison_grid(output_dir, ycb_dir)

    logger.info(f"\nThumbnails saved to: {output_dir}")


if __name__ == '__main__':
    main()
