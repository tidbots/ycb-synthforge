#!/usr/bin/env python3
"""
Fix tsdf OBJ files to properly reference materials.

The tsdf format OBJ files are missing the 'usemtl' directive,
which causes textures not to be loaded by BlenderProc/Blender.
This script adds the missing material reference.
"""

import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_material_name_from_mtl(mtl_path: Path) -> str:
    """Extract the material name from MTL file."""
    with open(mtl_path, 'r') as f:
        for line in f:
            if line.startswith('newmtl '):
                return line.strip().split(' ', 1)[1]
    return None


def fix_obj_file(obj_path: Path, backup: bool = True) -> bool:
    """
    Fix an OBJ file by adding usemtl directive if missing.

    Args:
        obj_path: Path to the OBJ file
        backup: Whether to create a backup of the original file

    Returns:
        True if file was modified, False otherwise
    """
    mtl_path = obj_path.with_suffix('.mtl')

    if not mtl_path.exists():
        logger.warning(f"MTL file not found: {mtl_path}")
        return False

    # Get material name from MTL file
    material_name = get_material_name_from_mtl(mtl_path)
    if not material_name:
        logger.warning(f"Could not find material name in: {mtl_path}")
        return False

    # Read OBJ file
    with open(obj_path, 'r') as f:
        content = f.read()

    # Check if usemtl already exists
    if 'usemtl ' in content:
        logger.info(f"Already has usemtl: {obj_path}")
        return False

    # Find the mtllib line and insert usemtl after it
    lines = content.split('\n')
    new_lines = []
    usemtl_added = False

    for line in lines:
        new_lines.append(line)

        # Add usemtl after mtllib line
        if line.startswith('mtllib ') and not usemtl_added:
            new_lines.append(f'usemtl {material_name}')
            usemtl_added = True

    # If no mtllib found, add both at the beginning
    if not usemtl_added:
        mtl_filename = mtl_path.name
        new_lines = [f'mtllib {mtl_filename}', f'usemtl {material_name}'] + lines
        usemtl_added = True

    # Create backup if requested
    if backup:
        backup_path = obj_path.with_suffix('.obj.backup')
        if not backup_path.exists():
            shutil.copy2(obj_path, backup_path)
            logger.info(f"Backup created: {backup_path}")

    # Write modified content
    with open(obj_path, 'w') as f:
        f.write('\n'.join(new_lines))

    logger.info(f"Fixed: {obj_path} (added usemtl {material_name})")
    return True


def fix_all_tsdf_files(ycb_dir: Path, backup: bool = True) -> dict:
    """Fix all tsdf OBJ files in the YCB directory."""
    results = {
        'fixed': [],
        'already_ok': [],
        'failed': [],
        'not_found': [],
    }

    for obj_dir in sorted(ycb_dir.iterdir()):
        if not obj_dir.is_dir():
            continue

        obj_name = obj_dir.name
        tsdf_obj = obj_dir / 'tsdf' / 'textured.obj'

        if not tsdf_obj.exists():
            results['not_found'].append(obj_name)
            continue

        try:
            if fix_obj_file(tsdf_obj, backup=backup):
                results['fixed'].append(obj_name)
            else:
                results['already_ok'].append(obj_name)
        except Exception as e:
            logger.error(f"Error fixing {obj_name}: {e}")
            results['failed'].append(obj_name)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Fix tsdf OBJ files to properly reference materials'
    )
    parser.add_argument(
        '--ycb-dir',
        type=str,
        default='/workspace/models/ycb',
        help='Path to YCB models directory'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create backup files'
    )
    parser.add_argument(
        '--single',
        type=str,
        help='Fix only a single object (e.g., "001_chips_can")'
    )

    args = parser.parse_args()
    ycb_dir = Path(args.ycb_dir)

    if not ycb_dir.exists():
        logger.error(f"YCB directory not found: {ycb_dir}")
        return 1

    if args.single:
        # Fix single object
        obj_path = ycb_dir / args.single / 'tsdf' / 'textured.obj'
        if not obj_path.exists():
            logger.error(f"OBJ file not found: {obj_path}")
            return 1
        fix_obj_file(obj_path, backup=not args.no_backup)
    else:
        # Fix all
        logger.info(f"Fixing tsdf OBJ files in: {ycb_dir}")
        results = fix_all_tsdf_files(ycb_dir, backup=not args.no_backup)

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Fixed:      {len(results['fixed'])} objects")
        print(f"Already OK: {len(results['already_ok'])} objects")
        print(f"Failed:     {len(results['failed'])} objects")
        print(f"Not found:  {len(results['not_found'])} objects")

        if results['fixed']:
            print(f"\nFixed objects:")
            for name in results['fixed']:
                print(f"  - {name}")

        if results['failed']:
            print(f"\nFailed objects:")
            for name in results['failed']:
                print(f"  - {name}")

    return 0


if __name__ == '__main__':
    exit(main())
