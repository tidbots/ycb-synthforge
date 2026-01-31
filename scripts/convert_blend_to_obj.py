import blenderproc as bproc
# Convert Blender file to OBJ format for use with BlenderProc.
import bpy
import sys
import os
from pathlib import Path

def convert_blend_to_obj(blend_path: str, output_dir: str, texture_dir: str = None):
    """Convert a .blend file to OBJ format."""

    blend_path = Path(blend_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Open the blend file
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))

    # Select all mesh objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj

    # Export to OBJ
    obj_path = output_dir / "textured.obj"
    bpy.ops.wm.obj_export(
        filepath=str(obj_path),
        export_selected_objects=True,
        export_materials=True,
        export_uv=True,
        export_normals=True,
    )

    print(f"Exported to: {obj_path}")

    # Copy texture if provided
    if texture_dir:
        texture_dir = Path(texture_dir)
        for tex_file in texture_dir.glob("*"):
            if tex_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tga']:
                import shutil
                dest = output_dir / tex_file.name
                shutil.copy(tex_file, dest)
                print(f"Copied texture: {dest}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: blenderproc run convert_blend_to_obj.py <blend_file> <output_dir> [texture_dir]")
        sys.exit(1)

    blend_path = sys.argv[1]
    output_dir = sys.argv[2]
    texture_dir = sys.argv[3] if len(sys.argv) > 3 else None

    convert_blend_to_obj(blend_path, output_dir, texture_dir)
