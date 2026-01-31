import blenderproc as bproc
# Convert FBX file to OBJ format for use with BlenderProc.
import bpy
import sys
import shutil
from pathlib import Path

def convert_fbx_to_obj(fbx_path: str, output_dir: str):
    """Convert a .fbx file to OBJ format."""

    fbx_path = Path(fbx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear default scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import FBX file
    bpy.ops.import_scene.fbx(filepath=str(fbx_path))

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

    # Copy textures from source directory
    texture_src = fbx_path.parent
    for tex_file in texture_src.glob("*"):
        if tex_file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tga']:
            dest = output_dir / tex_file.name
            shutil.copy(tex_file, dest)
            print(f"Copied texture: {dest}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: blenderproc run convert_fbx_to_obj.py <fbx_file> <output_dir>")
        sys.exit(1)

    fbx_path = sys.argv[1]
    output_dir = sys.argv[2]

    convert_fbx_to_obj(fbx_path, output_dir)
