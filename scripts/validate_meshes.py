#!/usr/bin/env python3
"""
YCB Mesh Validation Script
Validates mesh quality for all YCB objects and identifies potential issues.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Try to import trimesh, fall back to basic OBJ parsing if not available
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False
    print("Warning: trimesh not available, using basic validation only")

import numpy as np


def parse_obj_basic(obj_path: Path) -> Dict:
    """Basic OBJ file parser for validation without trimesh."""
    vertices = []
    faces = []
    normals = []

    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('v '):
                parts = line.split()[1:4]
                vertices.append([float(x) for x in parts])
            elif line.startswith('vn '):
                parts = line.split()[1:4]
                normals.append([float(x) for x in parts])
            elif line.startswith('f '):
                # Parse face indices (handle v/vt/vn format)
                parts = line.split()[1:]
                face_indices = []
                for p in parts:
                    idx = p.split('/')[0]
                    face_indices.append(int(idx) - 1)  # OBJ is 1-indexed
                faces.append(face_indices)

    return {
        'vertices': np.array(vertices) if vertices else np.array([]),
        'faces': faces,
        'normals': np.array(normals) if normals else np.array([]),
    }


def validate_mesh_basic(obj_path: Path) -> Dict:
    """Basic mesh validation without trimesh."""
    issues = []
    warnings = []
    stats = {}

    try:
        data = parse_obj_basic(obj_path)
        vertices = data['vertices']
        faces = data['faces']
        normals = data['normals']

        num_vertices = len(vertices)
        num_faces = len(faces)

        stats['num_vertices'] = num_vertices
        stats['num_faces'] = num_faces

        if num_vertices == 0:
            issues.append("No vertices found")
            return {'issues': issues, 'warnings': warnings, 'stats': stats}

        if num_faces == 0:
            issues.append("No faces found")
            return {'issues': issues, 'warnings': warnings, 'stats': stats}

        # Check for degenerate vertices (NaN or Inf)
        if np.any(np.isnan(vertices)) or np.any(np.isinf(vertices)):
            issues.append("Contains NaN or Inf vertices")

        # Calculate bounding box
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_size = bbox_max - bbox_min
        stats['bbox_size'] = bbox_size.tolist()

        # Check for extremely thin objects (might indicate mesh issues)
        min_dim = bbox_size.min()
        max_dim = bbox_size.max()
        if max_dim > 0 and min_dim / max_dim < 0.001:
            warnings.append(f"Extremely thin aspect ratio: {min_dim/max_dim:.6f}")

        # Check for duplicate vertices
        unique_vertices = np.unique(vertices, axis=0)
        num_duplicates = num_vertices - len(unique_vertices)
        if num_duplicates > num_vertices * 0.1:  # More than 10% duplicates
            warnings.append(f"Many duplicate vertices: {num_duplicates} ({100*num_duplicates/num_vertices:.1f}%)")

        # Check face validity
        max_vertex_idx = num_vertices - 1
        invalid_faces = 0
        degenerate_faces = 0

        for face in faces:
            # Check for out-of-range indices
            if any(idx < 0 or idx > max_vertex_idx for idx in face):
                invalid_faces += 1
                continue

            # Check for degenerate faces (same vertex repeated)
            if len(face) != len(set(face)):
                degenerate_faces += 1

        if invalid_faces > 0:
            issues.append(f"Invalid face indices: {invalid_faces} faces")

        if degenerate_faces > 0:
            warnings.append(f"Degenerate faces (repeated vertices): {degenerate_faces}")

        # Check normals if present
        if len(normals) > 0:
            # Check for zero-length normals
            normal_lengths = np.linalg.norm(normals, axis=1)
            zero_normals = np.sum(normal_lengths < 1e-6)
            if zero_normals > 0:
                warnings.append(f"Zero-length normals: {zero_normals}")

    except Exception as e:
        issues.append(f"Failed to parse: {str(e)}")

    return {'issues': issues, 'warnings': warnings, 'stats': stats}


def validate_mesh_trimesh(obj_path: Path) -> Dict:
    """Full mesh validation using trimesh."""
    issues = []
    warnings = []
    stats = {}

    try:
        mesh = trimesh.load(obj_path, force='mesh')

        stats['num_vertices'] = len(mesh.vertices)
        stats['num_faces'] = len(mesh.faces)
        stats['is_watertight'] = mesh.is_watertight
        stats['is_convex'] = mesh.is_convex
        stats['euler_number'] = mesh.euler_number
        stats['bbox_size'] = mesh.bounding_box.extents.tolist()

        # Check if mesh is empty
        if len(mesh.vertices) == 0:
            issues.append("Empty mesh (no vertices)")
            return {'issues': issues, 'warnings': warnings, 'stats': stats}

        if len(mesh.faces) == 0:
            issues.append("No faces in mesh")
            return {'issues': issues, 'warnings': warnings, 'stats': stats}

        # Check for degenerate faces
        if hasattr(mesh, 'degenerate_faces'):
            degenerate = mesh.degenerate_faces
            if len(degenerate) > 0:
                warnings.append(f"Degenerate faces: {len(degenerate)}")

        # Check for non-manifold edges
        if hasattr(mesh, 'edges_unique'):
            # Non-manifold edges have more than 2 adjacent faces
            edge_face_count = np.bincount(mesh.edges_unique.flatten())
            non_manifold = np.sum(edge_face_count > 2)
            if non_manifold > 0:
                warnings.append(f"Non-manifold edges: {non_manifold}")

        # Check face normals
        if hasattr(mesh, 'face_normals'):
            # Check for NaN normals (indicates degenerate faces)
            nan_normals = np.sum(np.any(np.isnan(mesh.face_normals), axis=1))
            if nan_normals > 0:
                issues.append(f"Faces with invalid normals: {nan_normals}")

        # Check aspect ratio
        extents = mesh.bounding_box.extents
        if extents.max() > 0:
            aspect_ratio = extents.min() / extents.max()
            if aspect_ratio < 0.001:
                warnings.append(f"Extreme aspect ratio: {aspect_ratio:.6f}")

        # Check for inverted faces (normals pointing inward)
        if mesh.is_watertight:
            volume = mesh.volume
            if volume < 0:
                issues.append("Inverted normals (negative volume)")

        # Check mesh volume vs surface area ratio
        if mesh.is_watertight and mesh.area > 0:
            volume = abs(mesh.volume)
            # For a sphere, V/A ratio is r/3. Very low ratio might indicate issues
            va_ratio = volume / mesh.area
            stats['volume_area_ratio'] = va_ratio

    except Exception as e:
        issues.append(f"Failed to load mesh: {str(e)}")

    return {'issues': issues, 'warnings': warnings, 'stats': stats}


def validate_all_ycb_meshes(ycb_dir: Path) -> List[Dict]:
    """Validate all YCB object meshes."""
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

            print(f"Validating: {obj_name}/{format_name}...", end=' ', flush=True)

            if HAS_TRIMESH:
                result = validate_mesh_trimesh(obj_path)
            else:
                result = validate_mesh_basic(obj_path)

            result['object_name'] = obj_name
            result['format'] = format_name
            result['path'] = str(obj_path)

            # Determine status
            if result['issues']:
                result['status'] = 'ERROR'
                print(f"ERROR: {', '.join(result['issues'])}")
            elif result['warnings']:
                result['status'] = 'WARNING'
                print(f"WARNING: {', '.join(result['warnings'])}")
            else:
                result['status'] = 'OK'
                print("OK")

            results.append(result)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Validate YCB mesh quality')
    parser.add_argument('--ycb-dir', type=str,
                        default='/workspace/models/ycb',
                        help='Path to YCB models directory')
    parser.add_argument('--output', type=str,
                        default='/workspace/data/mesh_validation_results.json',
                        help='Output JSON file path')

    args = parser.parse_args()

    ycb_dir = Path(args.ycb_dir)

    if not ycb_dir.exists():
        print(f"Error: YCB directory not found: {ycb_dir}")
        sys.exit(1)

    print(f"Validating meshes in: {ycb_dir}")
    print(f"Using {'trimesh' if HAS_TRIMESH else 'basic'} validation\n")

    results = validate_all_ycb_meshes(ycb_dir)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    errors = [r for r in results if r['status'] == 'ERROR']
    warnings = [r for r in results if r['status'] == 'WARNING']
    ok = [r for r in results if r['status'] == 'OK']

    print(f"Total meshes checked: {len(results)}")
    print(f"  OK:       {len(ok)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Errors:   {len(errors)}")

    if errors:
        print("\nMeshes with ERRORS:")
        for r in errors:
            print(f"  - {r['object_name']}/{r['format']}: {', '.join(r['issues'])}")

    if warnings:
        print("\nMeshes with WARNINGS:")
        for r in warnings:
            print(f"  - {r['object_name']}/{r['format']}: {', '.join(r['warnings'])}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed results saved to: {output_path}")

    # Return exit code based on errors
    sys.exit(1 if errors else 0)


if __name__ == '__main__':
    main()
