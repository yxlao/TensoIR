import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import camtools as ct
import numpy as np
import open3d as o3d

# - _geometries_cache_path will be set by get_geometries_cache_path() in the
#   first run. Example of the path: /tmp/open3d_lf05831.pkl
# - Every time get_geometries_cache_path() is called, it will return the same
#   value, this value will be persisted during the whole execution of the
#   program.
# - Tempfiles will be stored in the /tmp folder, and it will NOT be be cleared.
#   Typically, the /tmp folder will be cleared by the OS in the next reboot.
# - If you want to clear the cache manually, run: rm -rf /tmp/open3d_*.pkl
_geometries_cache_path = None


def plot_geometries(
    geometries: List[o3d.geometry.Geometry],
    load_cache: bool = True,
    update_cache: bool = True,
) -> None:
    """
    Plot and cache geometries.
    """
    # Handle cache.
    if load_cache:
        geometries = load_geometries(get_geometries_cache_path()) + geometries
    if update_cache:
        save_geometries(get_geometries_cache_path(), geometries)

    o3d.visualization.draw_geometries(geometries)


def get_geometries_cache_path() -> Path:
    global _geometries_cache_path
    if _geometries_cache_path is None:
        _geometries_cache_path = Path(
            tempfile.NamedTemporaryFile(delete=False,
                                        prefix="open3d",
                                        suffix=".pkl").name)
        save_geometries(_geometries_cache_path, [])
        print(f"[plotter] Geometries cache created: {_geometries_cache_path}")
    return _geometries_cache_path


def save_geometries(path: Path,
                    geometries: List[o3d.geometry.Geometry]) -> None:
    """
    Save geometries to a file using pickle.

    PointCloud  : points, colors, normals
    TriangleMesh: vertices, triangles, vertex_colors, vertex_normals, triangle_normals
    LineSet     : points, lines, colors
    """
    # data = [
    #     {"type": "PointCloud", "points": xxx, "colors": xxx, "normals": xxx},
    #     {"type": "TriangleMesh", "vertices": xxx, "triangles": xxx, "vertex_colors": xxx, ...},
    #     ...
    # ]
    data = []
    for geometry in geometries:
        if isinstance(geometry, o3d.geometry.PointCloud):
            data.append({
                "type": "PointCloud",
                "points": np.asarray(geometry.points),
                "colors": np.asarray(geometry.colors),
                "normals": np.asarray(geometry.normals),
            })
        elif isinstance(geometry, o3d.geometry.TriangleMesh):
            data.append({
                "type":
                "TriangleMesh",
                "vertices":
                np.asarray(geometry.vertices),
                "triangles":
                np.asarray(geometry.triangles),
                "vertex_colors":
                np.asarray(geometry.vertex_colors),
                "vertex_normals":
                np.asarray(geometry.vertex_normals),
                "triangle_normals":
                np.asarray(geometry.triangle_normals),
            })
        elif isinstance(geometry, o3d.geometry.LineSet):
            data.append({
                "type": "LineSet",
                "points": np.asarray(geometry.points),
                "lines": np.asarray(geometry.lines),
                "colors": np.asarray(geometry.colors),
            })
        else:
            raise NotImplementedError("Unsupported geometry type.")

    with open(path, "wb") as f:
        pickle.dump(data, f)

    print(f"[plotter] Saved {len(data)} geometries to {path}")


def load_geometries(path: Path) -> List[o3d.geometry.Geometry]:
    """
    Load geometries from a file using pickle.
    """
    with open(path, "rb") as f:
        data = pickle.load(f)

    geometries = []
    for item in data:
        if item["type"] == "PointCloud":
            geometry = o3d.geometry.PointCloud()
            geometry.points = o3d.utility.Vector3dVector(item["points"])
            geometry.colors = o3d.utility.Vector3dVector(item["colors"])
            geometry.normals = o3d.utility.Vector3dVector(item["normals"])
        elif item["type"] == "TriangleMesh":
            geometry = o3d.geometry.TriangleMesh()
            geometry.vertices = o3d.utility.Vector3dVector(item["vertices"])
            geometry.triangles = o3d.utility.Vector3iVector(item["triangles"])
            geometry.vertex_colors = o3d.utility.Vector3dVector(
                item["vertex_colors"])
            geometry.vertex_normals = o3d.utility.Vector3dVector(
                item["vertex_normals"])
            geometry.triangle_normals = o3d.utility.Vector3dVector(
                item["triangle_normals"])
        elif item["type"] == "LineSet":
            geometry = o3d.geometry.LineSet()
            geometry.points = o3d.utility.Vector3dVector(item["points"])
            geometry.lines = o3d.utility.Vector2iVector(item["lines"])
            geometry.colors = o3d.utility.Vector3dVector(item["colors"])
        else:
            raise NotImplementedError("Unsupported geometry type.")

        geometries.append(geometry)

    print(f"[plotter] Loaded {len(geometries)} geometries from {path}")

    return geometries


def plot_cameras_and_scene_bbox(
    Ks,
    Ts,
    scene_bbox,
    mesh=None,
    camera_size=None,
    load_cache: bool = True,
    update_cache: bool = True,
):
    """
    Ks: list of intrinsics, (N, 3, 3).
    Ts: list of extrinsics, (N, 4, 4).
    scene_bbox: [[x_min, y_min, z_min], [x_max, y_max, z_max]], (2, 3).
    """
    # Camera frames.
    camera_size = 0.1 if camera_size is None else camera_size
    camera_frames = ct.camera.create_camera_frames(Ks,
                                                       Ts,
                                                       size=camera_size,
                                                       center_line=False)

    # Scene box frames.
    x_min, y_min, z_min = scene_bbox[0]
    x_max, y_max, z_max = scene_bbox[1]
    scene_bbox_points = np.array([
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max],
    ])
    scene_bbox_lines = np.array([
        [0, 1],
        [0, 2],
        [0, 4],
        [1, 3],
        [1, 5],
        [2, 3],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ])
    scene_bbox_frame = o3d.geometry.LineSet()
    scene_bbox_frame.points = o3d.utility.Vector3dVector(scene_bbox_points)
    scene_bbox_frame.lines = o3d.utility.Vector2iVector(scene_bbox_lines)
    scene_bbox_frame.colors = o3d.utility.Vector3dVector(
        np.array([[1, 0, 0]] * len(scene_bbox_lines)))

    geometries = [scene_bbox_frame, camera_frames]

    if mesh is not None:
        mesh.compute_vertex_normals()
        geometries.append(mesh)

    # Handle cache.
    if load_cache:
        geometries = load_geometries(get_geometries_cache_path()) + geometries
    if update_cache:
        save_geometries(get_geometries_cache_path(), geometries)

    o3d.visualization.draw_geometries(geometries)


def plot_rays(ray_os,
              ray_ds,
              near,
              far,
              sample_rate=0.001,
              load_cache: bool = True,
              update_cache: bool = True):
    """
    ray_os: (N, 3).
    ray_ds: (N, 3).
    """
    num_samples = int(len(ray_os) * sample_rate)

    # Sample evenly
    sample_indices = np.linspace(0, len(ray_os) - 1, num_samples).astype(int)
    ray_os = ray_os[sample_indices]
    ray_ds = ray_ds[sample_indices]

    ls = o3d.geometry.LineSet()
    src_points = ray_os + ray_ds * near
    dst_points = ray_os + ray_ds * far
    all_points = np.concatenate([src_points, dst_points], axis=0)
    all_lines = np.array([[i, i + len(ray_os)] for i in range(len(ray_os))])
    ls.points = o3d.utility.Vector3dVector(all_points)
    ls.lines = o3d.utility.Vector2iVector(all_lines)
    ls.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0]] * len(all_lines)))

    geometries = [ls]

    # Handle cache.
    if load_cache:
        geometries = load_geometries(get_geometries_cache_path()) + geometries
    if update_cache:
        save_geometries(get_geometries_cache_path(), geometries)

    o3d.visualization.draw_geometries(geometries)
