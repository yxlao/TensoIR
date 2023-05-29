import camtools as ct
import open3d as o3d
import numpy as np


def plot_cameras_and_scene_bbox(Ks, Ts, scene_bbox, camera_size=None):
    """
    Ks: list of intrinsics, (N, 3, 3).
    Ts: list of extrinsics, (N, 4, 4).
    scene_bbox: [[x_min, y_min, z_min], [x_max, y_max, z_max]], (2, 3).
    """
    # Camera frames.
    camera_size = 0.1 if camera_size is None else camera_size
    camera_frames = ct.camera.create_camera_ray_frames(Ks,
                                                       Ts,
                                                       size=camera_size)

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

    # Visualize.
    o3d.visualization.draw_geometries([scene_bbox_frame, camera_frames])
