import json
import os
from pathlib import Path
from typing import List

import camtools as ct
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataLoader.plotter import plot_cameras_and_scene_bbox


class ObjectRelightingDataset(Dataset):
    def __init__(
        self,
        scene_dir,
        split="train",
        downsample=1.0,
    ):
        pass


def read_camera_txt(camera_path: Path):
    # Must be .txt
    assert camera_path.suffix == ".txt"
    params = np.loadtxt(camera_path)
    K, R, t, (width, height, channels) = (
        params[:3],
        params[3:6],
        params[6],
        params[7].astype(int),
    )
    T = ct.convert.R_t_to_T(R, t)
    return K, T


def read_cameras_from_txts(camera_paths: List[Path]):
    cameras = [
        read_camera_txt(camera_path) for camera_path in camera_paths
    ]
    Ks = [K for K, _ in cameras]
    Ts = [T for _, T in cameras]
    return Ks, Ts


def main():
    scene_name = "antman"

    # Only the "test" folder is required.
    ord_root = Path.home() / "research" / "object-relighting-dataset"
    scene_dir = ord_root / "dataset" / scene_name / "test"

    # Load the training set: {scene_dir}/inputs.
    inputs_dir = scene_dir / "inputs"
    train_camera_paths = sorted(inputs_dir.glob("camera_*.txt"))
    train_im_rgb_paths = sorted(inputs_dir.glob("image_*.png"))
    train_im_mask_paths = sorted(inputs_dir.glob("mask_*.png"))
    num_train = len(train_camera_paths)
    assert num_train == len(train_camera_paths)
    assert num_train == len(train_im_rgb_paths)
    assert num_train == len(train_im_mask_paths)
    train_Ks, train_Ts = read_cameras_from_txts(train_camera_paths)

    # Load test set: {scene_dir}.
    test_camera_paths = sorted(scene_dir.glob("gt_camera_*.txt"))
    test_im_rgb_paths = sorted(scene_dir.glob("gt_image_*.png"))
    test_im_mask_paths = sorted(scene_dir.glob("gt_mask_*.png"))
    num_test = len(test_camera_paths)
    assert num_test == 9
    assert num_test == len(test_camera_paths)
    assert num_test == len(test_im_rgb_paths)
    assert num_test == len(test_im_mask_paths)
    test_Ks, test_Ts = read_cameras_from_txts(test_camera_paths)

    # Read bounding boxes.
    # dataset/antman/test/inputs/object_bounding_box.txt
    # xmin xmax ymin ymax zmin zmax, one value per line
    bbox_path = scene_dir / "inputs" / "object_bounding_box.txt"
    bbox = np.loadtxt(bbox_path)
    x_min, x_max, y_min, y_max, z_min, z_max = bbox

    if True:
        plot_cameras_and_scene_bbox(
            Ks = train_Ks,
            Ts = train_Ts,
            scene_bbox=np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]]),
            camera_size=0.05,
        )
        plot_cameras_and_scene_bbox(
            Ks = test_Ks,
            Ts = test_Ts,
            scene_bbox=np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]]),
            camera_size=0.05,
        )


if __name__ == "__main__":
    main()
