from pathlib import Path

from torch.utils.data import Dataset
import torch
import cv2
import json
from tqdm import tqdm
import os


class ObjectRelightingDataset(Dataset):
    def __init__(
        self,
        scene_dir,
        split="train",
        downsample=1.0,
    ):
        pass


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

    # Load test set: {scene_dir}.
    test_camera_paths = sorted(scene_dir.glob("gt_camera_*.txt"))
    test_im_rgb_paths = sorted(scene_dir.glob("gt_image_*.png"))
    test_im_mask_paths = sorted(scene_dir.glob("gt_mask_*.png"))
    num_test = len(test_camera_paths)
    assert num_test == 9
    assert num_test == len(test_camera_paths)
    assert num_test == len(test_im_rgb_paths)
    assert num_test == len(test_im_mask_paths)



if __name__ == "__main__":
    main()
