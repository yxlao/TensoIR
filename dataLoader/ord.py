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

from dataLoader.plotter import plot_cameras_and_scene_bbox, plot_rays

from dataLoader.ray_utils import get_ray_directions, get_rays

from matplotlib import pyplot as plt


class ORD(Dataset):
    def __init__(
            self,
            scene_dir: Path,
            split="train",
            downsample=1.0,
            light_name=None,  # Ignored
            light_rotation=None,  # Ignored
            scene_bbox=None,  # Ignored
            is_stack=None,  # Ignored
            random_test=None, # Ignored
    ):
        # Remember inputs.
        self.scene_dir = scene_dir
        self.split = split
        self.downsample = downsample

        # Read entire dataset.
        result_dict = ORD.parse_ord_dataset(self.scene_dir, self.downsample)

        # Unpack result_dict
        if self.split == "train":
            Ks = result_dict["train_Ks"]
            Ts = result_dict["train_Ts"]
            im_rgbs = result_dict["train_im_rgbs"]
            im_masks = result_dict["train_im_masks"]
        elif self.split == "test" or self.split == "val":
            Ks = result_dict["test_Ks"]
            Ts = result_dict["test_Ts"]
            im_rgbs = result_dict["test_im_rgbs"]
            im_masks = result_dict["test_im_masks"]
        else:
            raise ValueError(f"split must be train, test or val, got {split}.")

        # Use im_masks to set im_rgbs's background to white.
        im_rgbs = im_rgbs * im_masks.unsqueeze(3) + (1 - im_masks.unsqueeze(3))
        # plt.imshow(im_rgbs[0].numpy())
        # plt.show()

        num_images = len(im_rgbs)
        self.img_wh = (im_rgbs[0].shape[1], im_rgbs[0].shape[0])
        scene_bbox = result_dict["scene_bbox"]
        near_far = result_dict["near_far"]

        # Compute directions
        w, h = self.img_wh  # Pay attention to the order.
        fx = Ks[0][0, 0]
        fy = Ks[0][1, 1]
        self.directions = get_ray_directions(h, w, [fx, fy])  # (h, w, 3)
        self.directions = self.directions / torch.norm(
            self.directions, dim=-1, keepdim=True)

        # Compute rays
        self.all_rays = []
        for i in range(num_images):
            c2w = torch.linalg.inv(Ts[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)
        self.all_rays = torch.cat(self.all_rays, 0)

        # T = result_dict["train_Ts"][0].astype(np.float32)
        # c2w = torch.tensor(ct.convert.T_to_pose(T))
        # rays_o, rays_d = get_rays(self.directions, c2w)

        # All properties, some are not needed.
        # All tensors are stored in Torch on CPU
        # Below are the values from Blender mic scene.
        # - N_vis: -1
        # - all_depth: []
        # - all_light_idx: torch.Size([64000000, 1]), torch.int64
        # - all_masks: torch.Size([64000000, 1]), torch.bool
        # - all_rays: torch.Size([64000000, 6]), torch.float32
        # - all_rgbs: torch.Size([64000000, 3]), torch.float32
        # - blender2opencv: [[ 1  0  0  0]
        # -  [ 0 -1  0  0]
        # -  [ 0  0 -1  0]
        # -  [ 0  0  0  1]]
        # - center: torch.Size([1, 1, 3]), torch.float32
        # - directions: torch.Size([800, 800, 3]), torch.float32
        # - downsample: 1.0
        # - focal: 1111.1110311937682
        # - image_paths: 100 image paths
        # - img_wh: (800, 800)
        # - intrinsics: torch.Size([3, 3]), torch.float32
        # - is_stack: False
        # - meta: with keys dict_keys(['camera_angle_x', 'frames'])
        # - near_far: [2.0, 6.0]
        # - poses: torch.Size([100, 4, 4]), torch.float32
        # - proj_mat: torch.Size([100, 3, 4]), torch.float32
        # - radius: torch.Size([1, 1, 3]), torch.float32
        # - root_dir: ./data/nerf_synthetic/mic/
        # - scene_bbox: torch.Size([2, 3]), torch.float32
        # - split: train
        # - transform: ToTensor()
        # - white_bg: True
        total_num_pixels = num_images * h * w
        self.N_vis = -1
        self.all_depth = []
        self.all_light_idx = torch.zeros((total_num_pixels, 1),
                                         dtype=torch.long)
        all_masks = im_masks.reshape((total_num_pixels, -1))
        all_masks[all_masks > 0.5] = 1
        all_masks[all_masks <= 0.5] = 0
        all_masks = all_masks.bool()
        self.all_masks = all_masks
        self.all_rays = self.all_rays
        self.all_rgbs = im_rgbs.reshape((total_num_pixels, -1))
        self.blender2opencv = None
        self.scene_bbox = scene_bbox
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.directions = self.directions
        self.downsample = downsample
        self.focal = fx
        self.image_paths = None
        self.img_wh = self.img_wh
        self.intrinsics = Ks[0]
        self.is_stack = False
        self.meta = None
        self.near_far = near_far
        self.poses = torch.stack([torch.linalg.inv(T) for T in Ts]).float()
        self.proj_mat = None
        self.root_dir = None
        self.split = self.split
        self.transform = None
        self.white_bg = True

        # Visualize.
        if False:
            plot_cameras_and_scene_bbox(
                Ks=[
                    self.intrinsics.cpu().numpy()
                    for _ in range(len(self.poses))
                ],
                Ts=[
                    ct.convert.pose_to_T(pose)
                    for pose in self.poses.cpu().numpy()
                ],
                scene_bbox=self.scene_bbox.cpu().numpy(),
                camera_size=0.25,
            )
            # plot_rays(
            #     ray_os=self.all_rays[:h * w, :3].cpu().numpy(),
            #     ray_ds=self.all_rays[:h * w, 3:].cpu().numpy(),
            #     # near=self.near_far[0],
            #     # far=self.near_far[1],
            #     sample_rate=0.01,
            #     near=0.01,
            #     far=1.0,
            # )

    def __len__(self):
        """
        Returns the number of images.
        """
        if self.split == "train":
            raise NotImplementedError("In train, you should not call __len__")

        num_rays = len(self.all_rgbs)  # (len(self.meta['frames'])*h*w, 3)
        width, height = self.img_wh
        num_images = int(num_rays / (width * height))
        # assert num_images == len(self.meta['frames'])
        return num_images

    def __getitem__(self, idx):
        print(f"BlenderDataset.__getitem__(): {idx}")

        # use data in the buffers
        if self.split == 'train':
            sample = {'rays': self.all_rays[idx], 'rgbs': self.all_rgbs[idx]}
            raise NotImplementedError(
                "In train, you should not call __getitem__")

        # create data for each image separately
        else:
            width, height = self.img_wh
            wth = width * height
            num_images = self.__len__()

            # [128000000, 3] -> [200, 800 * 800, 3]
            all_rgbs = self.all_rgbs.reshape(num_images, height * width, 3)
            # [128000000, 6] -> [200, 800 * 800, 6]
            all_rays = self.all_rays.reshape(num_images, height * width, 6)
            # [128000000, 1] -> [200, 800 * 800, 1]
            all_masks = self.all_masks.reshape(num_images, height * width, 1)
            # [128000000, 1] -> [200, 800 * 800, 1]
            all_light_idx = self.all_light_idx.reshape(num_images,
                                                       height * width, 1)

            sample = {
                'img_wh': self.img_wh,  # (int, int)
                'light_idx': all_light_idx[idx].view(-1, wth,
                                                     1),  # [light_num, H*W, 1]
                'rays': all_rays[idx],  # [H*W, 6]
                'rgbs': all_rgbs[idx].view(-1, wth, 3),  # [light_num, H*W, 3]
                'rgbs_mask': all_masks[idx]  # [H*W, 1]
            }
            print(f"light_idx.shape: {sample['light_idx'].shape}")
            print(f"rays.shape     : {sample['rays'].shape}")
            print(f"rgbs.shape     : {sample['rgbs'].shape}")
            print(f"rgbs_mask.shape: {sample['rgbs_mask'].shape}")

        return sample

    @staticmethod
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

    @staticmethod
    def read_cameras_from_txts(camera_paths: List[Path]):
        cameras = [
            ORD.read_camera_txt(camera_path) for camera_path in camera_paths
        ]
        Ks = [K for K, _ in cameras]
        Ts = [T for _, T in cameras]
        return Ks, Ts

    @staticmethod
    def parse_ord_dataset(scene_dir, downsample=1.0):
        """
        Args:
            - scene_dir: 

        Return:
            - result_dict["train_Ks"]      : (num_train, 3, 3).
            - result_dict["train_Ts"]      : (num_train, 4, 4).
            - result_dict["train_im_rgbs"] : (num_train, height, width, 3).
            - result_dict["train_im_masks"]: (num_train, height, width), 0-1, float.
            - result_dict["test_Ks"]       : (num_test, 3, 3).
            - result_dict["test_Ts"]       : (num_test, 4, 4).
            - result_dict["test_im_rgbs"]  : (num_test, height, width, 3).
            - result_dict["test_im_masks"] : (num_test, height, width), 0-1, float.
            - result_dict["scene_bbox"]    : [[x_min, y_min, z_min], 
                                              [x_max, y_max, z_max]].
        """


        scene_dir = Path(scene_dir)
        if not scene_dir.is_dir():
            raise ValueError(f"scene_dir {scene_dir} is not a directory.")
        
        # Load the training set: {scene_dir}/inputs.
        inputs_dir = scene_dir / "inputs"
        if not inputs_dir.is_dir():
            raise ValueError(f"inputs_dir {inputs_dir} is not a directory.")
        train_camera_paths = sorted(inputs_dir.glob("camera_*.txt"))
        train_im_rgb_paths = sorted(inputs_dir.glob("image_*.png"))
        train_im_mask_paths = sorted(inputs_dir.glob("mask_binary_*.png"))
        num_train = len(train_camera_paths)
        assert num_train == len(train_camera_paths)
        assert num_train == len(train_im_rgb_paths)
        assert num_train == len(train_im_mask_paths)
        train_Ks, train_Ts = ORD.read_cameras_from_txts(train_camera_paths)
        # (num_train, h, w, 3)
        train_im_rgbs = np.array([ct.io.imread(p) for p in train_im_rgb_paths])
        # (num_train, 1165, 1746), float, from 0-1
        train_im_masks = np.array(
            [ct.io.imread(p) for p in train_im_mask_paths])
        train_im_masks[train_im_masks < 0.5] = 0.0
        train_im_masks[train_im_masks >= 0.5] = 1.0
        assert (train_im_masks.shape[-1] == 3)
        train_im_masks = train_im_masks[..., 0]
        print(f"Num train images: {num_train}")

        # Load test set: {scene_dir}.
        test_camera_paths = sorted(scene_dir.glob("gt_camera_*.txt"))
        test_im_rgb_paths = sorted(scene_dir.glob("gt_image_*.png"))
        test_im_mask_paths = sorted(scene_dir.glob("gt_mask_*.png"))
        num_test = len(test_camera_paths)
        assert num_test == len(test_camera_paths)
        assert num_test == len(test_im_rgb_paths)
        assert num_test == len(test_im_mask_paths)
        test_Ks, test_Ts = ORD.read_cameras_from_txts(test_camera_paths)
        # (num_test, h, w, 3)
        test_im_rgbs = np.array([ct.io.imread(p) for p in test_im_rgb_paths])
        # (num_test, 1165, 1746), float, from 0-1
        test_im_masks = np.array([ct.io.imread(p) for p in test_im_mask_paths])
        test_im_masks[test_im_masks < 0.5] = 0.0
        test_im_masks[test_im_masks >= 0.5] = 1.0
        assert (test_im_masks.shape[-1] == 3)
        test_im_masks = test_im_masks[..., 0]
        print(f"Num test images: {num_test}")

        # Downsample: changes the image and intrinsics
        if downsample != 1.0:
            assert downsample in {2.0, 4.0}

            def downsample_K(K):
                K_new = K.copy()
                K_new[0, 0] /= downsample
                K_new[1, 1] /= downsample
                K_new[0, 2] /= downsample
                K_new[1, 2] /= downsample
                return K_new

            def downsample_image(im):
                width = int(im.shape[1] / downsample)
                height = int(im.shape[0] / downsample)
                return ct.image.resize(im, shape_wh=(width, height))

            train_Ks = np.array([downsample_K(K) for K in train_Ks])
            train_im_rgbs = np.array(
                [downsample_image(im) for im in train_im_rgbs])
            train_im_masks = np.array(
                [downsample_image(im) for im in train_im_masks])
            test_Ks = np.array([downsample_K(K) for K in test_Ks])
            test_im_rgbs = np.array(
                [downsample_image(im) for im in test_im_rgbs])
            test_im_masks = np.array(
                [downsample_image(im) for im in test_im_masks])

        # Read bounding boxes.
        # dataset/antman/test/inputs/object_bounding_box.txt
        # xmin xmax ymin ymax zmin zmax, one value per line
        bbox_path = scene_dir / "inputs" / "object_bounding_box.txt"
        bbox = np.loadtxt(bbox_path)
        x_min, x_max, y_min, y_max, z_min, z_max = bbox

        # Compute min/max distance to bounding box vertex to get near far estimate.
        # Camera centers (N, 3)
        train_Cs = np.array([ct.convert.T_to_C(T) for T in train_Ts])
        bbox_vertices = np.array([
            [x_min, y_min, z_min],
            [x_min, y_min, z_max],
            [x_min, y_max, z_min],
            [x_min, y_max, z_max],
            [x_max, y_min, z_min],
            [x_max, y_min, z_max],
            [x_max, y_max, z_min],
            [x_max, y_max, z_max],
        ])
        distances = np.linalg.norm(train_Cs[:, None, :] -
                                   bbox_vertices[None, :, :],
                                   axis=-1)
        estimated_near = float(np.min(distances))
        estimated_far = float(np.max(distances))
        print(f"Estimated near: {estimated_near:.3f}, "
              f"far: {estimated_far:.3f}")

        # Give it some slacks.
        scene_bbox_estimated = np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])
        scene_bbox = np.array([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        print(f"scene_bbox_estimated: {scene_bbox_estimated}")
        print(f"scene_bbox          : {scene_bbox} (actually used)")

        # Write to result_dict
        result_dict = {}
        result_dict["train_Ks"] = torch.tensor(train_Ks).float()
        result_dict["train_Ts"] = torch.tensor(train_Ts).float()
        result_dict["train_im_rgbs"] = torch.tensor(train_im_rgbs).float()
        result_dict["train_im_masks"] = torch.tensor(train_im_masks).float()
        result_dict["test_Ks"] = torch.tensor(test_Ks).float()
        result_dict["test_Ts"] = torch.tensor(test_Ts).float()
        result_dict["test_im_rgbs"] = torch.tensor(test_im_rgbs).float()
        result_dict["test_im_masks"] = torch.tensor(test_im_masks).float()
        result_dict["scene_bbox"] = torch.tensor(scene_bbox).float()
        result_dict["near_far"] = [estimated_near, estimated_far]

        return result_dict


def main():
    scene_name = "antman"
    ord = ORD(scene_dir=scene_name)


if __name__ == "__main__":
    main()
