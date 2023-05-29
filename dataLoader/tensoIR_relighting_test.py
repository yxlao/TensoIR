import os, random
import json
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from dataLoader.ray_utils import *

class tensoIR_Relighting_test(Dataset):
    def __init__(self,
                 root_dir,
                 hdr_dir,
                 split='train',
                 random_test=True,
                 N_vis=-1,
                 downsample=1.0,
                 sub=0,
                 light_rotation=['000', '045', '090', '135', '180', '225', '270', '315'],
                 light_names=["sunrise"]
                 ):
        """
        @param root_dir: str | Root path of dataset folder
        @param hdr_dir: str | Root path for HDR folder
        @param split: str | e.g. 'train' / 'test'
        @param random_test: bool | Whether to randomly select a test view and a lighting
        else [frames, h*w, 6]
        @param N_vis: int | If N_vis > 0, select N_vis frames from the dataset, else (-1) import entire dataset
        @param downsample: float | Downsample ratio for input rgb images
        """
        assert split in ['train', 'test']
        self.N_vis = N_vis
        self.root_dir = Path(root_dir)
        self.split = split
        self.split_list = [x for x in self.root_dir.iterdir() if x.stem.startswith(self.split)]
        if not random_test:
            self.split_list.sort() # to render video
        if sub > 0:
            self.split_list = self.split_list[:sub]
        self.img_wh = (int(800 / downsample), int(800 / downsample))  
        self.white_bg = True
        self.downsample = downsample
        self.transform = self.define_transforms()
        self.light_names = light_names
        self.near_far = [2.0, 6.0]  
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]]) * self.downsample
        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        # HDR configs
        self.scan = self.root_dir.stem  # Scan name e.g. 'lego', 'hotdog'

        self.light_rotation = light_rotation

        self.light_num = len(self.light_rotation)
        ## Load light data
        self.hdr_dir = Path(hdr_dir)
   
    def define_transforms(self):
        transforms = T.Compose([
            T.ToTensor(),
        ])
        return transforms


    def read_all_frames(self):
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_light_idx = []
        for idx in tqdm(range(self.__len__()), desc=f'Loading {self.split} data, view number: {self.__len__()}, rotaion number: {self.light_num}'):
            item_path = self.split_list[idx]
            item_meta_path = item_path / 'metadata.json'
            with open(item_meta_path, 'r') as f:
                meta = json.load(f)
            img_wh = (int(meta['imw'] / self.downsample), int(meta['imh'] / self.downsample))

            # Get ray directions for all pixels, same for all images (with same H, W, focal)
            focal = 0.5 * int(meta['imw']) / np.tan(0.5 * meta['cam_angle_x'])  # fov -> focal length
            focal *= img_wh[0] / meta['imw']
            directions = get_ray_directions(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
            directions = directions / torch.norm(directions, dim=-1, keepdim=True)

            cam_trans = np.array(list(map(float, meta["cam_transform_mat"].split(',')))).reshape(4, 4)
            pose = cam_trans @ self.blender2opencv
            c2w = torch.FloatTensor(pose)  # [4, 4]
            w2c = torch.linalg.inv(c2w)  # [4, 4]
            # Read ray data
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]

            for light_name_idx in range(len(self.light_names)):

                # Read RGB data
                cur_light_name = self.light_names[light_name_idx]
                relight_img_path = item_path / f'rgba_{cur_light_name}.png'
                relight_img = Image.open(relight_img_path)
                if self.downsample != 1.0:
                    relight_img = relight_img.resize(img_wh, Image.Resampling.LANCZOS)
                relight_img = self.transform(relight_img)  # [4, H, W]
                relight_img = relight_img.view(4, -1).permute(1, 0)  # [H*W, 4]
                ## Blend A to RGB
                relight_rgbs = relight_img[:, :3] * relight_img[:, -1:] + (1 - relight_img[:, -1:])  # [H*W, 3]
                ## Obtain background mask, bg = False
                relight_mask = (~(relight_img[:, -1:] == 0)).to(torch.bool)  # [H*W, 1]
                light_idx = torch.tensor(0, dtype=torch.int8).repeat((img_wh[0] * img_wh[1], 1)).to(torch.int8) # [H*W, 1], transform to in8 to save memory

                self.all_rays.append(rays)
                self.all_rgbs.append(relight_rgbs)
                self.all_masks.append(relight_mask)
                self.all_light_idx.append(light_idx)

        self.all_rays = torch.cat(self.all_rays, dim=0)  # [N*H*W, 6]
        self.all_rgbs = torch.cat(self.all_rgbs, dim=0)  # [N*H*W, 3]
        # self.all_masks = torch.cat(self.all_masks, dim=0)  # [N*H*W, 1]
        self.all_light_idx = torch.cat(self.all_light_idx, dim=0)  # [N*H*W, 1]




    def world2ndc(self, points, lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        return len(self.split_list)

    def __getitem__(self, idx):
        item_path = self.split_list[idx]

        item_meta_path = item_path / 'metadata.json'
        with open(item_meta_path, 'r') as f:
            meta = json.load(f)
        img_wh = (int(meta['imw'] / self.downsample), int(meta['imh'] / self.downsample))

        # Get ray directions for all pixels, same for all images (with same H, W, focal)
        focal = 0.5 * int(meta['imw']) / np.tan(0.5 * meta['cam_angle_x'])  # fov -> focal length
        focal *= img_wh[0] / meta['imw']
        directions = get_ray_directions(img_wh[1], img_wh[0], [focal, focal])  # [H, W, 3]
        directions = directions / torch.norm(directions, dim=-1, keepdim=True)

        cam_trans = np.array(list(map(float, meta["cam_transform_mat"].split(',')))).reshape(4, 4)
        pose = cam_trans @ self.blender2opencv
        c2w = torch.FloatTensor(pose)  # [4, 4]
        w2c = torch.linalg.inv(c2w)  # [4, 4]


        relight_rgbs_list = []
        light_idx_list = []
        for light_name_idx in range(len(self.light_names)):
            # Read RGB data
            cur_light_name = self.light_names[light_name_idx]
            relight_img_path = item_path / f'rgba_{cur_light_name}.png'
            # relight_img_path = item_path / f'rgba_city.png'
            relight_img = Image.open(relight_img_path)
            if self.downsample != 1.0:
                relight_img = relight_img.resize(img_wh, Image.Resampling.LANCZOS)
            relight_img = self.transform(relight_img)  # [4, H, W]
            relight_img = relight_img.view(4, -1).permute(1, 0)  # [H*W, 4]
            ## Blend A to RGB                                         |<- white background ->|
            relight_rgbs = relight_img[:, :3] * relight_img[:, -1:] + (1 - relight_img[:, -1:])  # [H*W, 3]
            # relight_rgbs = relight_img[:, :3] 
            light_idx = torch.tensor(0, dtype=torch.int).repeat((img_wh[0] * img_wh[1], 1)) # [H*W, 1]

            relight_rgbs_list.append(relight_rgbs)
            light_idx_list.append(light_idx)
        
        relight_rgbs = torch.stack(relight_rgbs_list, dim=0)    # [rotation_num, H*W, 3]
        light_idx = torch.stack(light_idx_list, dim=0)          # [rotation_num, H*W, 1]
        ## Obtain background mask, bg = False
        relight_mask = ~(relight_img[:, -1:] == 0)

        # Read albedo image
        albedo_path = item_path / f'albedo.png'
        albedo = Image.open(albedo_path)
        if self.downsample != 1.0:
            albedo = albedo.resize(img_wh, Image.Resampling.LANCZOS)
        albedo = self.transform(albedo)
        albedo = albedo.view(4, -1).permute(1, 0)
        ## Blend A to RGB
        albedo = albedo[:, :3] * albedo[:, -1:] + (1 - albedo[:, -1:])  # [H*W, 3]

        # Read ray data
        rays_o, rays_d = get_rays(directions, c2w)
        rays = torch.cat([rays_o, rays_d], 1)  # [H*W, 6]

        # Read normal data
        normal_path = item_path / 'normal.png'
        normal_img = Image.open(normal_path)
        normal = np.array(normal_img)[..., :3] / 255  # [H, W, 3] in range [0, 1]
        normal = (normal - 0.5) * 2.0  # [H, W, 3] in range (-1, 1)

        normal_bg = np.array([0.0, 0.0, 1.0])
        normal_alpha = np.array(normal_img)[..., [-1]] / 255  # [H, W, 1] in range [0, 1]
        normal = normal * normal_alpha + normal_bg * (1.0 - normal_alpha)  # [H, W, 3]

        normal_wihte_bg = np.array([1.0, 1.0, 1.0])
        normal_alpha = np.array(normal_img)[..., [-1]] / 255  # [H, W, 1] in range [0, 1]
        normal_white = normal * normal_alpha + normal_wihte_bg * (1.0 - normal_alpha)  # [H, W, 3]

        ## Downsample
        if self.downsample != 1.0:
            normal = cv2.resize(normal, img_wh[::-1], interpolation=cv2.INTER_NEAREST)
   
        normal = torch.FloatTensor(normal)  # [H, W, 3]
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)
        normals = normal.view(-1, 3)  # [H*W, 3]

        normal_white = torch.FloatTensor(normal_white)  # [H, W, 3]
        # normal_white = normal_white / torch.norm(normal_white, dim=-1, keepdim=True)
        normals_white = normal_white.view(-1, 3)  # [H*W, 3]

        item = {
            'img_wh': img_wh,                # (int, int)
            'light_idx': light_idx,          # [light_num, H*W, 1]
            'rgbs': relight_rgbs,            # [light_num, H*W, 3],
            'rgbs_mask': relight_mask,       # [H*W, 1]
            'albedo': albedo,                # [H*W, 3]
            'rays': rays,                    # [H*W, 6]
            'normals': normals,              # [H*W, 3],
            'normals_white': normals_white,  # [H*W, 3],
            'c2w': c2w,                      # [4, 4]
            'w2c': w2c                       # [4, 4]
        }
        return item


if __name__ == "__main__":
    from opt import config_parser

    args = config_parser()
    light_names= ['bridge','city', 'courtyard', 'forest', 'fireplace', 'forest', 'interior', 'museum', 'night', 'snow', 'square', 'studio',
                             'sunrise', 'sunset', 'tunnel']
    dataset = tensoIR_Relighting_test(
        root_dir='/home/haian/Dataset/NeRF_DATA/Eight_Rotation/hotdog_rotate/',
        hdr_dir='/home/haian/Dataset/light_probes/low_res_envmaps/',
        split='test',
        random_test=False,
        light_names=light_names,
        downsample=1.0
    )

    # Test 1: Get single item
    item = dataset.__getitem__(0)
    print(item['albedo'].shape)
    print(item['rgbs_mask'].shape)
    import ipdb; ipdb.set_trace()
    # Test 2: Iteration
    # train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=1, drop_last=True, shuffle=True)
    # train_iter = iter(train_dataloader)
    # for i in range(20):
    #     try:
    #         item = next(train_iter)
    #         print(item.keys())
    #         print(item['rays'].shape)
    #     except StopIteration:
    #         print('Start a new iteration from the dataloader')
    #         train_iter = iter(train_dataloader)

    # Test 3: Test dataset all stack
    # test_dataset = TensoRFactorDataset(
    #     root_dir='/code/MVSNeRFactor/data/nerfactor_synthesis/hotdog',
    #     hdr_dir='/code/MVSNeRFactor/data/low_res_envmaps_32_16',
    #     split='test',
    #     downsample=1.0,
    #     is_stack=True
    # )
    # print(test_dataset.all_rays.shape)  # [4, 640000, 6]
    # print(test_dataset.all_rgbs.shape)  # [4, 640000, 3]
