import torch,cv2
from torch.utils.data import Dataset
import json
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms as T

from dataLoader.ray_utils import *
from dataLoader.plotter import plot_cameras_and_scene_bbox
import camtools as ct


class BlenderDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=1.0, is_stack=False, N_vis=-1, **kwargs):

        self.N_vis = N_vis
        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.img_wh = (int(800/downsample),int(800/downsample))
        self.define_transforms()

        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.define_proj_mat()

        self.white_bg = True
        self.near_far = [2.0,6.0]

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.downsample=downsample

        # Print all properties.
        import ipdb; ipdb.set_trace(); pass
        all_properties = [
            "N_vis",
            "all_depth",
            "all_light_idx",
            "all_masks",
            "all_rays",
            "all_rgbs",
            "blender2opencv",
            "center",
            "directions",
            "downsample",
            "focal",
            "image_paths",
            "img_wh",
            "intrinsics",
            "is_stack",
            "meta",
            "near_far",
            "poses",
            "proj_mat",
            "radius",
            "root_dir",
            "scene_bbox",
            "split",
            "transform",
            "white_bg",
        ]
        for key in all_properties:
            try:
                val = getattr(self, key)
            except:
                val = None
            if isinstance(val, torch.Tensor):
                print(f"{key}: {val.shape}, {val.dtype}")
            elif key == "image_paths":
                print(f"{key}: {len(val)} image paths")
            elif key == "meta":
                print(f"{key}: with keys {val.keys()}")
            else:
                print(f"{key}: {val}")

        import ipdb; ipdb.set_trace(); pass


    def read_depth(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
        return depth

    def read_meta(self):

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            self.meta = json.load(f)

        w, h = self.img_wh
        self.focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x'])  # original focal length, fov -> focal
        self.focal *= self.img_wh[0] / 800  # modify focal length to match size self.img_wh


        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(h, w, [self.focal,self.focal])  # (h, w, 3)
        self.directions = self.directions / torch.norm(self.directions, dim=-1, keepdim=True)
        self.intrinsics = torch.tensor([[self.focal,0,w/2],[0,self.focal,h/2],[0,0,1]]).float()

        self.image_paths = []
        self.poses = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_masks = []
        self.all_depth = []
        self.downsample=1.0

        img_eval_interval = 1 if self.N_vis < 0 else len(self.meta['frames']) // self.N_vis
        idxs = list(range(0, len(self.meta['frames']), img_eval_interval))
        for i in tqdm(idxs, desc=f'Loading data {self.split} ({len(idxs)})'):#img_list:#

            frame = self.meta['frames'][i]
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses += [c2w]

            image_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            self.image_paths += [image_path]
            img = Image.open(image_path)

            if self.downsample!=1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img)  # (4, h, w)
            img = img.view(4, -1).permute(1, 0)  # (h*w, 4) RGBA
            img = img[:, :3] * img[:, -1:] + (1 - img[:, -1:])  # blend A to RGB
            self.all_rgbs += [img]
            img_mask = ~(img[:, -1:] == 0)
            self.all_masks += [img_mask.squeeze(0)]

            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            self.all_rays += [torch.cat([rays_o, rays_d], 1)]  # (h*w, 6)


        self.poses = torch.stack(self.poses)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames'])*h*w, 6)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames'])*h*w, 3)
            self.all_masks = torch.cat(self.all_masks, 0)  # (len(self.meta['frames'])*h*w, 1)
#             self.all_depth = torch.cat(self.all_depth, 0)  # (len(self.meta['frames'])*h*w, 3)
            self.all_light_idx = torch.zeros((*self.all_rays.shape[:-1], 1),dtype=torch.long)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)  # (len(self.meta['frames']),h*w, 6)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames']),h,w,3)
            self.all_masks = torch.stack(self.all_masks, 0).reshape(-1,*self.img_wh[::-1])  # (len(self.meta['frames']),h,w,1)
            self.all_light_idx = torch.zeros((*self.all_rays.shape[:-1], 1),dtype=torch.long).reshape(-1,*self.img_wh[::-1])

        # Try plotting with camtools
        if False:
            plot_cameras_and_scene_bbox(
                Ks=[self.intrinsics.cpu().numpy() for _ in range(len(self.poses))],
                Ts=[ct.convert.pose_to_T(pose) for pose in  self.poses.cpu().numpy()],
                scene_bbox=self.scene_bbox.cpu().numpy(),
            )
            

    def define_transforms(self):
        self.transform = T.ToTensor()

    def define_proj_mat(self):
        self.proj_mat = self.intrinsics.unsqueeze(0) @ torch.inverse(self.poses)[:,:3]

    def world2ndc(self,points,lindisp=None):
        device = points.device
        return (points - self.center.to(device)) / self.radius.to(device)

    def __len__(self):
        """
        Returns the number of images.
        """
        if self.split == "train":
            raise NotImplementedError("In train, you should not call __len__")
        
        num_rays = len(self.all_rgbs)  # (len(self.meta['frames'])*h*w, 3)
        width, height = self.img_wh
        num_images = int(num_rays / (width * height))
        return num_images
        
    
    def __getitem__(self, idx):
        print(f"BlenderDataset.__getitem__(): {idx}")

        # use data in the buffers
        if self.split == 'train':
            sample = {
                      'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]
                      }
            raise NotImplementedError("In train, you should not call __getitem__")

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
            all_light_idx = self.all_light_idx.reshape(num_images, height * width, 1)

            sample = {
                'img_wh': self.img_wh,                            # (int, int)
                'light_idx': all_light_idx[idx].view(-1, wth, 1), # [light_num, H*W, 1]
                'rays': all_rays[idx],                            # [H*W, 6]
                'rgbs': all_rgbs[idx].view(-1, wth, 3),           # [light_num, H*W, 3]
                'rgbs_mask': all_masks[idx]                       # [H*W, 1]
            }
            print(f"light_idx.shape: {sample['light_idx'].shape}")
            print(f"rays.shape     : {sample['rays'].shape}")
            print(f"rgbs.shape     : {sample['rgbs'].shape}")
            print(f"rgbs_mask.shape: {sample['rgbs_mask'].shape}")

        return sample


if __name__ == '__main__':
    dataset = BlenderDataset(datadir='../data/nerf_synthetic/lego')
    item = dataset.__getitem__(0)
    for key, value in item.items():
        if type(value) == torch.Tensor:
            print(f'key:{key} tensor.shape:{value.shape}')
        else:
            print(f'key:{key} value:{value.shape}')

    print(f'rays.shape {dataset.all_rays.shape}')  # [640000, 6]

    print(f'rgbs.shape : {dataset.all_rgbs.shape}')  # [640000, 3]
