import os
from tqdm import tqdm
import imageio
import numpy as np

from opt import config_parser
import torch
import torch.nn as nn
from utils import visualize_depth_numpy
# ----------------------------------------
# use this if loaded checkpoint is generate from single-light or rotated multi-light setting
from models.tensoRF_rotated_lights import raw2alpha, TensorVMSplit, AlphaGridMask

# # use this if loaded checkpoint is generate from general multi-light setting
# from models.tensoRF_general_multi_lights import TensorVMSplit, AlphaGridMask
# ----------------------------------------
from dataLoader.ray_utils import safe_l2_normalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from dataLoader import dataset_dict
from models.relight_utils import *
brdf_specular = GGX_specular
# from utils import rgb_ssim, rgb_lpips
from models.relight_utils import Environment_Light
from renderer import compute_rescale_ratio_rgb


def tone_map(linear_rgbs):
    linear_rgbs = torch.clamp(linear_rgbs, min=0.0, max=1.0)
    if linear_rgbs.shape[0] > 0:
        srgbs = linear2srgb_torch(linear_rgbs)
    else:
        srgbs = linear_rgbs
    return srgbs


@torch.no_grad()
def relight(dataset, args):

    if not os.path.exists(args.ckpt):
        print('the checkpoint path for tensoIR does not exists!!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensoIR = eval(args.model_name)(**kwargs)
    tensoIR.load(ckpt)

    W, H = dataset.img_wh
    near_far = dataset.near_far

    rgb_frames_list = []
    optimized_normal_list = []

    aligned_albedo_list = []
    roughness_list = []

    envir_light = Environment_Light(args.hdrdir, light_names)

    ####
    light_rotation_idx = 0
    ####

    # TODO: Fix me with proper rescale_value:
    # - This is the code to estimate rescale_value
    #   ```
    # global_rescale_value_single, global_rescale_value_three = compute_rescale_ratio_rgb(tensoIR, dataset)
    # rescale_value = global_rescale_value_three
    # print(f"rescale_value computed with RGB (not accurate): {rescale_value}")
    #   ```
    # - For armodillo, the rescale ratio is tensor([0.1594, 0.0485, 0.0070], device='cuda:0')
    #   rescale_value = torch.tensor([0.1594, 0.0485, 0.0070], device='cuda:0')
    # - For mic, the rescale ratio computed with RGB is:
    #   rescale_value =  tensor([1.0013, 1.0013, 1.0013], device='cuda:0')
    #   Therefore, we simply use [1, 1, 1] for datasets without gt albedo.
    rescale_value = torch.tensor([1.0, 1.0, 1.0], device='cuda:0')

    relight_psnr = dict()
    relight_l_alex, relight_l_vgg, relight_ssim = dict(), dict(), dict()
    for cur_light_name in args.light_names:
        relight_psnr[f'{cur_light_name}'] = []
        relight_l_alex[f'{cur_light_name}'] = []
        relight_l_vgg[f'{cur_light_name}'] = []
        relight_ssim[f'{cur_light_name}'] = []

    for idx in tqdm(range(len(dataset)), desc="Rendering relight images"):
        relight_pred_img_with_bg, relight_pred_img_without_bg, relight_gt_img = dict(
        ), dict(), dict()
        for cur_light_name in args.light_names:
            relight_pred_img_with_bg[f'{cur_light_name}'] = []
            relight_pred_img_without_bg[f'{cur_light_name}'] = []
            relight_gt_img[f'{cur_light_name}'] = []

        cur_dir_path = os.path.join(args.geo_buffer_path,
                                    f'{dataset.split}_{idx:0>3d}')
        os.makedirs(cur_dir_path, exist_ok=True)
        item = dataset[idx]
        frame_rays = item['rays'].squeeze(0).to(device)  # [H*W, 6]
        # gt_normal = item['normals'].squeeze(0).cpu() # [H*W, 3]s
        gt_mask = item['rgbs_mask'].squeeze(0).squeeze(-1).cpu()  # [H*W]

        # gt_rgb = item['rgbs'].squeeze(0).reshape(len(light_names), H, W, 3).cpu()  # [N, H, W, 3]
        # TODO: currently gt_rgb is simply replicated from [H, W, 3] -> [N, H, W, 3]
        gt_rgb = item['rgbs'].squeeze(0).reshape(H, W, 3).cpu()
        gt_rgb = gt_rgb.unsqueeze(0).repeat(len(args.light_names), 1, 1, 1)

        # TOOD: current gt_albedo is disabled.
        # gt_albedo = item['albedo'].squeeze(0).to(device) # [H*W, 3]
        light_idx = torch.zeros(
            (frame_rays.shape[0], 1),
            dtype=torch.int).to(device).fill_(light_rotation_idx)

        chunk_idxs = torch.split(torch.arange(frame_rays.shape[0]),
                                 args.batch_size)  # choose the first light idx
        for chunk_idx in tqdm(chunk_idxs, desc="Rendering chunks"):
            with torch.enable_grad():
                rgb_chunk, depth_chunk, normal_chunk, albedo_chunk, roughness_chunk, fresnel_chunk, acc_chunk, *temp = tensoIR(
                    frame_rays[chunk_idx],
                    light_idx[chunk_idx],
                    is_train=False,
                    white_bg=True,
                    ndc_ray=False,
                    N_samples=-1)

            acc_chunk_mask = (acc_chunk > args.acc_mask_threshold)
            rays_o_chunk, rays_d_chunk = frame_rays[
                chunk_idx][:, :3], frame_rays[chunk_idx][:, 3:]
            # [bs, 3]
            surface_xyz_chunk = rays_o_chunk + depth_chunk.unsqueeze(
                -1) * rays_d_chunk
            # [surface_point_num, 3]
            masked_surface_pts = surface_xyz_chunk[acc_chunk_mask]
            # [surface_point_num, 3]
            masked_normal_chunk = normal_chunk[acc_chunk_mask]
            # [surface_point_num, 3]
            masked_albedo_chunk = albedo_chunk[acc_chunk_mask]
            # [surface_point_num, 1]
            masked_roughness_chunk = roughness_chunk[acc_chunk_mask]
            # [surface_point_num, 1]
            masked_fresnel_chunk = fresnel_chunk[acc_chunk_mask]
            # [surface_point_num, 1]
            masked_light_idx_chunk = light_idx[chunk_idx][acc_chunk_mask]

            ## Get incident light directions
            for idx, cur_light_name in enumerate(args.light_names):
                masked_light_dir, masked_light_rgb, masked_light_pdf = envir_light.sample_light(
                    cur_light_name, masked_normal_chunk.shape[0],
                    512)  # [bs, envW * envH, 3]
                surf2l = masked_light_dir  # [surface_point_num, envW * envH, 3]
                surf2c = -rays_d_chunk[
                    acc_chunk_mask]  # [surface_point_num, 3]
                surf2c = safe_l2_normalize(surf2c,
                                           dim=-1)  # [surface_point_num, 3]

                # surf2l:[surface_point_num, envW * envH, 3] *
                #        masked_normal_chunk:[surface_point_num, 3]
                # -> cosine:[surface_point_num, envW * envH]
                cosine = torch.einsum("ijk,ik->ij", surf2l,
                                      masked_normal_chunk)
                # [surface_point_num, envW * envH] mask half of the incident light that is behind the surface
                cosine_mask = (cosine > 1e-6)
                # [surface_point_num, envW * envH, 1]
                visibility = torch.zeros((*cosine_mask.shape, 1),
                                         device=device)
                # [surface_point_num, envW * envH, 3]
                masked_surface_xyz = masked_surface_pts[:, None, :].expand(
                    (*cosine_mask.shape, 3))
                # [num_of_vis_to_get, 3]
                cosine_masked_surface_pts = masked_surface_xyz[cosine_mask]
                # [num_of_vis_to_get, 3]
                cosine_masked_surf2l = surf2l[cosine_mask]
                # [num_of_vis_to_get, 1]
                cosine_masked_visibility = torch.zeros(
                    cosine_masked_surf2l.shape[0], 1, device=device)

                chunk_idxs_vis = torch.split(
                    torch.arange(cosine_masked_surface_pts.shape[0]), 100000)

                for chunk_vis_idx in chunk_idxs_vis:
                    # [chunk_size, 3]
                    chunk_surface_pts = cosine_masked_surface_pts[
                        chunk_vis_idx]
                    # [chunk_size, 3]
                    chunk_surf2light = cosine_masked_surf2l[chunk_vis_idx]
                    nerv_vis, nerfactor_vis = compute_transmittance(
                        tensoIR=tensoIR,
                        surf_pts=chunk_surface_pts,
                        light_in_dir=chunk_surf2light,
                        nSample=96,
                        vis_near=0.05,
                        vis_far=1.5)  # [chunk_size, 1]
                    if args.vis_equation == 'nerfactor':
                        cosine_masked_visibility[
                            chunk_vis_idx] = nerfactor_vis.unsqueeze(-1)
                    elif args.vis_equation == 'nerv':
                        cosine_masked_visibility[
                            chunk_vis_idx] = nerv_vis.unsqueeze(-1)
                    visibility[cosine_mask] = cosine_masked_visibility

                ## Get BRDF specs
                nlights = surf2l.shape[1]

                # relighting
                specular_relighting = brdf_specular(
                    masked_normal_chunk, surf2c, surf2l,
                    masked_roughness_chunk, masked_fresnel_chunk
                )  # [surface_point_num, envW * envH, 3]
                masked_albedo_chunk_rescaled = masked_albedo_chunk * rescale_value
                surface_brdf_relighting = masked_albedo_chunk_rescaled.unsqueeze(
                    1
                ).expand(
                    -1, nlights, -1
                ) / np.pi + specular_relighting  # [surface_point_num, envW * envH, 3]
                direct_light = masked_light_rgb
                light_rgbs = visibility * direct_light  # [bs, envW * envH, 3]
                light_pix_contrib = surface_brdf_relighting * light_rgbs * cosine[:, :,
                                                                                  None] / masked_light_pdf
                # [bs, 3]
                surface_relight_rgb_chunk = torch.mean(light_pix_contrib,
                                                       dim=1)

                ### Tonemapping
                surface_relight_rgb_chunk = torch.clamp(
                    surface_relight_rgb_chunk, min=0.0, max=1.0)
                if surface_relight_rgb_chunk.shape[0] > 0:
                    surface_relight_rgb_chunk = linear2srgb_torch(
                        surface_relight_rgb_chunk)

                # [bs, 3]
                bg_color = envir_light.get_light(cur_light_name, rays_d_chunk)
                bg_color = torch.clamp(bg_color, min=0.0, max=1.0)
                bg_color = linear2srgb_torch(bg_color)
                relight_without_bg = torch.ones_like(bg_color)
                relight_with_bg = torch.ones_like(bg_color)
                relight_without_bg[acc_chunk_mask] = surface_relight_rgb_chunk

                acc_temp = acc_chunk[..., None]
                acc_temp[acc_temp <= 0.9] = 0.0
                relight_with_bg = acc_temp * relight_without_bg + (
                    1.0 - acc_temp) * bg_color

                relight_pred_img_with_bg[cur_light_name].append(
                    relight_with_bg.detach().clone().cpu())
                relight_pred_img_without_bg[cur_light_name].append(
                    relight_without_bg.detach().clone().cpu())

        os.makedirs(os.path.join(cur_dir_path, 'relighting_with_bg'),
                    exist_ok=True)
        os.makedirs(os.path.join(cur_dir_path, 'relighting_without_bg'),
                    exist_ok=True)

        for cur_light_name in args.light_names:
            relight_map_with_bg = torch.cat(
                relight_pred_img_with_bg[cur_light_name],
                dim=0).reshape(H, W, 3).numpy()
            relight_map_without_bg = torch.cat(
                relight_pred_img_without_bg[cur_light_name],
                dim=0).reshape(H, W, 3).numpy()

            imageio.imwrite(
                os.path.join(cur_dir_path, 'relighting_with_bg',
                             f'{cur_light_name}.png'),
                (relight_map_with_bg * 255).astype('uint8'))
            imageio.imwrite(
                os.path.join(cur_dir_path, 'relighting_without_bg',
                             f'{cur_light_name}.png'),
                (relight_map_without_bg * 255).astype('uint8'))


if __name__ == "__main__":
    args = config_parser()
    print(args)
    print("*" * 80)
    print('The result will be saved in {}'.format(
        os.path.abspath(args.geo_buffer_path)))

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    torch.cuda.manual_seed_all(20211202)
    np.random.seed(20211202)

    # The following args are not defined in opt.py
    args.acc_mask_threshold = 0.5
    args.if_render_normal = False
    args.vis_equation = 'nerv'

    dataset = dataset_dict[args.dataset_name]

    # names of the environment maps used for relighting
    light_names = [
        "gt_env_512_rotated_0000",
        # "gt_env_512_rotated_0001",
        # "gt_env_512_rotated_0002",
        # "gt_env_512_rotated_0003",
        # "gt_env_512_rotated_0004",
        # "gt_env_512_rotated_0005",
        # "gt_env_512_rotated_0006",
        # "gt_env_512_rotated_0007",
        "gt_env_512_rotated_0008",
    ]

    args.light_names = light_names

    # test_dataset = dataset(args.datadir,
    #                        args.hdrdir,
    #                        split='test',
    #                        random_test=False,
    #                        downsample=args.downsample_test,
    #                        light_names=light_names,
    #                        light_rotation=args.light_rotation)
    test_dataset = dataset(args.datadir,
                           split='test',
                           random_test=False,
                           downsample=args.downsample_test,
                           light_rotation=args.light_rotation)
    relight(test_dataset, args)