#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from utils.point_utils import get_image_coor_from_world_points2
from utils.general_utils import build_scaling_rotation, build_rotation
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def smooth_loss(disp, img):
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)
    return grad_disp_x.mean() + grad_disp_y.mean()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def get_fea_loss(fea1, fea2):
    #c h w
    gathered_norm1 = fea1.norm(dim=0, keepdim=True)
    gathered_norm2 = fea2.norm(dim=0, keepdim=True)

    corr = (fea1 * fea2).sum(dim=0, keepdim=True) \
               / gathered_norm1.clamp(min=1e-9) / gathered_norm2.clamp(min=1e-9) # 1 1 h w
    corr = corr.squeeze()
    corr_loss = (1 - corr).abs()
    diff_mask = (corr_loss < 0.5).float()
    sample_loss = (corr_loss * diff_mask).mean()
    return sample_loss

def get_disk_fea_loss(fea1, fea2):
    #c h w
    gathered_norm1 = fea1.norm(dim=0, keepdim=True)
    gathered_norm2 = fea2.norm(dim=0, keepdim=True)

    corr = (fea1 * fea2).sum(dim=0, keepdim=True) \
               / gathered_norm1.clamp(min=1e-9) / gathered_norm2.clamp(min=1e-9) # 1 1 h w
    corr = corr.squeeze()
    corr_loss = (1 - corr).abs()
    return corr_loss

def generate_binary_mask(n, k, device, n_views=3):
    mask = torch.zeros(n, dtype=torch.int, device=device)
    
    indices = torch.randperm(n, device=device)[:k]
    mask[indices] = 1
    mask_sig = mask.clone() > 0.5
    mask = mask[None].repeat(n_views, 1).reshape(-1) > 0.5
    
    return mask, mask_sig

def generate_2d_grid(device, k=2.12, n=7):
    linspace = torch.linspace(-k, k, n, device=device)
    x, y = torch.meshgrid(linspace, linspace, indexing='ij')
    grid = torch.stack([x, y], dim=-1).view(-1, 2)
    return grid[None]#1 k2 2

def get_disk_reg_loss(gaussians, view_ref, view_src, surf_normal, mask, patch_num=7, n_views=3):
    sample_num = patch_num**2
    rand_mask, mask_sig = generate_binary_mask(n=mask.shape[0]//n_views, k=1024*10, n_views=n_views, device=mask.device)
    mask = torch.logical_and(rand_mask, mask)
    scales = gaussians.get_scaling[mask]#n 2
    scales = torch.cat([scales, 0*torch.ones_like(scales[:,:1])], dim=-1) #n 3
    gs_rotation = build_rotation(gaussians.get_rotation[mask]) #n 3 3
    gs_normal = torch.nn.functional.normalize(gs_rotation[:, :, 2], p=2.0, dim=-1, eps=1e-12, out=None)#n 3
    rend_normal = surf_normal.reshape(3, -1).permute(1, 0)[mask_sig] # 3 h w
    normal_error = (1 - (rend_normal * gs_normal).sum(dim=-1)).mean()#[None]

    L = build_scaling_rotation(scales, gaussians._rotation[mask]) #RS #n 3 3 
    L = L[:, None, :, :]#n 1 3 3 

    sample_points2d = generate_2d_grid(scales.device, n=patch_num)#1 k2 2
    assert sample_points2d.shape[1] == sample_num
    sample_points3d = torch.cat([sample_points2d, torch.zeros_like(sample_points2d[:, :, :1])], dim=-1)#1 k 3

    mean = gaussians._xyz[mask][:, None]#.detach()#n 1 3

    samples = torch.matmul(L, sample_points3d[:, :, :, None])#1 k 1 3 @ (n k 3 3)t   n k 1 3
    samples = samples.squeeze(-1) + mean #n k 3
    samples = samples.reshape(-1, 3)#n 3 

    coor, _, _ = get_image_coor_from_world_points2(samples, view_ref, mode="scale") # 2 n
    coor = coor.permute(1, 0)[None, None]#1 1 n 2

    _, h, w = view_ref.original_image.cuda().shape

    coor_src, _, _ = get_image_coor_from_world_points2(samples, view_src, mode="scale") # 2 n
    coor_src = coor_src.permute(1, 0)[None, None]#1 1 n 2


    ref_fea = view_ref.feature
    src_fea = view_src.feature

    #coor_list.append(coor) # 2 h w 
    proj_ref = F.grid_sample(ref_fea, coor, mode='bilinear', padding_mode='zeros',
                                              align_corners=False).squeeze()# 1 8 1 n
    proj_ref = proj_ref.reshape(8, -1)

    proj_src = F.grid_sample(src_fea, coor_src, mode='bilinear', padding_mode='zeros',
                                              align_corners=False).squeeze()# 1 8 1 n
    proj_src = proj_src.reshape(8, -1)

    loss = get_disk_fea_loss(proj_ref, proj_src)
    loss = loss.reshape(-1, sample_num).mean(dim=-1)

    return loss[loss < 0.5].mean() + normal_error

def get_view_idx(views, name):
    for idx, view in enumerate(views):
        if view.image_name == name:
            return idx
        