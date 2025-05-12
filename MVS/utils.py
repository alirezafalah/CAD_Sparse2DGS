import numpy as np
import re
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math
import json

class DotDict(dict):
    """A dictionary that supports dot notation access to keys."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{key}'") 

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

def depths_to_points(view, depthmap):
    c2w = (view.world_view_transform.T).inverse()
    W, H = view.image_width, view.image_height
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output

def read_json_to_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def load_cam_from_txt(filepath):

    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != '']  

    assert len(lines) == 3 + 4 + 1

    K = np.array([[float(v) for v in lines[i].split()] for i in range(3)])
    w2c = np.array([[float(v) for v in lines[i].split()] for i in range(3, 7)])
    dp_min, dp_max = map(float, lines[7].split())

    return K, w2c, dp_min, dp_max

def get_depth_value(depth_min, depth_max, num_depth=192):
    depth_interval = (depth_max - depth_min) / num_depth
    init_depth_hypotheses = np.arange(depth_min, depth_interval * (num_depth - 0.5) + depth_min, depth_interval, dtype=np.float32)
    depth_values =          np.arange(depth_min, depth_interval * (num_depth - 0.5) + depth_min, depth_interval, dtype=np.float32)
    return init_depth_hypotheses, depth_values

def depth2wpoint(depth, K, w2c, w, h):
    '''
    depth 1 h w
    K 3 3 
    w2c 4 4
    '''
    u = torch.arange(0, w, device=depth.device).unsqueeze(0).repeat(h, 1).float()# h w 
    v = torch.arange(0, h, device=depth.device).unsqueeze(1).repeat(1, w).float()# h w 
    ones = torch.ones_like(u)# h w 
    image_coor = (torch.stack([u, v, ones], dim=-1) * depth[0, :, :, None]).reshape(-1, 3)[:, :, None]  # h w 3 - > n 3 1
    cam_coor = torch.matmul(torch.inverse(K[None]), image_coor)#1 3 3 @ n 3 1-> n 3 1
    cam_coor_h = torch.cat([cam_coor, torch.ones_like(cam_coor[:, 1, None, :])], dim=1)#  n 4 1 
    world_coor_h = torch.matmul(torch.inverse(w2c[None]),  cam_coor_h)#n 4 1
    point_cloud = world_coor_h[:, :3, 0] /  world_coor_h[:, 3] # n 3

    return point_cloud