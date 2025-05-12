import os
import torch
import yaml
import numpy as np
import cv2 as cv 
from .utils import *
import copy
from PIL import Image
current_dir = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(current_dir, 'config.yaml')

def convert_dataset_dtu(view_cam=None, scan=None):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    args = DotDict(config["dataset"])

    for i, cam in enumerate(view_cam):
       
        view_k = cam.K.float().cuda() 
        view_w2c = cam.w2c.float().cuda()
        proj_mat = torch.zeros(2, 4, 4).cuda()
        proj_mat[0, :4, :4] = view_w2c
        proj_mat[1, :3, :3] = view_k

        K, w2c, dp_min, dp_max = load_cam_from_txt(os.path.join(args.path, scan, f'cam_{cam.image_name}.txt'))

        init_depth_hypotheses, depth_values = get_depth_value(dp_min, dp_max, num_depth=192)

        setattr(view_cam[i], 'init_depth_hypotheses', torch.from_numpy(init_depth_hypotheses).float().cuda())
        setattr(view_cam[i], 'depth_values', torch.from_numpy(depth_values).float().cuda())
        setattr(view_cam[i], 'proj_mat', proj_mat.float().cuda())

    return view_cam
        






        

    
