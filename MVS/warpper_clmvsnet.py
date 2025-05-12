from .convert_gs_to_mvs import *
from .get_mvs_model import get_mvs_model
import torch
import torch.nn.functional as F
import cv2
def get_mvs_depth(view_cam, scan=None):
    mvs_model = get_mvs_model().eval()
    view_cam = convert_dataset_dtu(view_cam, scan=scan)
    image_list = []
    proj_mat_list = []
    for view in view_cam:
        image = view.original_image.cuda() #3 h w 
        image_list.append(image)
        proj_mat_list.append(view.proj_mat)
    
    for i in range(len(view_cam)):
        print('cal {} mvs depth'.format(view_cam[i].image_name))
        image_list_new = copy.deepcopy(image_list)
        
        ref_image = image_list_new.pop(i)

        image_list_new.insert(0, ref_image)

        images = torch.stack(image_list_new)[None]#1 n 3 h w 

        proj_mat_new_list = copy.deepcopy(proj_mat_list)

        ref_proj = proj_mat_new_list.pop(i)

        proj_mat_new_list.insert(0, ref_proj)

        proj_matrices = torch.stack(proj_mat_new_list)#n 2 4 4 stage3

        stage2_pjmats = proj_matrices.clone()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2
        stage1_pjmats = proj_matrices.clone()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4

        proj_matrices_ms = {
            "stage1": stage1_pjmats[None],
            "stage2": stage2_pjmats[None],
            "stage3": proj_matrices[None] #b n 2 4 4
        }

        data = {}
        data["proj_matrices"] = proj_matrices_ms
        data["imgs"] = images #1 n 3 h w
        data["init_depth_hypotheses"] = view_cam[i].init_depth_hypotheses[None].cuda() #b numdepth
        mvs_out_dict = mvs_model(data, mode='test')
        depth = mvs_out_dict["depth"]
        photometric_confidence = mvs_out_dict["photometric_confidence"]
        h, w = images.shape[3:]
        point = depth2wpoint(depth, K=view_cam[i].K.float(), w2c=view_cam[i].w2c.float(), w=w, h=h)
        setattr(view_cam[i], 'depth', depth)
        setattr(view_cam[i], 'photometric_confidence', photometric_confidence)
        setattr(view_cam[i], 'points', point)
        setattr(view_cam[i], 'feature', mvs_out_dict["ref_fea"])
        print('mvs_features dim is {}'.format(view_cam[i].feature.shape))
    return view_cam
