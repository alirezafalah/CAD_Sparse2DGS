from MVS.CLMVSNet.networks.clmvsnet import CLMVSNet
from MVS.utils import DotDict, read_json_to_dict
import yaml
import os
import torch
import sys
current_dir = os.path.dirname(os.path.abspath(__file__)) #get abs dir path
yaml_path = os.path.join(current_dir, 'config.yaml')

def get_clmvsnet_model(name='CLMVSNet'):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    args = DotDict(config[name])
    mvs = globals()[name](args).cuda()
    print('successfully building {} model !'.format(name))
    checkpoint = torch.load(args.ckpt_path, map_location="cpu") 
    mvs.load_state_dict(checkpoint["model"])
    print('load pretrained model from {}'.format(args.ckpt_path))
    return mvs

def get_mvs_model(name=None):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    args = DotDict(config)
    if name is not None:
        args.mvs_model = name
    if args.mvs_model == "clmvsnet":
        return get_clmvsnet_model()


