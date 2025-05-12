import numpy as np
import os
import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm
import cv2
import torch
from sklearn.decomposition import PCA
from plyfile import PlyData, PlyElement

def save_point_cloud_to_ply(points_tensor, filename='point_cloud.ply'):
    points_tensor = points_tensor.detach().cpu()
   
    if points_tensor.dim() != 2 or points_tensor.size(1) != 3:
        raise ValueError("points_tensor must n x 3 ")
    
    points_np = points_tensor.numpy()

    vertices = np.array([tuple(point) for point in points_np],
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

    vertex_element = PlyElement.describe(vertices, 'vertex')

    ply_data = PlyData([vertex_element])
    ply_data.write(filename)

def visualize_depth(depth, mask=None, depth_min=None, depth_max=None, direct=False):
  """Visualize the depth map with colormap.
     Rescales the values so that depth_min and depth_max map to 0 and 1,
     respectively.
  """
  if not direct:
      depth = 1.0 / (depth + 1e-6)
  invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
  if mask is not None:
      invalid_mask += np.logical_not(mask)
  if depth_min is None:
      depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
  if depth_max is None:
      depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
  depth[depth < depth_min] = depth_min
  depth[depth > depth_max] = depth_max
  depth[invalid_mask] = depth_max
  depth_scaled = (depth - depth_min) / (depth_max - depth_min)
  depth_scaled_uint8 = np.uint8(depth_scaled * 255)
  depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
  depth_color[invalid_mask, :] = 0
  depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
  return depth_color

def visualize_feature_map(feature_map):
    batch_size, channels, height, width = feature_map.shape
    feature_map_np = feature_map.detach().cpu().numpy().reshape(channels, height * width).T
    pca = PCA(n_components=3)
    feature_map_pca = pca.fit_transform(feature_map_np)
    feature_map_pca_torch = torch.tensor(feature_map_pca.T.reshape(3, height, width))
    feature_map_pca_torch = (feature_map_pca_torch - feature_map_pca_torch.min()) / (feature_map_pca_torch.max() - feature_map_pca_torch.min())
    return feature_map_pca_torch