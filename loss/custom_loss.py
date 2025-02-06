import torch
import numpy as np
import torch.nn.functional as F
import time
from pytorch3d.ops import estimate_pointcloud_normals
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points, knn_gather
from utils.convert import pano_to_lidar

def chamfer_distance_low_capacity(keypoints1,keypoints2):
    '''
    kp1:B N 3
    kp2:B N 3
    '''
    dist1, idx1,_= knn_points(keypoints1, keypoints2, K=1, return_nn=False)#dist:BM1,idx:BM1
    dist2, idx2,_= knn_points(keypoints2, keypoints1, K=1, return_nn=False)
    dist=dist1.mean()+dist2.mean()
    return dist,idx1,idx2
def chamfer_based_norm_loss_low_capacity(keypoints1,keypoints2,idx1,idx2):
    norm1=estimate_pointcloud_normals(keypoints1,neighborhood_size=30)
    norm2=estimate_pointcloud_normals(keypoints2,neighborhood_size=30)
    nearst_norm1=knn_gather(norm2,idx1)
    nearst_norm2=knn_gather(norm1,idx2)
    nearst_norm1=torch.squeeze(nearst_norm1)
    nearst_norm2=torch.squeeze(nearst_norm2)

    n1=torch.norm(norm1-nearst_norm1,dim=-1)
    n2=torch.norm(norm2-nearst_norm2,dim=-1)
    n=(n1**2).mean()+(n2**2).mean()
    return n
