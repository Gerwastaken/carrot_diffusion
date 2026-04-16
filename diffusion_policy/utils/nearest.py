# 对于第一帧，去训练集里找最相似的
from dataclasses import dataclass
import os
import json
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import collections.abc as container_abcs

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

from dataset.constants import *
from dataset.projector import Projector
from utils.transformation import rot_trans_mat, apply_mat_to_pose, apply_mat_to_pcd, xyz_rot_transform

import open3d as o3d
import pytorch3d.ops as torch3d_ops
from utils.tools import timer
from utils.pointcloud import load_point_cloud, xyz1trans
from model.common.pytorch_util import dict_apply
from dataset.realworld2 import normalize_tcp, unnormalize_tcp, normalize_tcp_deform, unnormalize_tcp_deform
from dataset.preprocess import DemoLoader, FrameData, load_cfg, aug_mat
from utils.separator import separate
from utils.visualization import vis3d

from scripts.tps3d import TPS3d
from scripts.data_deform import decode_trajectory, encode_trajectory



class NearestFinder:
    def __init__(
            self,
            source_list,
        ):
        self.source_list = source_list
        init_states_list = []
        
        for source in source_list:
            init_states = []
            data_path =  os.path.join(source, 'train')
            calib_path = os.path.join(source, "calib")
            all_demos = sorted(os.listdir(data_path))
            num_demos = len(all_demos)
            for demo_id in tqdm(range(num_demos)):
                demo_path = os.path.join(data_path, all_demos[demo_id])
                demo = DemoLoader(demo_path, [], calib_path)
                data = FrameData(demo_path, [], demo.frame_ids[0], demo.projector)
                init_states.append(data.anchor_points[0])
            init_states_list.append(np.stack(init_states))
        self.init_states_list = init_states_list

    def find_nearest(self, query_state):
        nearests = []
        for source_init_states in self.init_states_list:
            dists = np.linalg.norm(source_init_states - query_state, axis=-1).sum(-1)
            nearest_idx = np.argmin(dists)
            nearests.append(source_init_states[nearest_idx])
        return nearests
    
    def vis_nearest(self, query_state, cloud):
        nearests = self.find_nearest(query_state)
        for nearest in nearests:
            vis3d(
                cloud,
                point=nearest
            )
        return nearests