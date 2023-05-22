from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import scipy
import scipy.io
import cv2
import trimesh
from utils import dotty, P_matrix_to_rot_trans_vectors, pytorch_camera
from Render import render_mesh
from Meshes import Meshes
from utils import save_images



class SceneDataset(torch.utils.data.Dataset):

    def __init__(self,
                 img_res,
                 images_path,
                 camera_path,
                 camera_loader,
                 masks_path=None,
                 depths_path=None,
                 meshes_path=None):
        super().__init__()
        
        self.img_res = img_res
        self.images_path = images_path
        self.camera_path = camera_path
        self.masks_path = masks_path
        self.depths_path = depths_path
        self.meshes_path = meshes_path
        self.camera_loader = camera_loader
        
        self.pose = camera_loader(camera_path)
        
    