import os
import torch
import numpy as np

import utils.general as utils

def load_cameras(camera_path, n_images):
    camera_dict = np.load(camera_path)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    Ps = []
    Ks = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = utils.load_K_Rt_from_P(None, P)
        Ks(torch.from_numpy(intrinsics).float())
        Ps.append(torch.from_numpy(pose).float())
    
    return Ps, Ks


class DTU(torch.utils.data.Dataset):
    def __init__(self,
                 path_str,
                 name_str,
                 device='cpu')->None:
        super().__init__()

        self.device = device

        self.globe_path = f'{path_str}/scan/{name_str}/'
        self.camera_paths = f'{self.globe_path}cameras.npz'

        self.image_paths = sorted(utils.glob_imgs('{0}/image'.format(self.globe_path)))
        self.mask_paths = sorted(utils.glob_imgs('{0}/mask'.format(self.globe_path)))

        self.n_images = len(self.image_paths)

        self.Ps, Ks = load_cameras(self.camera_paths, self.n_images)
        self.K = Ks[0,...].to(self.device)

    
    def __len__(self):
        return self.images_num

    def get_item(self, index):

        rgb = utils.load_rgb(self.image_paths[index]).to(self.device)
        mask = utils.load_mask(self.mask_paths[index]).to(self.device)

        pose = self.Ps[index,...].to(self.device)

        rgb[~mask] = 0

        return{

            'rgb': rgb[None],
            'pose': pose,
            'mask': mask[None]
        }







