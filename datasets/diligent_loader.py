from multiprocessing.sharedctypes import Value
import numpy as np
import torch
import scipy
import scipy.io
import cv2
import trimesh
from utils.general import P_matrix_to_rot_trans_vectors, pytorch_camera
# from Render import render_mesh
from model.Meshes import Meshes
from utils.general import save_images
import utils.general as utils
import utils.plots as plots

def make_torch_tensor(*args, device='cpu'):
    return tuple(torch.tensor(a).to(device) for a in args)

def get_images_path(path_str, name_str, view_id):
    
    path = f'{path_str}/mvpmsData/{name_str}PNG/view_{view_id:02}/'
    images_path = utils.glob_imgs(path)
    
    images_num = len(images_path) # one for mask one for normal_gt
    
    mask_path = [f'{path}mask.png']*images_num
    
    lights_int_path = f'{path}light_intensities.txt'
    lights_dirs_path = f'{path}light_directions.txt'
    
    
    return images_path, mask_path, lights_dirs_path, lights_int_path

def read_pose(cam_path, view_id, device):
    R = scipy.io.loadmat(cam_path)[f'Rc_{view_id}']
    T = scipy.io.loadmat(cam_path)[f'Tc_{view_id}']
    K = np.zeros((3,4), dtype=np.float32)
    K[:3,:3] = scipy.io.loadmat(cam_path)['KK']
    
    P = np.eye(4, dtype=np.float32)
    P[:3,:3] = R
    P[:3,3] = T.reshape(-1)
    
    return torch.from_numpy(P).to(device), torch.from_numpy(K).to(device)

def load_diligent_mesh(mesh_path, device):
    
    mesh = trimesh.load(mesh_path)
    verts, faces = make_torch_tensor(mesh.vertices.astype(np.float32), mesh.faces, device=device)
    shift = verts.mean(0)
    scale = (verts - shift).abs().max()
    transf = torch.eye(4).to(device)
    transf[:3,:3] = torch.eye(3).to(device) * scale
    transf[:3,3] = shift

    return (verts - shift) / scale, faces, transf


class Diligent(torch.utils.data.Dataset):
    def __init__(self,
                 path_str,
                 name_str, 
                 view_num=1,
                 device='cpu') -> None:
        super().__init__()
        
        # setting of Diligent
        self.images_per_view = 96
        self.device=device
        
        self.globe_path = f'{path_str}/mvpmsData/{name_str}PNG/'
        self.mesh_path = '{0}mesh_Gt.ply'.format(self.globe_path)
        self.camera_paths=f'{self.globe_path}Calib_Results.mat'
        self.view_num = view_num
        
        self.images_paths = []
        self.masks_paths = []
        self.lights_int_path = []
        self.lights_dirs_path = []
        
        self.Ps = [] 
        
        for view_id in range(view_num):
            
            image_path, mask_path, light_int_path, light_dirs_path = get_images_path(path_str, name_str, view_id+1)
            
            self.images_paths += image_path
            self.masks_paths += (mask_path)
            self.lights_int_path.append(light_int_path)
            self.lights_dirs_path.append(light_dirs_path)
            P, K = read_pose(self.camera_paths, view_id+1, self.device)
            r, t = P_matrix_to_rot_trans_vectors(P)  # check for r, t
            self.Ps.append(P.expand((len(image_path),)+P.shape))
            
        # make images square
        K[0,2] -= 50
        self.K = K.to(device)
        
        _, _, transf = load_diligent_mesh(self.mesh_path, device=self.device)
        scale = transf[0,0]
        self.Ps = torch.cat(self.Ps,dim=0) @ transf
        self.Ps[:,:3] = self.Ps[:,:3] / scale
        self.transf = transf
        
        self.lights, self.lights_int, colocated = self.get_lights()
        
        # colocated mask
        self.images_paths = np.array(self.images_paths)[colocated > 0].tolist()
        self.masks_paths = np.array(self.masks_paths)[colocated > 0].tolist()
        self.Ps = self.Ps[colocated > 0,...]
        
        self.images_num = len(self.images_paths)
        
        
    def __len__(self):
        return self.images_num
    
    def get_item(self, index):
        
        # rgb = torch.Tensor(utils.load_rgb(self.images_paths[index])).to(self.device)
        rgb = cv2.imread(self.images_paths[index])[...,::-1].astype(np.float32)
        rgb = torch.Tensor(rgb).to(self.device)
        pose = self.Ps[index:index+1,...]
        mask = utils.load_mask(self.masks_paths[index])
        light_int = self.lights_int[index:index+1]
        rgb = rgb / light_int.reshape(light_int.shape[0],1,light_int.shape[1]) / 65535
        
        # rgb = torch.Tensor(rgb).to(self.device)
        # make image square
        rgb = rgb[:,50:-50]
        mask = mask[:,50:-50]
        # normal = normal[:,50:-50]
        
        rgb[~mask] = 0
        mask = torch.from_numpy(mask).to(self.device).float()
        
        # switch channel
        rgb_ = torch.zeros_like(rgb)
        rgb_[...,0] = rgb[...,2]
        rgb_[...,2] = rgb[...,0]
        rgb_ = rgb.to(self.device)
                
        return {
            'rgb': rgb_[None],
            'pose': pose,
            'mask': mask[None]
        }
    
    def get_lights(self):
        lights_int = []
        lights_dirs = []
        colocated_masks = []
        for i in range(self.view_num):
            P, _ = read_pose(self.camera_paths, i+1, self.device)
            lights_int_ = np.loadtxt(self.lights_int_path[i]).astype(np.float32)
            lights_dirs_ = np.loadtxt(self.lights_dirs_path[i]).astype(np.float32)
            colocated_mask = (lights_dirs_[...,-1] > 0.65)
            lights_dir = torch.from_numpy(lights_dirs_[colocated_mask]@np.diag([-1,-1,-1])).type(torch.float32).to(self.device)

            lights_dirs.append(lights_dir@P[:3,:3])
        
            lights_int.append(torch.from_numpy(lights_int_[colocated_mask]).type(torch.float32).to(self.device))
            
            colocated_masks.append(torch.from_numpy(colocated_mask).type(torch.int))
            
        return torch.cat(lights_dirs, dim=0), torch.cat(lights_int, dim=0), torch.cat(colocated_masks, dim=0)
    
    


if __name__ == '__main__':
    # import matplotlib.pyplot as plt
    
    # path_str = '/storage/group/dataset_mirrors/01_incoming/DiLiGenT-MV'
    path_str = '/usr/prakt/s0131/DNS/DiLiGenT-MV'
    name_str = 'reading'
    view_num = 3
    
    diligent_dataset = Diligent(path_str=path_str,
                                name_str=name_str,
                                view_num=view_num)
    
    
    sample = diligent_dataset.get_item(30)
    
    lights_dir, light_int = diligent_dataset.get_lights()
    
    print(len(light_int))  
    print(len(diligent_dataset))
    print(lights_dir.shape)
    
    # plots.save_img(sample['rgb'], '/home/wiss/sang/git/velocity_field/exps/test.png')
    plots.save_img(sample['rgb'], '/usr/prakt/s0131/deform_implicits/results/exps/test.png')
            
        
    
            
            

        

            
        
        