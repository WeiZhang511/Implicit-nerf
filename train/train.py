import pytorch3d
import torch
import numpy as np
from tqdm import tqdm
import h5py
import trimesh
from itertools import chain
import cv2
import os
import argparse
from pyhocon import ConfigFactory
from datetime import datetime



# from model.Synthetic import synthesize_imgs, random_RT

from torch.nn.functional import mse_loss as mse
from model.loss import velocity_loss, clipped_mae, chamfer_3d, level_set_loss, sdf_value_loss, eikonal_loss
from model.render import render_mesh
from model.model import MLP, PositionEncoding, ResNet, Sequential, ShapeNet, BRDFNet
import utils.plots as plots

from utils.general import manual_seed, rand_ico_sphere, save_models, load_models
from model.meshes import Meshes, Pointclouds
from utils.general import dotty, sample_lights_from_equirectangular_image, save_images, random_dirs, P_matrix_to_rot_trans_vectors, pytorch_camera, compile_video
from eval.diligent_eval import diligent_eval_chamfer
from utils.general import mkdir_ifnotexists

@torch.no_grad()
def sample_mesh(shape_net, brdf_net, conf, init_mesh=None, normal_net=None):
    params = dotty(params)
    if init_mesh is None:
        init_mesh = rand_ico_sphere(conf.get_config('sampling.ico_sphere_level'), device=device)
            
    s = init_mesh.verts_packed()

    x_arr, v_arr, dv_ds_arr, d2v_d2s_arr = shape_net(s, 0)
    theta_x = brdf_net(s)

    x = x_arr[-1]

    faces = init_mesh.faces_packed()
    mesh = Meshes(verts=[x], faces=[faces], vert_textures=[theta_x])

    return mesh

def clamp_vertex_grad(grad, thre):
    ret = grad + 0
    ret[torch.logical_not(torch.abs(ret) < thre)] = 0
    return ret


def train(dataset, shape_net, brdf_net, optimizer, conf,  call_back=None, init_mesh=None,  camera_settings=None, camera_settings_silhoutte=None):
    
    device = conf.get_config('settings')['device']
    if torch.cuda.is_available() and (device == 'cuda'):
        device = 'cuda'
        print('cuda is availiable, run in gpu.')
    else:
        device = 'cpu'
        print('cuda is not availiable, run in cpu.')
        
    n_images = len(dataset)
    n_iterations = conf.get_config('training')['n_iterations']
    loss_conf = conf['loss']


    plot_path = conf.get_config('settings')['plot_dir']
    exp_path = os.path.join(plot_path,conf.get_config('settings')['expname'])
    mkdir_ifnotexists(exp_path)
    timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
    mkdir_ifnotexists(os.path.join(exp_path, timestamp))
    plot_dir = os.path.join(exp_path, timestamp)


    null_init = init_mesh is None

    def closure(epoch):
        #################################
        ## sample mesh from neural nets
        nonlocal init_mesh
        if null_init:
            # init mesh from implicit surface
            ico_sphere = plots.generate_mesh(shape_net.implicit_mlp, resolution=100)
            vertices = torch.from_numpy(ico_sphere.vertices.view(np.ndarray)).type(torch.float32).to(device)
            faces = torch.from_numpy(ico_sphere.faces.view(np.ndarray)).type(torch.float32).to(device)
            n_verts = vertices.shape[0]

            # init mesh from iso sphere
            # init_mesh = rand_ico_sphere(params['sampling.ico_sphere_level'], device=device)
            # n_verts = init_mesh.verts_packed().shape[0]

            init_mesh = Meshes(verts=[vertices], faces=[faces])
        
            # s = pytorch3d.ops.sample_points_from_meshes(init_mesh, n_verts).reshape(n_verts, 3)
        
        s = init_mesh.verts_packed()

        if (loss_conf['lambda_velocity']==0) or (loss_conf['alpha'] == 0) or conf.get_config('training')['compute_velocity_seperately']:
            x_arr, v_arr, dv_ds_arr, d2v_d2s_arr, phi, phi_t, grad_phi = shape_net(s, 1)
        else:
            x_arr, v_arr, dv_ds_arr, d2v_d2s_arr, phi, phi_t, grad_phi = shape_net(s, 2)

        x = x_arr[-1]

        if conf.get_config('training')['vertex_grad_clip'] is not None:
            hook = x.register_hook(lambda grad: clamp_vertex_grad(grad, conf.get_config('training')['vertex_grad_clip']))

        theta_x = brdf_net(s)
        
        faces = init_mesh.faces_packed()


        # mesh = Meshes(verts=[x], faces=[faces], vert_textures=[theta_x])
        mesh = Pointclouds(points=[x], normals=[grad_phi[-1]], features=[theta_x])

        #################################
        ## render images with mesh
        ## and compute losses
        batch_idx = torch.randperm(n_images)[:conf.get_config('training')['n_image_per_batch']]
        # print('sampled images:', batch_idx)
        loss_image, loss_silhouette, loss_velocity, loss_level_set, loss_sdf, loss_eikonal = 0, 0, 0, 0, 0, 0
        visual_ind = 0
        for i in batch_idx:

            sample = dataset.get_item(i)
            gt_image = sample['rgb']
            gt_silhouette = sample['mask']
            rotations = sample['pose'][:,:3,:3]
            translations = sample['pose'][:,:3, 3]
            light_dirs, light_ints, _ = dataset.get_lights()

            light_pose = None
            if light_dirs is not None:
                light_pose = light_dirs[i:i+1]
            if loss_conf['lambda_image'] != 0:
                prd_image = render_mesh(mesh, 
                        modes='image_ct', #######
                        rotations=rotations, 
                        translations=translations, 
                        image_size=conf['rendering.image_size'], 
                        blur_radius=conf['rendering.rgb.blur_radius'], 
                        faces_per_pixel=conf['rendering.rgb.points_per_pixel'], 
                        device=device, background_colors=None, light_poses=light_pose, materials=None, camera_settings=camera_settings,
                        sigma=conf['rendering.rgb.sigma'], gamma=conf['rendering.rgb.gamma'])
               
                print('render image ', i)
                max_intensity = conf['rendering.rgb.max_intensity'] #* (np.random.rand()+1)

                loss_tmp = clipped_mae(gt_image.clamp_max(max_intensity), prd_image, max_intensity) / conf['training.n_image_per_batch']
                
                
                (loss_tmp * loss_conf['lambda_image']).backward(retain_graph=True)
                loss_image += loss_tmp.detach()

            if loss_conf['lambda_silhouette'] != 0:
                prd_silhouette = render_mesh(mesh, 
                        modes='silhouette', 
                        rotations=rotations, 
                        translations=translations, 
                        image_size=conf['rendering.image_size'], 
                        blur_radius=conf['rendering.silhouette.blur_radius'], 
                        faces_per_pixel=conf['rendering.silhouette.points_per_pixel'], 
                        device=device, background_colors=None, light_poses=None, materials=None, camera_settings=camera_settings_silhoutte,
                        sigma=conf['rendering.silhouette.sigma'], gamma=conf['rendering.silhouette.gamma'])

                loss_tmp = mse(gt_silhouette, prd_silhouette[...,0]) / conf['training.n_image_per_batch']
                (loss_tmp * loss_conf['lambda_silhouette']).backward(retain_graph=True)
                loss_silhouette += loss_tmp.detach()

            
            if epoch % conf['settings.plot_freq'] == 0 and visual_ind < 5:
                plots.plot_images(prd_image, gt_image, plot_dir, epoch, i)
                visual_ind += 1

        if loss_conf['lambda_velocity'] == 0:
            pass
        elif (loss_conf['alpha'] == 0) or (not conf['training.compute_velocity_seperately']):
            loss_tmp = velocity_loss(v_arr, d2v_d2s_arr, loss_conf['alpha'])
            (loss_tmp * loss_conf['lambda_velocity']).backward(retain_graph=True)
            loss_velocity = loss_tmp.detach()
            if loss_conf['velocity_sdf'] != 0:
                loss_level_set = level_set_loss(phi_t, grad_phi, v_arr)
                (loss_level_set * loss_conf['velocity_sdf']).backward(retain_graph=True)
            if loss_conf['sdf_loss']!=0:
                loss_sdf = sdf_value_loss(phi)
                (loss_sdf * loss_conf['sdf_loss']).backward(retain_graph=True)
            if loss_conf['eikonal_loss'] != 0:
                loss_eikonal = eikonal_loss(grad_phi)
                (loss_eikonal * loss_conf['eikonal_loss']).backward(retain_graph=True)

        else:
            if init_mesh is not None:
                n_verts = ico_sphere.verts_packed().shape[0]
                s = pytorch3d.ops.sample_points_from_meshes(init_mesh, n_verts).reshape(n_verts, 3)
            else:
                s = ico_sphere.verts_packed()
            n_pts_total = s.shape[0]
            for _s in torch.split(s, conf['training.n_pts_per_split'], dim=0):
                n_pts = _s.shape[0]
                _x_arr, _v_arr, _dv_ds_arr, _d2v_d2s_arr, phi, phi_t, grad_phi = shape_net(_s, 2)
                loss_tmp = velocity_loss(_v_arr, _d2v_d2s_arr, loss_conf['alpha']) * n_pts / n_pts_total
                (loss_tmp * loss_conf['lambda_velocity']).backward(retain_graph=True)
                loss_velocity += loss_tmp.detach()

                loss_level_set_tmp = level_set_loss(phi_t, grad_phi, v_arr)
                (loss_level_set_tmp * loss_conf['velocity_sdf']).backward(retain_graph=True)
                loss_level_set += loss_level_set_tmp.detach()

                loss_sdf_tmp = sdf_value_loss(phi)
                (loss_sdf_tmp * loss_conf['sdf_loss']).backward(retain_graph=True)
                loss_sdf += loss_sdf_tmp.detach()

                loss_eikonal_tmp = eikonal_loss(grad_phi)
                (loss_eikonal_tmp * loss_conf['eikonal_loss']).backward(retain_graph=False)
                loss_eikonal += loss_eikonal_tmp.detach()
                
        if loss_conf['loss.lambda_edge'] != 0:
            loss_edge = pytorch3d.loss.mesh_edge_loss(mesh)
            (loss_edge * loss_conf['loss.lambda_edge']).backward(retain_graph=True)
        else:
            loss_edge = 0

        if loss_conf['loss.lambda_normal_consistency'] != 0:
            loss_normal_consistency = pytorch3d.loss.mesh_normal_consistency(mesh)
            (loss_normal_consistency * loss_conf['loss.lambda_normal_consistency']).backward(retain_graph=True)
        else:
            loss_normal_consistency = 0

        if loss_conf['loss.lambda_laplacian_smoothing'] != 0:
            loss_laplacian_smoothing = pytorch3d.loss.mesh_laplacian_smoothing(mesh)
            (loss_laplacian_smoothing * loss_conf['loss.lambda_laplacian_smoothing']).backward(retain_graph=True)
        else:
            loss_laplacian_smoothing = 0



        loss =  loss_image * loss_conf['lambda_image'] + \
                loss_silhouette * loss_conf['lambda_silhouette'] + \
                loss_velocity * loss_conf['lambda_velocity']  + \
                loss_level_set * loss_conf['velocity_sdf'] + \
                loss_sdf * loss_conf['sdf_loss'] + \
                loss_eikonal * loss_conf['eikonal_loss'] + \
                loss_normal_consistency * loss_conf['loss.lambda_normal_consistency']+ \
                loss_laplacian_smoothing * loss_conf['loss.lambda_laplacian_smoothing']

        return mesh.detach(), (float(loss), float(loss_image), float(loss_silhouette), float(loss_velocity), float(loss_level_set), float(loss_sdf), float(loss_eikonal), float(loss_normal_consistency), float(loss_laplacian_smoothing))
    
    # pbar = tqdm(range(n_iterations))

    # for N_IT in pbar:
    for epoch in range(n_iterations):
        optimizer.zero_grad()
        mesh, losses = closure(epoch)

        optimizer.step()
        # pbar.set_description('|'.join(f'{l:.2e}' for l in losses).replace('e', ''].replace('|', ' || ', 1))
        print('({0}/{1}): loss = {2}, rgb_loss = {3}, silhouette_loss = {4}, velocity_loss = {5}, level_set_loss = {6}, eikonal_loss = {7}, sdf_loss = {8}, normal_consistency: {9}, laplacian_smoothing: {10}'.format( epoch, n_iterations, losses[0],
                                losses[1],
                                losses[2],
                                losses[3],
                                losses[4],
                                losses[6],
                                losses[5],
                                losses[7],
                                losses[8]))
        if epoch % conf.get_config('settings')['plot_freq'] ==0:
            plots.save_mesh(shape_net.implicit_mlp, plot_dir, epoch)
            plots.export_pointcloud(mesh, plot_dir, epoch)
    #     if call_back is not None:
    #         call_back(mesh, losses[0])

    # call_back(end=True)
    return losses


if __name__ == '__main__':

    from datasets.diligent_loader import Diligent
    from datasets.dtu_loader import DTU


    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./conf/diligent.conf', help='configure file name')

    opt = parser.parse_args()
    conf = ConfigFactory.parse_file(opt.conf)
    device = conf.get_config('settings')['device']

    if conf.get_config('settings')['dataset_type'] == 'diligent':
        dataset = Diligent(path_str=conf.get_config('settings')['dataset_path'],
                       name_str=conf.get_config('settings')['expname'],
                       view_num=conf.get_config('settings')['view_num'],
                       device=device)
    elif conf.get_config('settings')['dataset_type'] == 'dtu':
        dataset = DTU(path_str=conf.get_config('settings')['dataset_path'],
                       name_str=conf.get_config('settings')['expname'],
                       device=device)
    else:
        raise NotImplementedError


    pos_encode_weight = torch.cat(tuple(torch.eye(3) * (1.5**i) for i in range(0,14,1)), dim=0) #######
    pos_encode_out_weight = torch.cat(tuple( torch.tensor([1.0/(1.3**i)]*3) for i in range(0,14,1)), dim=0) #######
    
    model_conf = conf.get_config('model')
    print(model_conf['implicit_mlp.activations'])
    implicit_mlp=Sequential(
                        PositionEncoding(pos_encode_weight, pos_encode_out_weight),
                        MLP(pos_encode_weight.shape[0]*2, model_conf['implicit_mlp.dims'], model_conf['implicit_mlp.activations'], 
                        geometric_init=True, multires=14))

    shape_net = ShapeNet(velocity_mlp= Sequential(
                        PositionEncoding(pos_encode_weight, pos_encode_out_weight),
                        MLP(pos_encode_weight.shape[0]*2, model_conf['velocity_mlp.dims'], model_conf['velocity_mlp.activations']),  
                        ), implicit_mlp=implicit_mlp, T=conf.get_config('sampling')['T']
                        ).to(device)


    brdf_net = BRDFNet( Sequential(
                        PositionEncoding(pos_encode_weight, pos_encode_out_weight),  
                        MLP(pos_encode_weight.shape[0]*2, model_conf['brdf.dims']+[model_conf['brdf.n_lobes']*3+3], model_conf['brdf.activations']),    
                        ), constant_fresnel=True).to(device)

    optimizer = torch.optim.Adam(list(shape_net.parameters())+list(brdf_net.parameters()), lr=conf.get_config('training')['lr'])
    camera_settings = pytorch_camera(conf.get_config('rendering')['image_size'], dataset.K)
    camera_settings_silhoutte = pytorch_camera(conf.get_config('rendering')['image_size'], dataset.K)
    
    train(dataset, shape_net, brdf_net, optimizer, conf=conf,  
            call_back=None, 
            camera_settings = camera_settings,
            camera_settings_silhoutte=camera_settings_silhoutte
            )

    # load_models(f'{checkpoint_name}', brdf_net=brdf_net, shape_net=shape_net, 
                    # optimizer=optimizer)
    
    # mesh = sample_mesh(shape_net, brdf_net, **params)#init_mesh=init_mesh, **params)
    # trimesh.Trimesh( ( mesh.verts_packed().detach() @ transf[:3,:3].T + transf[:3,-1]).cpu().numpy(), mesh.faces_packed().cpu().numpy()).export(f'{checkpoint_name}.obj')

    # compile_video(mesh, f'{checkpoint_name}.mp4', distance=2, render_mode='image_ct', **params)