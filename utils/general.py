import os
from glob import glob
import torch
import cv2
import numpy as np
import imageio
import skimage
from pytorch3d.utils.ico_sphere import ico_sphere
import random
from model.meshes import Meshes
from dotty_dict import dotty
from pytorch3d.renderer import look_at_view_transform, DirectionalLights
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.transforms import so3_exponential_map, so3_rotation_angle

HAT_INV_SKEW_SYMMETRIC_TOL=1e-5

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def glob_imgs(path):
    imgs = []
    for ext in ['0*.png', '0*.jpg', '0*.JPEG', '0*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs


def load_rgb(path):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    # TODO: check why transpose
    # pixel values between [-1,1]
    # img -= 0.5
    # img *= 2.
    # img = img.transpose(2, 0, 1)
    return img

def load_mask(path):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)
    object_mask = alpha > 127.5

    return object_mask


def concat_home_dir(path):
    return os.path.join(os.environ['HOME'],'data',path)


def save_images(images, path_prefix, max_val=None):
    images = images.detach().cpu().numpy()
    if max_val is None:
        max_val = images.max()
        
    images = np.clip(images, 0, max_val)
    images *= 255 / max_val
    images = (images).astype(np.uint8)
    for i, img in enumerate(images):
        cv2.imwrite(f'{path_prefix}{i:02}.png', img[...,::-1])
        
def P_matrix_to_rot_trans_vectors(P):
    '''
    convert mvs-style P matrix
    to pytorch3d style rotation and translation vectors
    '''
    P = P / P[...,-1:,-1:]
    R_mvs = P[...,:3,:3]
    T_mvs = P[...,:3,3:]

    transf = torch.tensor([
        [-1,0,0],
        [0,-1,0],
        [0,0,1],
    ]).to(P.device).to(P.dtype)
    R_pt3d = (transf @ R_mvs).transpose(-1,-2) # transf@R_mvs
    # r_pt3d = R2r(R_pt3d, eps=0.0005)

    t_pt3d = (transf @ T_mvs).squeeze(-1)
    return R_pt3d, t_pt3d

def pytorch_camera(img_size, K):
    '''
    convert mvs-style intrinsic matrix to
    pytorch-style camera
    img_size must be square
    K: 3-by-4 matrix
    '''
    focal_length = torch.diag(K)[:2].reshape(-1,2)
    principal_point = K[:2,2].reshape(-1,2)

    camera_settings = dict(focal_length=focal_length,principal_point=principal_point, image_size=((img_size,img_size),))
    return camera_settings


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose   


def hat_inv(h):
    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = (h + h.permute(0, 2, 1)).abs().max()
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v

def so3_log_map(R, eps=0.0001):

    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")
    phi = so3_rotation_angle(R,eps)
    phi_sin = phi.sin()
    phi_denom = (
        torch.clamp(phi_sin.abs(), eps) * phi_sin.sign()
        + (phi_sin == 0).type_as(phi) * eps
    )
    log_rot_hat = (phi / (2.0 * phi_denom))[:, None, None] * (R - R.permute(0, 2, 1))
    log_rot = hat_inv(log_rot_hat)
    return log_rot

def r2R(x):
    '''
    convert axis angle to rotation matrix
    x is of shape (...,3)
    returns (..., 3,3)
    '''
    assert x.shape[-1] == 3
    trg_shape = x.shape + (3,)
    return so3_exponential_map(x.reshape(-1,3)).reshape(trg_shape)

def R2r(x, eps=0.0001):
    '''
    convert rotation matrix to axis angle representation
    x is of shape (...,3,3)
    returns (..., 3)
    '''
    assert x.shape[-2:] == (3,3)
    trg_shape = x.shape[:-1]
    return so3_log_map(x.reshape(-1,3,3), eps=eps).reshape(trg_shape)

def random_dirs(n, device):
    return torch.nn.functional.normalize(torch.randn(n, 3, device=device), dim=-1)

def compute_view_pt(r, t):
    return (-r2R(r)@t.unsqueeze(-1)).squeeze(-1)

def freeze_module(module, requires_grad=False):
    for p in module.parameters():
        p.requires_grad = requires_grad

def get_camera_parameters(camera, img_size, style='mvs', transpose=False):
    
    assert style in ("pytorch3d", "mvs")

    if not torch.is_tensor(img_size):
        img_size = torch.tensor(img_size, dtype=torch.int64, device=camera.device)
    if (img_size < 1).any():
        raise ValueError("Provided image size is invalid.")

    image_width, image_height = img_size.unbind(1)
    image_width = image_width.view(-1, 1)  # (N, 1)
    image_height = image_height.view(-1, 1)  # (N, 1)

    P = camera.get_world_to_view_transform().get_matrix()
    K_ndc = camera.get_projection_transform().get_matrix()

    r_w = (image_width - 1.0) / 2.0
    r_h = (image_height - 1.0) / 2.0
    _0 = torch.zeros_like(r_w)
    _1 = _0 + 1
    _ndc_2_screen = torch.cat((-r_h, _0, _0, 
                               _0, -r_h, _0, 
                               _0, _0, _0,
                               r_w, r_h, _1), dim=0).reshape(4,3,-1).permute(2,0,1) # (-1, 4, 3)
    K = K_ndc @ _ndc_2_screen

    if style == 'mvs':
        flip_mat = torch.diag(torch.tensor([-1,-1,1,1]).to(K.device).to(K.dtype)).reshape(-1,4,4)
        P = P @ flip_mat
        K = flip_mat @ K
    if not transpose:
        P = P.transpose(-1,-2)
        K = K.transpose(-1,-2)

    return K, P


@torch.no_grad()
def compile_video(mesh, filename, distance=3, render_mode='image_ct', ambient_net=None, **params):
    from model.render import render_mesh
    params = dotty(params)
    device = params['device']

    elevations, azimuths = zip(*product(np.arange(-30, 30.1, 30)+1, np.arange(0, 360, 5)+1))


    Rs, Ts = zip(*[look_at_view_transform(distance, elevation, azimuth, device=device) for elevation, azimuth in zip(elevations, azimuths)])
    Rs = torch.cat(Rs, dim=0)
    ts = torch.cat(Ts, dim=0)
    rs = R2r(Rs)
    video_writer = None

    for r, t in tqdm(list(zip(rs, ts))):
        frame = render_mesh(mesh, render_mode, r[None], t[None], params['rendering.rgb.image_size'],
                        0.0, 1, params['device'], sigma=1e-4, gamma=1e-4, ambient_net=ambient_net)
        if 'image' in render_mode or 'ambient' in render_mode or 'texture' in render_mode:
            frame = (frame / params['rendering.rgb.max_intensity']).clamp_max(1.0)
        elif 'normal' in render_mode:
            frame = (frame + 1) / 2

        frame = np.clip((frame[0] * 255).cpu().numpy(), 0, 255).astype(np.uint8) #######
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(filename, fourcc, 30, (frame.shape[1], frame.shape[0]))
        
        video_writer.write((frame * np.ones((3), np.uint8))[...,::-1])


def fibonacci_sphere(samples=1, radius=1, upper_sphere=False):
    '''
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    :param samples:
    :return:
    '''
    normal = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        if upper_sphere:
            y = 1 - (i / float(samples - 1))
        else:
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        _radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * _radius
        z = np.sin(theta) * _radius

        normal.append([x, z, y])
        
        
    normal = np.array(normal)
    points = normal * radius # on sphere normal is <x,y,z> 
    curvature = np.ones(samples) / radius # curvature of sphere is 1/R
    
    return torch.from_numpy(normal).float(), torch.from_numpy(points).float(), curvature

def load_models(path, **models):
    data = torch.load(path)
    for model_name in models:
        models[model_name].load_state_dict(data[model_name])
    return {k: data[k] for k in data if k not in models}
    
def save_models(path, meta, **models):
    data = {name: models[name].state_dict() for name in models}
    torch.save({**data, **meta}, path)

@torch.no_grad()
def sample_trimesh(shape_net, level, device):
    ico_sphere = rand_ico_sphere(level, device)
    verts, faces = ico_sphere.verts_packed(), ico_sphere.faces_packed()
    verts = shape_net(verts, 0)[0]
    return verts[-1], faces

def sample_lights_from_equirectangular_image(path, target_img_size, device, intensity_ratio=1.0):
    rgb_img = cv2.resize(cv2.imread(path), target_img_size).astype(np.float32)[...,::-1]
    rgb_img = rgb_img * 0 + rgb_img.shape[0] - np.arange(rgb_img.shape[0]).reshape(-1,1,1)
    rgb_img = torch.from_numpy(rgb_img.copy()).to(device) 
    # rgb_img = rgb_img * 0 + 1
    height, width = rgb_img.shape[:2]

    azimuths = torch.arange(0, width, dtype=torch.float32, device=device) / width * np.pi * 2 - (width-1) / width * np.pi
    elevations = -torch.arange(0, height, dtype=torch.float32, device=device) / height * np.pi + (height-1) / height * np.pi / 2

    elevations, azimuths = torch.meshgrid(elevations, azimuths)
    dirs = torch.stack((
        torch.cos(elevations)*torch.sin(azimuths), #x
        torch.cos(elevations)*torch.cos(azimuths), #y
        torch.sin(elevations), #z
    ), dim=-1)

    radiance_ratio = torch.cos(elevations).unsqueeze(-1)
    radiance_img = rgb_img * radiance_ratio
    radiance_img = radiance_img / radiance_img.sum() * intensity_ratio

    lights = DirectionalLights(direction=radiance_img.reshape(-1,3), diffuse_color=dirs.reshape(-1,3), device=device)
    return lights, (radiance_img.reshape(-1,3), dirs.reshape(-1,3))
    

def manual_seed(seed, deterministic=None):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic is not None:
        torch.set_deterministic(deterministic)
        torch.backends.cudnn.benchmark = not deterministic

def grad_clip(module, clip=0.1):
    for p in module.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(-clip, clip)

def rand_rot_mat(ndim, device):
    rand_rot, _ = torch.qr(torch.randn(ndim,ndim, device=device))
    if torch.det(rand_rot) < 0:
        rand_rot = -rand_rot
    return rand_rot

def rand_ico_sphere(level=0, device=None, seed=None):
    if seed is not None:
        manual_seed(seed)
    ish = ico_sphere(level, device)
    verts = ish.verts_packed() @ rand_rot_mat(3, device)
    verts = verts / torch.sqrt((verts**2).sum(-1, keepdims=True))
    faces = ish.faces_packed()
    return Meshes(verts=[verts], faces=[faces])

def get_pointcloud_attributes(pix_to_pt, pt_features):
    
    """only works with max 4 dims tensor
       pix_to_pt:  [N,W,H,K] or [N,W,H]
       pt_features: [N,P,C] or [N,P] if N=1, could be [P,C] or [P]
    
    """
    
    i_shape = pix_to_pt.shape
    batch_size = i_shape[0]
    if batch_size > 1:
        assert pt_features.shape[0] == batch_size
    if batch_size == 1 and pt_features.shape[0] != 1:
        pt_features = pt_features.unsqueeze(-len(pt_features.shape)-1)
    
    if len(pt_features.shape) > 2:
        if pt_features.shape[1] < pt_features.shape[2]:
            pt_features = pt_features.permute(0,2,1)
        pix_features = pt_features[:, pix_to_pt.flatten().long(), :]
        pix_features = torch.reshape(pix_features, i_shape + (pt_features.shape[-1],))
    
    else:
        pix_features = pt_features[:, pix_to_pt.flatten().long()]
        pix_features = torch.reshape(pix_features, i_shape)
    
    
    return pix_features



def check_nonumber(x, warning_name=None):
    num_nan = torch.isnan(x).sum()
    num_inf = torch.isinf(x).sum()
    if num_nan > 0 or num_inf > 0:
        if warning_name is not None:
            print('non number appears in', warning_name)
            print('nan:', num_nan)
            print('inf', num_inf)
        return True
    
    return False
