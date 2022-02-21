import os
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.io import loadmat
import json
import sys
import matplotlib.image as mpimg
import yaml
import open3d as o3d
import copy

import torch
import torch.nn.functional as F

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

from pytorch3d.utils import cameras_from_opencv_projection

# Custom utils functions
from tf_camera_helper import convert_w_t_c


def RtK_in_torch_format(K, RT, img_size):
    # fx, fy = K[0,0], K[1,1] # focal length in x and y axes
    # px, py = K[0,2], K[1,2] # principal points in x and y axes
    R, t =  RT[0:3,0:3], np.array([RT[0:3,3]]) #  rotation and translation matrix

    RR = torch.from_numpy(R).unsqueeze(0) # dim = (1, 3, 3)
    KK = torch.from_numpy(K).unsqueeze(0) # dim = (1, 3, 3)
    # following line transposes matrix
    # RR = torch.from_numpy(R).permute(1,0).unsqueeze(0) # dim = (1, 3, 3) 
    tt = torch.from_numpy(t) # dim = (1, 3)
    # f = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0) # dim = (1, 2)
    # p = torch.tensor((px, py), dtype=torch.float32).unsqueeze(0) # dim = (1, 2)
    img_size_t = torch.tensor(img_size).unsqueeze(0) # (width, height) of the image

    return RR, tt, KK, img_size_t

def lights_given_position(position, device):
    # lights = PointLights(device=device, location=[[-(1.67-4.2)/2, -(4.05 - 0.38)/2, 1.0]])
    # lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])
    # lights = PointLights(device=device, location=[[1.138, -1.92, 1.54]])
    # one, two, three = 1.88506202, -1.34736095,  1.11616586 # mesh center
    one, two, three = position
    lights = PointLights(device=device, location=[[ one, two, three]])
    #print(one, two, three)
    return lights


# import open3d as o3d
# %matplotlib inline
# def render_py3d_img(img_size, param, dest_file, mesh_dir, device):
def render_py3d_img(img_size, param, dest_file, mesh_dir, device = None):
    K = param.intrinsic.intrinsic_matrix
    RT = param.extrinsic
    # print(K, RT)

    mesh_obj_file = os.path.join(mesh_dir, 'mesh.obj')
    print("Testing IO for meshes ...")
    if device is not None:
        mesh = load_objs_as_meshes([mesh_obj_file], device=device)
    else:
        mesh = load_objs_as_meshes([mesh_obj_file])
    print("Mesh loading done !!!")
    texture_image=mesh.textures.maps_padded()
    # plt.figure(figsize=(7,7))
    # plt.imshow(texture_image.squeeze().cpu().numpy())
    # plt.show()
    RR, tt, KK, img_size_t = RtK_in_torch_format(K, RT, img_size)
    # print(RR)
    cameras_pytorch3d = cameras_from_opencv_projection(RR.float(), tt.float(), KK.float(), img_size_t.float())
    if device is not None:
        cameras_pytorch3d  = cameras_pytorch3d.to(device)

    raster_settings = RasterizationSettings(
                image_size=img_size, 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )
    # lights = lights_given_position(RT_wtoc[0:3, 3], device
    print("TODO2: Change RT variable explicitly to wtoc or ctow")
    RT = convert_w_t_c(RT)
    if device is not None:
        lights = lights_given_position(RT[0:3, 3], device)
        
    renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras_pytorch3d, 
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=device,
                    cameras=cameras_pytorch3d,
                    lights=lights
                )
            )
    rendered_images = renderer(mesh)
    rendered_image = rendered_images[0, ..., :3].cpu().numpy()

    plt.imsave(dest_file, rendered_image)
