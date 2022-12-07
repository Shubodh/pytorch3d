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
from io_helper import read_image, save_depth_image, read_depth_image_given_colorimg_path, read_depth_image_given_depth_path
from viz_helper import plot_images_simple


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
def render_py3d_img(img_id, img_size, param, dest_file, mesh_dir, device = None):
    """
    Given: camera params (both int and ext, img_size), mesh_parent_path
    Output: Renders image using py3d renderer. Saves image using plt.imsave(dest_file, rendered_image)
    """
    K = param.intrinsic.intrinsic_matrix
    RT = param.extrinsic
    RT_wtoc = RT
    RT_ctow = convert_w_t_c(RT_wtoc) #Here input RT is actually wtoc, output RT is ctow. Using this function as inverse and NOT as per how variables are written inside that function.
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
    if device is not None:
        lights = lights_given_position(RT_ctow[0:3, 3], device)
        
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

    print("Saving RGB rendered_image")
    plt.imsave(dest_file, rendered_image)

def render_py3d_img_and_depth(img_id, img_size, param, dest_file_prefix, mesh_dir, device = None, viz_rgb_depth = False):
    """
    Given: camera params (both int and ext, img_size), mesh_parent_path
    Output: Renders RGB AND DEPTH using py3d renderer. Saves image using plt.imsave(dest_file_prefix, rendered_image)
    """
    K = param.intrinsic.intrinsic_matrix
    RT = param.extrinsic
    RT_wtoc = RT
    RT_ctow = convert_w_t_c(RT_wtoc) #Here input RT is actually wtoc, output RT is ctow. Using this function as inverse and NOT as per how variables are written inside that function.
    # print(K, RT)

    mesh_obj_file = os.path.join(mesh_dir, 'mesh.obj')
    if device is not None:
        mesh = load_objs_as_meshes([mesh_obj_file], device=device)
    else:
        mesh = load_objs_as_meshes([mesh_obj_file])
    # print("Mesh loading done !!!")
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
    if device is not None:
        lights = lights_given_position(RT_ctow[0:3, 3], device)
        
    rasterizer=MeshRasterizer(
            cameras=cameras_pytorch3d, 
            raster_settings=raster_settings
    )
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SoftPhongShader(
            device=device,
            cameras=cameras_pytorch3d,
            lights=lights
        )
    )

    # 1. SAVING RGB IMAGE
    rendered_images = renderer(mesh)
    rgb_rendered_image = rendered_images[0, ..., :3].cpu().numpy()

    rgb_save_path = Path(str(dest_file_prefix) + ".color.jpg")

    try:
        plt.imsave(rgb_save_path, rgb_rendered_image)
        print(f"Saved RGB rendered image at {rgb_save_path}")
    except ValueError:
        pass
        print(f"ValueError: cannot save {rgb_save_path}. \n   SOLUTION: Please copy original image directly externally through CLI. Of course it is not rendered image, but it's ok to copy original itself as this is a very very rare case: 1 in 10000 or so. \n   Also, after finding the error image, you can debug it from main code: see debug_valueerror")

    # 2. SAVING DEPTH IMAGE
    fragments = rasterizer(mesh)
    depth_info = fragments.zbuf
    """
    fragments.zbuf is a (N, H, W, K) dimensional tensor
    #top k points. We'll take top 1
    source: https://github.com/facebookresearch/pytorch3d/issues/35#issuecomment-583870676
    """
    depth_image = depth_info[0,...,0].cpu().numpy()
    # In pytorch3d, holes in depth is rendered with -1 value. We replace with 0 as per RIO10 format.
    depth_image[depth_image == -1] = 0

    # depth_dest_path = Path(dest_file_prefix)
    # depth_save_path = Path(str(depth_dest_path.parents[0] / depth_dest_path.stem) + "_depth.png")
    depth_save_path = Path(str(dest_file_prefix) + ".rendered.depth.png")

    save_depth_image(depth_image, depth_save_path, depth_in_metres=True)


    
    if viz_rgb_depth:
        plot_images_simple(rgb_rendered_image, depth_image)
        plt.show()

    # print("ORIGINAL ARRAY")
    # print(np.unique(depth_image))
    # depth_image_2 = read_depth_image_given_depth_path(save_path)
    # print("READING SAVED ARRAY")
    # print(np.unique(depth_image_2))


