import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
# Util function for loading point clouds|
import numpy as np
from scipy.io import loadmat
import json
import sys

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
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

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    file_num = '024'
    base_path_simsub = '/home/saishubodh/rrc_projects_2021/graphVPR_project/'
    end_path = 'Hierarchical-Localization/graphVPR/ideas_SG/place-graphVPR/rand_json'
    json_dir = base_path_simsub + end_path
    p3p_num = 'p3p_{}'.format(file_num)
    p3p_dir = os.path.join(json_dir, p3p_num)
    p3p_files = os.listdir(p3p_dir)

    # TODO1: should change this to explicitly throw error if DUC_cutout not found.
    for i in p3p_files:
        if 'DUC_cutout' in i:
            break
            
    
    base_mat_path = "/scratch/saishubodh/InLoc_dataset/database/cutouts/DUC1/"
    jpg_file = os.path.join(base_mat_path + file_num, i)
    pc_file = jpg_file + '.mat'
    json_file = p3p_num + '.json'
    json_file = os.path.join(p3p_dir, json_file)

    # uncomment for sanity check
    # print(jpg_file, os.path.isfile(jpg_file))
    # print(json_file, os.path.isfile(json_file))
    # print(pc_file, os.path.isfile(pc_file))

    #create point cloud
    # TODO: modify/add loop for multi scans for entire room
    xyz_file  = loadmat(Path(pc_file))["XYZcut"]
    rgb_file = loadmat(Path(pc_file))["RGBcut"]
    xyz_sp = (xyz_file.shape)

    with open(json_file,'r') as f:
        json_data = json.load(f)
    xyz_file = (xyz_file.reshape((xyz_sp[0]*xyz_sp[1] ,3)))
    rgb_file = (rgb_file.reshape((xyz_sp[0]*xyz_sp[1] ,3)))

    verts = torch.Tensor(xyz_file).to(device)
    rgb = torch.Tensor(rgb_file).to(device)
    point_cloud = Pointclouds(points=[verts], features=[rgb])


    RT_np = np.array(json_data['extrinsic'])
    K_np = np.array(json_data['intrinsic']['intrinsic_matrix'])
    RT_np = RT_np.reshape(4,4).T
    K = K_np.reshape(3,3).T
    # print(RT_np)
    # print(K)

    fx, fy = K[0,0], K[1,1] # focal length in x and y axes
    px, py = K[0,2], K[1,2] # principal points in x and y axes
    R, t =  RT_np[0:3,0:3], np.array([RT_np[0:3,3]]) #  rotation and translation matrix

    # # First, (X, Y, Z) = R @ p_world + t, where p_world is 3D coordinte under world system
    # # To go from a coordinate under view system (X, Y, Z) to screen space, the perspective camera mode should consider
    # # the following transformation and we can get coordinates in screen space in the range of [0, W-1] and [0, H-1]
    # x_screen = fx * X / Z + px
    # y_screen = fy * Y / Z + py

    # In PyTorch3D, we need to build the input first in order to define camera. Note that we consider batch size N = 1
    RR = torch.from_numpy(R).unsqueeze(0) # dim = (1, 3, 3)
    # following line transposes matrix
    # RR = torch.from_numpy(R).permute(1,0).unsqueeze(0) # dim = (1, 3, 3) 
    tt = torch.from_numpy(t) # dim = (1, 3)
    # print(tt.size())
    f = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0) # dim = (1, 2)
    p = torch.tensor((px, py), dtype=torch.float32).unsqueeze(0) # dim = (1, 2)
    img_size = (1600, 1200) # (width, height) of the image

    # Now, we can define the Perspective Camera model. 
    # NOTE: you should consider negative focal length as input!!!
    cameras = PerspectiveCameras(R=RR, T=tt, focal_length=-f, principal_point=p,
        device=device,image_size=(img_size,))
    # print(cameras.K)
    # unsure about radius and points_per_pixel
    raster_settings = PointsRasterizationSettings(
        image_size=[1200,1600], 
        radius = 0.3,
        points_per_pixel = 20
    )
    # You can modify AlphaCompositor
    # to change background color
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    print("till here")
    images = renderer(point_cloud)
    #We need to convert 
    images_np = images[0, ..., :3].cpu().numpy()
    images_np[images_np < 0] = 0
    images_np = images_np.astype(np.uint8)
    plt.imsave(file_num + '_pytorchsyn.png',images_np)
    print(f"{file_num}_pytorchsyn.png saved")
