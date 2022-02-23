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
from pytorch3d_utils import render_py3d_img, render_py3d_img_and_depth
from tf_camera_helper import convert_w_t_c, camera_params
from places_creation import convex_hull, dbscan_clustering, rt_given_lookat
from o3d_helper import o3dframe_from_coords, o3dsphere_from_coords

#def viz_image(RT_list, camera, dest_dir, mesh_dir, device):
def render_all_imgs_from_RT_list(RT_list, camera, dest_dir, mesh_dir, device):
    """
    Render images based on RT_list
    """
    model = camera.model
    img_size = camera.img_size
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic.set_intrinsics(width = img_size[1],
                                                height = img_size[0],
                                                fx = model[0],
                                                fy = model[1],
                                                cx = model[2],
                                                cy = model[3])

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    for i, RT in enumerate(RT_list):
        print("TODO3: Lack of clarity on w_t_c. Rewrite the function as inverse_tf or something. It goes to render_ func and gets w_to_c'ed again")
        param.extrinsic = convert_w_t_c(RT)
        dest_file = '{:02d}'.format(i) + '.png'
        dest_file = os.path.join(dest_dir, dest_file)
        print(dest_file)

        # print(param.extrinsic, param.intrinsic.intrinsic_matrix, dest_file)
        # render_py3d_img(i, img_size, param, dest_file, mesh_dir, device)
        render_py3d_img_and_depth(i, img_size, param, dest_file, mesh_dir, device)

#def viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, camera, dest_dir, mesh_dir, device):
def create_list_of_rts_for_all_places(centroids_coordinates, sphere_center_coords):
    # Visualise for particular centroid
    # in this case we will visualise for
    # number 10
    list_of_rts = []
    for hull_point in range(len(centroids_coordinates)):
        # print("centroid, coordinates", centroids_coordinates[hull_point], sphere_center_coords)
        #sampling points along center and convex hull point location
        #sample_poses =  np.linspace(centroids_coordinates[hull_point], sphere_center_coords, num=5)

        #list_of_rts = []
        
        #rt_given_lookat(lookat,location)
        list_of_rts.append(rt_given_lookat(sphere_center_coords, centroids_coordinates[hull_point]))
        #Below for loop is giving NaNs. Skipping for now
        # for pose in sample_poses:
        #     #Here RT describes the location of the camera
        #     RT = rt_given_lookat(sphere_center_coords, pose)
        #     list_of_rts.append(RT)

    #print(list_of_rts)
    # viz_image(list_of_rts, camera, dest_dir, mesh_dir, device)
    return list_of_rts
        
#def synth_image(viz_pcd=False, custom_dir=False, device=None):
def render_places_main(viz_pcd=False, custom_dir=False, device=None):
    #Reading data paths
    mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/models01/"
    camera_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/"
   
    ada = not viz_pcd
    if ada==True:
        mesh_dir = "/scratch/saishubodh/RIO10_data/scene01/models01/"
        camera_dir = "/scratch/saishubodh/RIO10_data/scene01/seq01/"

    if custom_dir:
        mesh_dir = "../../../../../scene01/models01/"
        camera_dir = "../../../../../scene01/seq01/"

    #To edit:
    #Sequence number and where visualisations are saved:
    sequence_num = 'seq01_01/'
    dest_dir = "temp_dir"



    mesh_dir = os.path.join(mesh_dir, sequence_num)
    camera_dir = os.path.join(camera_dir, sequence_num)

    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh.obj"), True)

    #Apply Convex Hull
    pcd_hull = convex_hull(mesh)
    colored_pcd_hull, centroids_coordinates = dbscan_clustering(pcd_hull)


    # ### fixing up coordinate for clusters
    pcd_hull_points = (np.asarray(pcd_hull.points))
    pcd_hull_points[:,2] = np.ones((pcd_hull_points[:,2]).shape)
    sphere_center_coords = np.mean(pcd_hull_points, axis=0)
    
    centroids_coordinates[:,2] = np.ones((centroids_coordinates[:,2]).shape)

    #Set Camera parameters
    camera = camera_params(camera_dir)
    camera.set_intrinsics()
    print(device)

    list_of_rts = create_list_of_rts_for_all_places(centroids_coordinates, sphere_center_coords)
    # list_of_rts = viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, camera, dest_dir, mesh_dir,device)
    render_all_imgs_from_RT_list(list_of_rts, camera, dest_dir, mesh_dir, device)


if __name__ == '__main__':
    """
    z - blue
    x - red
    y - green

    z - look at
    y - down
    x - cross
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    viz_pcd = False
    # final_poses = poses_for_places(viz_pcd, True)
    render_places_main(viz_pcd, False, device)

