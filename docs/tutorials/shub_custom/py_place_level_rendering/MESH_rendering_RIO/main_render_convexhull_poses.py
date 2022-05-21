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
import argparse

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
from tf_camera_helper import convert_w_t_c, camera_params, moveback_tf_simple_given_pose
from places_creation import all_coords_from_mesh, create_list_of_rts_for_all_places
from o3d_helper import o3dframe_from_coords, o3dsphere_from_coords
from io_helper import write_individual_pose_txt_in_RIO_format

#def viz_image(RT_list, camera, dest_dir, mesh_dir, device):
def render_all_imgs_from_RT_list(fix_up_coord, RT_list, camera, dest_dir, mesh_dir, device):
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

    for i, RT_ctow in enumerate(RT_list):
        mb_bool = False
        if mb_bool:
            # a. moveback code
            RT_wtoc = convert_w_t_c(RT_ctow)
            moveback_dist = 0.0
            RT_wtoc_mb = moveback_tf_simple_given_pose(RT_wtoc, moveback_distance=moveback_dist)
            RT_ctow_mb = convert_w_t_c(RT_wtoc_mb)

            # b. moveback code
            print(f"debug RT_moveback: {i}")
            param.extrinsic = convert_w_t_c(RT_ctow_mb) # RT is RT_ctow, so this converts it to wtoc
            dest_file = 'places-'+ '{:04d}'.format(int(fix_up_coord * 100)) + '-' + '{:06d}'.format(i) + '_' + str(moveback_dist)  
        else:
        # You're getting this RT_list from the rt_given_lookat function, meaning it is pose, i.e. ctow
            param.extrinsic = convert_w_t_c(RT_ctow) # RT is RT_ctow, so this converts it to wtoc
            dest_file = 'places-'+ '{:04d}'.format(int(fix_up_coord * 100)) + '-' + '{:06d}'.format(i)

        dest_file_prefix = os.path.join(dest_dir, dest_file)
        print(f"\n{dest_file_prefix}")

        # print(param.extrinsic, param.intrinsic.intrinsic_matrix, dest_file)
        # render_py3d_img(i, img_size, param, dest_file, mesh_dir, device)

        # param.extrinsic is wtoc
        write_individual_pose_txt_in_RIO_format(RT_ctow, dest_file_prefix)
        render_py3d_img_and_depth(i, img_size, param, dest_file_prefix, mesh_dir, device)
        if i == 6 and mb_bool:
            print("sysexit")
            sys.exit()

        
#def synth_image(viz_pcd=False, custom_dir=False, device=None):
def render_places_main(ref_not_query, output_path, scene_id, viz_pcd=False, custom_dir=False, device=None):
    #Reading data paths
    #mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/models01/"
    #camera_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/"
    mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
    camera_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"
   
    ada = not viz_pcd
    if ada==True:
        #mesh_dir = "/scratch/saishubodh/RIO10_data/scene01/models01/"
        #camera_dir = "/scratch/saishubodh/RIO10_data/scene01/seq01/"
        mesh_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
        camera_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"

    if custom_dir:
        #mesh_dir = "../../../../../scene01/models01/"
        #camera_dir = "../../../../../scene01/seq01/"
        mesh_dir = "../../../../../scene" + scene_id + "/models" + scene_id + "/"
        camera_dir = "../../../../../scene" + scene_id + "/seq" + scene_id + "/"

    #To edit:
    #Sequence number and where visualisations are saved:
    #sequence_num = 'seq01_01/'
    if ref_not_query:
        sequence_num = 'seq' + scene_id + '_01/'
    else:
        sequence_num = 'seq' + scene_id + '_02/'
    #dest_dir = "temp_dir"
    dest_dir = str(output_path)

    mesh_dir = os.path.join(mesh_dir, sequence_num)
    camera_dir = os.path.join(camera_dir, sequence_num)

    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh.obj"), True)
    
    pcd_hull, centroids_coordinates, sphere_center_coords, fix_up_coord_list, linspace_num = all_coords_from_mesh(mesh, ref_not_query)

    #Set Camera parameters
    camera = camera_params(camera_dir)
    camera.set_intrinsics()
    print(device)
    
    for fix_up_coord in fix_up_coord_list:
        # print(f"Fixing up coord as {fix_up_coord}")

        centroids_coordinates[:,2] = np.ones((centroids_coordinates[:,2]).shape) * fix_up_coord

        list_of_rts = create_list_of_rts_for_all_places(pcd_hull, centroids_coordinates, sphere_center_coords, linspace_num)
        # list_of_rts = viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, camera, dest_dir, mesh_dir,device)
        render_all_imgs_from_RT_list(fix_up_coord, list_of_rts, camera, dest_dir, mesh_dir, device)


if __name__ == '__main__':
    """
    z - blue
    x - red
    y - green

    z - look at
    y - down
    x - cross
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--ref_or_query', type=str, required=True) #If reference, do "--ref_or_query ref". If query, anything else.
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    viz_pcd = False # viz_pcd doesn't have anything to do with visualization (use main_viz_.. for that).
    # final_poses = poses_for_places(viz_pcd, True)
    #scene_id = "01" # "02" #"01"
    scene_id = args.scene_id
    ref_not_query = args.ref_or_query
    ref_not_query = (ref_not_query=="ref")
    render_places_main(ref_not_query, args.output_path, scene_id, viz_pcd, False, device)

