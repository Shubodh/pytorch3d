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
from io_helper import write_individual_pose_txt_in_RIO_format, return_individual_pose_files_as_single_list

#def viz_image(RT_list, camera, dest_dir, mesh_dir, device):
def render_all_imgs_from_RT_list(RT_list, dict_name_pose, camera, dest_dir, mesh_dir, device):
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

    #for i, RT_ctow in enumerate(RT_list):
    for key, RT_ctow in dict_name_pose.items():
        # print("debug RT_moveback")
        # RT_wtoc = convert_w_t_c(RT_ctow)
        # RT_wtoc_mb = moveback_tf_simple_given_pose(RT_wtoc, moveback_distance=0.5)
        # RT_ctow_mb = convert_w_t_c(RT_wtoc_mb)
        # print(RT_wtoc, "\n", RT_wtoc_mb)
        # print(RT_ctow, "\n", RT_ctow_mb)
        # sys.exit()
        # You're getting this RT_list from the rt_given_lookat function, meaning it is pose, i.e. ctow
        param.extrinsic = convert_w_t_c(RT_ctow) # RT is RT_ctow, so this converts it to wtoc
        #dest_file = 'places-' + '{:06d}'.format(i)
        #print(key, RT_ctow)
        dest_file = key
        #dest_file = 'frame-rendered-'+ '{:06d}'.format(i)
        dest_file_prefix = os.path.join(dest_dir, dest_file)
        print(f"\n{dest_file_prefix}")

        # print(param.extrinsic, param.intrinsic.intrinsic_matrix, dest_file)
        # render_py3d_img(i, img_size, param, dest_file, mesh_dir, device)

        # param.extrinsic is wtoc
        write_individual_pose_txt_in_RIO_format(RT_ctow, dest_file_prefix)
        i=0
        render_py3d_img_and_depth(i, img_size, param, dest_file_prefix, mesh_dir, device)

        
#def synth_image(viz_pcd=False, custom_dir=False, device=None):
def render_places_main(ref_not_query, scene_id, output_path=None,  device=None):
    output_path = "/scratch/saishubodh/InLoc_like_RIO10/sampling10/scene" + scene_id + "_RRI_with_QRI/"

    camera_dir = "/scratch/saishubodh/InLoc_like_RIO10/sampling10/scene" + scene_id + "_ROI_with_QOI/"
    mesh_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
    # camera_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"


    if ref_not_query:
        sequence_num_mesh = 'seq' + scene_id + '_01/'
        sequence_num_cam =  'database/cutouts/'
    else:
        sequence_num_mesh = 'seq' + scene_id + '_02/'
        sequence_num_cam =  'query/'

    dest_dir = str(output_path) + sequence_num_cam
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    mesh_dir = os.path.join(mesh_dir, sequence_num_mesh)
    camera_dir = os.path.join(camera_dir, sequence_num_cam)

    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh.obj"), True)
    
    # pcd_hull, centroids_coordinates, sphere_center_coords, fix_up_coord_list, linspace_num = all_coords_from_mesh(mesh, ref_not_query)

    #Set Camera parameters
    camera = camera_params(camera_dir)
    camera.set_intrinsics()
    print(device)
    
    # for fix_up_coord in fix_up_coord_list:
    #     # print(f"Fixing up coord as {fix_up_coord}")

    #     centroids_coordinates[:,2] = np.ones((centroids_coordinates[:,2]).shape) * fix_up_coord

    # list_of_rts = create_list_of_rts_for_all_places(pcd_hull, centroids_coordinates, sphere_center_coords, linspace_num)
    list_of_rts, dict_name_pose = return_individual_pose_files_as_single_list(camera_dir, scene_id)

    debug_valueerror =False #True #  

    if not debug_valueerror:
        render_all_imgs_from_RT_list(list_of_rts, dict_name_pose, camera, dest_dir, mesh_dir, device)

    else:
        print("DEBUGGING: ")
        start, end = 6059, 6064 #6062, 6106 #6059, 6066  # 6059, 6062 #
        new_list_of_rts = list_of_rts[start:end] 
        new_dict_name_pose = {k: dict_name_pose[k] for k in sorted(dict_name_pose.keys())[start:end]}
        #print(new_dict, "\n", new_list_of_rts)
        #sys.exit()
        render_all_imgs_from_RT_list(new_list_of_rts, new_dict_name_pose, camera, dest_dir, mesh_dir, device)


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
    # parser.add_argument('--output_path', type=Path, required=True)
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
    render_places_main(ref_not_query=ref_not_query, scene_id=scene_id, device=device)

