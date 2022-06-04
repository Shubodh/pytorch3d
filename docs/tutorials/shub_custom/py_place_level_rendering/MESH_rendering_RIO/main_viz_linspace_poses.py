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


# Custom utils functions
# from pytorch3d_utils import render_py3d_img
from tf_camera_helper import convert_w_t_c, camera_params
from places_creation import all_coords_from_mesh, rt_given_lookat, create_list_of_rts_for_all_places
from o3d_helper import o3dframe_from_coords, o3dsphere_from_coords, create_o3d_param_and_viz_image


# def create_o3d_param_and_viz_image(mesh, RT, camera):
#     model = camera.model
#     img_size = camera.img_size
#     param = o3d.camera.PinholeCameraParameters()
#     param.intrinsic.set_intrinsics(width = img_size[1],
#                                                 height = img_size[0],
#                                                 fx = model[0],
#                                                 fy = model[1],
#                                                 cx = model[2],
#                                                 cy = model[3])

    # NOTE THIS LINE: different from create_o3d_param_and_viz_image you're importing
#     param.extrinsic = convert_w_t_c(RT)
#     load_view_point(mesh, img_size, param)
    

# def viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, viz_pcd, camera):
def misc_viz_linspace_poses(centroids_coordinates, sphere_center_coords, mesh, viz_pcd, camera):
    # Visualise for particular centroid
    # in this case we will visualise for
    # number 10
    test_num = 10
    print("centroid, coordinates", centroids_coordinates[test_num], sphere_center_coords)

    #sampling points along center and convex hull point location
    sample_poses =  np.linspace(centroids_coordinates[test_num], sphere_center_coords, num=4)
    sample_poses = sample_poses[:-1] #Want all poses except the last one, which is sphere_center itself.
    

    poses_list = [mesh]
    for i_cent in range(sample_poses.shape[0]):
        sphere = o3dsphere_from_coords(sample_poses[i_cent], color=[1, 0.206, 0.7], radius=0.1)
        poses_list.append(sphere)

    #Camera is located here
    sphere_center = o3dsphere_from_coords(sphere_center_coords, color=[0.5, 0.706, 0], radius=0.1)
    
    #Camera is looking here
    given = o3dsphere_from_coords(centroids_coordinates[test_num], color=[0.5, 0.706, 0], radius=0.1)
    # poses_list.append(sphere_center)
    # poses_list.append(given)
    
    #plot as mesh. This transform is camera to world
    #for image synth purposes, we will have to do world to camera

    # 1. Lookat hull centroids from sphere_center
    RT = rt_given_lookat(centroids_coordinates[test_num],sphere_center_coords)
    camera_center = o3dframe_from_coords(RT, color=[0.6, 0.706, 1], radius=0.2)
    poses_list.append(camera_center)

    RT_wtoc = convert_w_t_c(RT)
    create_o3d_param_and_viz_image(mesh, RT_wtoc, camera)
    
    #2. if we want other way round: Lookat sphere_center from hull centroids
    RT_other = rt_given_lookat(sphere_center_coords, centroids_coordinates[test_num])
    camera_center_other = o3dframe_from_coords(RT_other, color=[0.6, 0.706, 1], radius=0.2)
    poses_list.append(camera_center_other)

    RT_wtoc = convert_w_t_c(RT_other)
    create_o3d_param_and_viz_image(mesh, RT_wtoc, camera)

    # sample_poses: Visualizing not just spheres but also all frames at linspace sampled poses.
    for i_coord in range(sample_poses.shape[0]):
        RT = rt_given_lookat(sphere_center_coords, sample_poses[i_coord])
        camera_center = o3dframe_from_coords(RT, color=[0.6, 0.706, 1], radius=0.2)
        poses_list.append(camera_center)

    if viz_pcd:
        o3d.visualization.draw_geometries(poses_list)
    return 

def viz_linspace_poses(centroids_coordinates, sphere_center_coords, fix_up_coord_list, linspace_num, mesh, viz_pcd, camera, pcd_hull, points_viz):
    poses_list = [mesh]
    for fix_up_coord in fix_up_coord_list:
        print(f"Fixing up coord for hull points as {fix_up_coord}")

        centroids_coordinates[:,2] = np.ones((centroids_coordinates[:,2]).shape) * fix_up_coord

        list_of_rts = create_list_of_rts_for_all_places(pcd_hull, centroids_coordinates, sphere_center_coords, linspace_num)

        for RT in list_of_rts:
            # RT = rt_given_lookat(sphere_center_coords, sample_poses[i_coord])
            if points_viz:
                camera_center = o3dsphere_from_coords(RT[:3,3], color=[1, 0.706, 0], radius = 0.05)
            else:
                camera_center = o3dframe_from_coords(RT, color=[0.6, 0.706, 1], size=0.1)
            poses_list.append(camera_center)

    sphere_center = o3dsphere_from_coords(sphere_center_coords, color=[0, 0.706, 0], radius = 0.1)
    poses_list.append(sphere_center)
    print(f"Total number of RT's roughly (+2/3) are {len(poses_list)}")
    if viz_pcd:
        o3d.visualization.draw_geometries(poses_list)

# def synth_image(viz_pcd=False, custom_dir=False):
def main_linspace_poses(ref_not_query, scene_id, viz_pcd=False, custom_dir=False, points_viz=True):
    #Reading data paths
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

    mesh_dir = os.path.join(mesh_dir, sequence_num)
    camera_dir = os.path.join(camera_dir, sequence_num)

    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh.obj"), True)

    pcd_hull, centroids_coordinates, sphere_center_coords, fix_up_coord_list, linspace_num = all_coords_from_mesh(mesh, ref_not_query)

    camera = camera_params(camera_dir)
    camera.set_intrinsics()

    # misc_viz_linspace_poses(centroids_coordinates, sphere_center_coords, mesh, viz_pcd, camera)
    viz_linspace_poses(centroids_coordinates, sphere_center_coords, fix_up_coord_list, linspace_num, mesh, viz_pcd, camera, pcd_hull, points_viz)


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
    parser.add_argument('--scene_id', type=str, required=True) #01
    parser.add_argument('--ref_or_query', type=str, required=True) # Acceptable args: 'ref' / 'query'
    #If reference, do "--ref_or_query ref". If query, anything else.
    args = parser.parse_args()

    viz_pcd = True
    points_viz = True #If you want to visualize balls, True; if frames, False.
    # final_poses = poses_for_places(viz_pcd, True)
    scene_id = args.scene_id
    ref_not_query = args.ref_or_query
    ref_not_query = (ref_not_query=="ref")

    # if ref_not_query:
    #     main_linspace_poses(ref_not_query, scene_id, viz_pcd=viz_pcd, custom_dir=False, points_viz=points_viz)
    main_linspace_poses(ref_not_query, scene_id, viz_pcd=viz_pcd, custom_dir=False, points_viz=False)