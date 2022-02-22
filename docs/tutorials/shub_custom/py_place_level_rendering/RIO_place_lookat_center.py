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


# Custom utils functions
# from pytorch3d_utils import render_py3d_img
from tf_camera_helper import convert_w_t_c, camera_params
from places_creation import convex_hull, dbscan_clustering, rt_given_lookat
from o3d_helper import o3dframe_from_coords, o3dsphere_from_coords, create_o3d_param_and_viz_image


def load_view_point(pcd, img_size, param):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=img_size[0], width=img_size[1])
    print(img_size[0], img_size[1])
    ctr = vis.get_view_control()
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    vis.run()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    return image

def viz_image(mesh, RT, camera):
    model = camera.model
    img_size = camera.img_size
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic.set_intrinsics(width = img_size[1],
                                                height = img_size[0],
                                                fx = model[0],
                                                fy = model[1],
                                                cx = model[2],
                                                cy = model[3])

    param.extrinsic = convert_w_t_c(RT)
    load_view_point(mesh, img_size, param)
    



def viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, viz_pcd, camera):
    # Visualise for particular centroid
    # in this case we will visualise for
    # number 10
    test_num = 10
    print("centroid, coordinates", centroids_coordinates[test_num], sphere_center_coords)

    #sampling points along center and convex hull point location
    sample_poses =  np.linspace(centroids_coordinates[test_num], sphere_center_coords, num=4)
    

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

    RT = rt_given_lookat(centroids_coordinates[test_num],sphere_center_coords)
    camera_center = o3dframe_from_coords(RT, color=[0.6, 0.706, 1], radius=0.2)
    poses_list.append(camera_center)


    # visualise_camera(mesh, K, RT)
    if viz_pcd:
        o3d.visualization.draw_geometries(poses_list)

    viz_image(mesh, RT, camera)
    
    #if we want other way round
    RT = rt_given_lookat(sphere_center_coords, centroids_coordinates[test_num])
    viz_image(mesh, RT, camera)
    return 


    

def synth_image(viz_pcd=False, custom_dir=False):
    #Reading data paths
    mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/models01/seq01_01/"
    camera_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/seq01_01/"
   
    ada = not viz_pcd
    if ada==True:
        mesh_dir = "/scratch/saishubodh/RIO10_data/scene01/models01/seq01_01/"
        camera_dir = "scratch/saishubodh/RIO10_data/scene01/seq01/seq01_01/"

    if custom_dir:
        mesh_dir = "../../../../../scene01/models01/seq01_01/"
        camera_dir = "../../../../../scene01/seq01/seq01_01/"

    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh.obj"), True)

    #Apply Convex Hull
    pcd_hull = convex_hull(mesh)
    colored_pcd_hull, centroids_coordinates = dbscan_clustering(pcd_hull)

    # ### fixing up coordinate for clusters
    pcd_hull_points = (np.asarray(pcd_hull.points))
    pcd_hull_points[:,2] = np.ones((pcd_hull_points[:,2]).shape)
    sphere_center_coords = np.mean(pcd_hull_points, axis=0)
    
    centroids_coordinates[:,2] = np.ones((centroids_coordinates[:,2]).shape)

    camera = camera_params(camera_dir)
    camera.set_intrinsics()
    viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, viz_pcd, camera)


if __name__ == '__main__':
    """
    z - blue
    x - red
    y - green

    z - look at
    y - down
    x - cross
    """
    viz_pcd = True
    # final_poses = poses_for_places(viz_pcd, True)
    synth_image(viz_pcd, True)