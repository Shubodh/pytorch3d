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
import fileinput

from scipy.spatial.transform import Rotation as R

# import open3d as o3d
# %matplotlib inline

# Custom utils functions
# from pytorch3d_utils import render_py3d_img
from tf_camera_helper import convert_w_t_c, camera_params
from places_creation import convex_hull, dbscan_clustering, rt_given_lookat
from o3d_helper import o3dframe_from_coords, o3dsphere_from_coords, create_o3d_param_and_viz_image

def read_pose_file_and_viz(mesh, camera, filename):
    list_of_rts = []

    for line in fileinput.input(files=filename):
        list_of_rts.append(line.strip())

    for rt in list_of_rts:
        rt_parts = rt.split()
        sequence_num = rt_parts[0]
        w = float(rt_parts[1])
        x = float(rt_parts[2])
        y = float(rt_parts[3])
        z = float(rt_parts[4])
        tx = float(rt_parts[5])
        ty = float(rt_parts[6])
        tz = float(rt_parts[7])
        rt_mat = R.from_quat([[x, y, z, w]])
        RT_ctow = np.zeros((4,4))
        RT_ctow[0:3,0:3] = rt_mat.as_matrix()[0]
        RT_ctow[0:3,3] = np.array([tx,ty,tz])
        RT_ctow[3,3] = 1

        # wtoc because o3d visualization needs it in wtoc, standard pinhole param convention. See below function description for more info.
        RT_wtoc = convert_w_t_c(RT_ctow)

        create_o3d_param_and_viz_image(mesh, RT_wtoc, camera)

def save_poses_to_file_in_RIO_format(filename, sequence_num, list_of_rts):
    count = 0 
    lines_file = []
    for rt in list_of_rts:
        # print(rt[0:3,0:3])
        r = R.from_matrix(rt[0:3,0:3])
        (x,y,z,w) = r.as_quat()
        t = rt[0:3,3]
        str_rt = sequence_num + '/frame_{:06d}'.format(count)
        str_rt = str_rt + " " + str(w)
        str_rt = str_rt + " " + str(x)
        str_rt = str_rt + " " + str(y)
        str_rt = str_rt + " " + str(z)

        str_rt = str_rt + " " + str(t[0])
        str_rt = str_rt + " " + str(t[1])
        str_rt = str_rt + " " + str(t[2])

        # print(str_rt)
        lines_file.append(str_rt)
        count += 1

    with open(filename,'w') as f:
        for i in lines_file:
            f.write(i + '\n')
    print(f"poses written to filename: {filename}")

def save_poses_main(mesh_dir, camera_dir, dest_dir,sequence_num, filename=None):
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

    list_of_rts_ctow = []
    for hull_point in range(len(centroids_coordinates)):
        '''
        rt_given_lookat(lookat,location)
        '''
        RT_ctow = rt_given_lookat(sphere_center_coords, centroids_coordinates[hull_point])
        # print(RT)
        # RT_wtoc = convert_w_t_c(RT_ctow)
        list_of_rts_ctow.append(RT_ctow)

    if filename is None:
        filename = os.path.join(dest_dir, sequence_num + '.txt')

    save_poses_to_file_in_RIO_format(filename, sequence_num, list_of_rts_ctow)
    read_pose_file_and_viz(mesh, camera, filename)



if __name__ == '__main__':
    """
    file is saved as 
    "sequence_num.txt" unless specified

    Enter:
    mesh_dir(wherer mesh.obj is stored)
    Camera dir(where K matrix is)
    sequence number for which you want to save poses
    destination directory to save.
    """
    viz_pcd = True # False
    custom_dir = False

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

    #Sequence number and where visualisations are saved:
    # MAKE SURE IT IS SAVE WITHOUT '/'
    sequence_num = 'seq01_01'
    dest_dir = "temp_dir"

    #Following code should work properly if above files are okay.

    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    mesh_dir = os.path.join(mesh_dir, sequence_num)
    camera_dir = os.path.join(camera_dir, sequence_num)

    save_poses_main(mesh_dir, camera_dir, dest_dir,sequence_num)

    # final_poses = poses_for_places(viz_pcd, True)
    # synth_image(viz_pcd, False, device)
