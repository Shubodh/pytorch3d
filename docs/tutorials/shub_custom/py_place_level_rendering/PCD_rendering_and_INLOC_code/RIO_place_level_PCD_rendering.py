import numpy as np
import open3d as o3d
import json
from scipy.io import loadmat
import sys
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import yaml


from utils import synthesize_img_given_viewpoint, load_pcd_mat, load_view_point, parse_camera_file_RIO, parse_pose_file_RIO



if __name__=='__main__':
    out_msg = """ CURRENT STATUS / HOW TO KNOW ISSUE HAS BEEN FIXED: (Also see pcd_from_mesh function)
    load_view_point shows viewpoint at correct position precisely.
    however synthesize_img... still gives empty image. But this is not important as of now as 
    we want to do pytorch3d view rendering than pin hole projection. So skip this task, not needed."""

    print(out_msg)
    sample_path = "../sample_data"
#Ensure camera_file, pose_file, and rgb_file are set correctly and exist for further cells to work
    on_ada = False

    pose_file_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/seq01_01/"
    ada_prefix = "/scratch/saishubodh/" 
    pose_file_dir_ada = ada_prefix+ "RIO10_data/scene01/seq01/seq01_01/"
    if on_ada == True:
        pose_file_dir = pose_file_dir_ada

    # camera_file = os.path.join(pose_file_dir, 'camera.yaml')

    #change file num: 
    #added format code, so just
    #integer format
    num = 131
    camera_file = 'camera.yaml'
    pose_file = 'frame-{:06d}.pose.txt'.format(num)
    rgb_file = 'frame-{:06d}.color.jpg'.format(num)

    camera_file = os.path.join(pose_file_dir, camera_file)
    pose_file = os.path.join(pose_file_dir, pose_file)
    rgb_file = os.path.join(pose_file_dir, rgb_file)

    #print(camera_file, os.path.isfile(camera_file))
    #print(pose_file, os.path.isfile(pose_file))
    #print(rgb_file, os.path.isfile(rgb_file))
    img = plt.imread(rgb_file)
    plt.imshow(img)
    plt.show()

    # os.path.isfile(camera_file)
    # os.listdir(mesh_file_sample)

    #camera parsing
    param, K, img_size = parse_camera_file_RIO(camera_file)
    H, W = img_size
    param, RT, _ = parse_pose_file_RIO(pose_file, param)
    
    mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/models01/seq01_01/"
    mesh_dir_ada = ada_prefix+ "RIO10_data/scene01/seq01/seq01_01/"
    if on_ada == True:
        mesh_dir = mesh_dir_ada
    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh.obj"), True)
    #pcd = pcd_from_mesh(mesh)

    #mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir ,"labels.ply"))
    pcd = o3d.io.read_point_cloud(os.path.join(mesh_dir ,"labels.ply"))
    # print(mesh)
    # o3d.visualization.draw_geometries([mesh])
    # print(pcd)
    # o3d.visualization.draw_geometries([pcd])
    # img_viz = load_view_point(mesh, img_size, param)

    H, W = 240, 135
    print(H, W)
    img_synth = synthesize_img_given_viewpoint(pcd, K, RT, H, W, save=True)
    

    # #load_view_point(mesh, img_size, param)
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(img)
    # #ax2.imshow(img_synth)
    # plt.show()
