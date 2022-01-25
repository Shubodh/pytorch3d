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


from utils import synthesize_img_given_viewpoint, load_pcd_mat

def parse_camera_file(camera_file):
    with open(camera_file, 'r') as f:
        yaml_file = yaml.load(f, Loader=yaml.FullLoader)
        
    intrinsics = yaml_file['camera_intrinsics']
    img_size = (intrinsics['height'],intrinsics['width']) #H,W
    model = intrinsics['model']
    K = np.zeros((3,3))
    K[0,0] = model[0]
    K[1,1] = model[1]
    K[0,2] = model[2]
    K[1,2] = model[3]
    print("camera model:" ,model)
    print("img size", img_size)
    # print(K)
    #Set intrinsics here itself:
    param = o3d.camera.PinholeCameraParameters()
    intrinsic = param.intrinsic.set_intrinsics(width = img_size[1],
                                                    height = img_size[0],
                                                    fx = model[0],
                                                    fy = model[1],
                                                    cx = model[2],
                                                    cy = model[3])
    # param.intrinsic = intrinsic
    # print(img_size)
    print(param.intrinsic.intrinsic_matrix)
    return param

def parse_pose_file(pose_file, param):
    #Reading the pose file
    with open(pose_file,'r') as f:
        pose_lines = f.readlines()
    for row in pose_lines:
        print(row)
    pose_lines = [line.strip() for line in pose_lines]
    pose_lines = [line.split(' ') for line in pose_lines]
    pose_vals = [float(i) for line in pose_lines for i in line]
    RT_mat = np.array(pose_vals)
    RT = RT_mat.reshape((4,4))
    # print(param.extrinsic)
    param.extrinsic = RT
    print(param.extrinsic)
    return param


if __name__=='__main__':
    sample_path = "../sample_data"
#Ensure camera_file, pose_file, and rgb_file are set correctly and exist for further cells to work

    pose_file_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/seq01_01/"
    camera_file = os.path.join(pose_file_dir, 'camera.yaml')

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

    print(camera_file, os.path.isfile(camera_file))
    print(pose_file, os.path.isfile(pose_file))
    print(rgb_file, os.path.isfile(rgb_file))
    img = plt.imread(rgb_file)
    #plt.imshow(img)
    #plt.show()

    # os.path.isfile(camera_file)
    # os.listdir(mesh_file_sample)

    #camera parsing
    param = parse_camera_file(camera_file)
    param = parse_pose_file(pose_file, param)
    
    sys.exit()
    # CURRENTLY CODING

    mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/models01/seq01_01/"
    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh.obj"), True)
    # pcd_RIO = mesh_RIO.sample_points_uniformly(number_of_points=50000)

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices #Try sampling later instead (either sample or poisson)
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    # mesh2 = o3d.io.read_triangle_mesh(os.path.join(mesh_dir ,"labels.ply"))
    print(mesh.has_vertex_colors(), np.asarray(mesh.vertex_colors))
    # print(np.asarray((mesh.textures)))
    print(mesh)
    print(pcd)
    o3d.visualization.draw_geometries([mesh])
    o3d.visualization.draw_geometries([pcd])

    # Here::: start here
    img_viz = load_view_point(mesh, img_size, param)
    img_synth = synthesize_img_given_viewpoint(pcd, viewpoint_json)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(np.asarray(image))
    ax2.imshow(img)
    plt.show()