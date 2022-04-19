import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys
import argparse


def o3dglobalframe_from_coords(RT, size=0.2):
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=size)
    coord_mesh.transform(RT)
    return coord_mesh

def o3dsphere_from_coords(RT,color,radius = 0.02):
    # color = np.random.rand(3)
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere_mesh.paint_uniform_color(color)
    sphere_mesh.transform(RT)
    return sphere_mesh


def get_RT(pose_file):
    #Reading the pose file
    with open(pose_file,'r') as f:
        pose_lines = f.readlines()


    pose_lines = [line.strip() for line in pose_lines]
    pose_lines = [line.split(' ') for line in pose_lines]
    pose_vals = [float(i) for line in pose_lines for i in line]
    RT_mat = np.array(pose_vals)
    RT_mat = RT_mat.reshape((4,4))
    
    return RT_mat

def plot_poses(camera_dir, mesh_dir, sequence_num, interval=5, points_viz = True):
    """
    Visualizes poses in a scene.
    args:
        camera_dir -> directory where sequences are recorded
        sequence_num -> directory for sequence
        interval -> how many frames to skip
        points_viz -> spheres instead of frames
    TODO: add a dest path option
    """
    seq_dir = os.path.join(camera_dir, sequence_num)
    mesh_dir = os.path.join(mesh_dir, sequence_num)
    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh.obj"), True)


    list_files = os.listdir(seq_dir)
    total_files = len(list_files)

    total_frames = total_files - 1 #1 file is camera file
    total_frames = total_frames // 3
    print(total_frames)

    poses_files = ["frame-{:06d}.pose.txt".format(i) for i in range(total_frames)]
    poses_files = [os.path.join(seq_dir, i) for i in poses_files]

    
    

    # Sanity check
    # print(poses_files)
    # for i in poses_files:
    #     print(i, os.path.isfile(i))
    list_meshes_viz = []
    #add big starting
    # print(poses_files[0])
    # sys.exit()
    RT = get_RT(poses_files[0])

    if points_viz:
        pose_point = o3dsphere_from_coords(RT, color=np.random.rand(3), radius=0.1)
    else:
        pose_point = o3dglobalframe_from_coords(RT, size = 0.8)
    list_meshes_viz.append(pose_point)
    color = np.random.rand(3)
    count = 1
    for pose in poses_files[1:]:
        if count%interval == 0:
            RT = get_RT(pose)
            if points_viz:
                pose_point = o3dsphere_from_coords(RT, color=color)
            else:
                pose_point = o3dglobalframe_from_coords(RT)

            list_meshes_viz.append(pose_point)
            # print(pose, RT)
        count += 1

    list_meshes_viz.append(mesh)
    o3d.visualization.draw_geometries(list_meshes_viz)

    

if __name__ == '__main__':
    """
    Visualizes GT poses in a scene.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, required=True) #01
    parser.add_argument('--ref_or_query', type=str, required=True) # Acceptable args: 'ref' / 'query'
    #If reference, do "--ref_or_query ref". If query, anything else.
    args = parser.parse_args()

    scene_id = args.scene_id
    ref_not_query = args.ref_or_query
    ref_not_query = (ref_not_query=="ref")

    custom_dir = False
    viz_pcd = True

    mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
    camera_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"

    ada = not viz_pcd
    if ada:
        mesh_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
        camera_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"

    if custom_dir:
        mesh_dir = "../../../../../scene" + scene_id + "/models" + scene_id + "/"
        camera_dir = "../../../../../scene" + scene_id + "/seq" + scene_id + "/"
    # MAKE SURE IT IS SAVE WITHOUT '/'
    if ref_not_query:
        sequence_num = 'seq' + scene_id + '_01/'
    else:
        sequence_num = 'seq' + scene_id + '_02/'

    #interval: how many frames you want to skip
    #points_viz: if set to true you visualise points, 
    #instead of co-ord frames
    if ref_not_query:
        plot_poses(camera_dir, mesh_dir, sequence_num, interval=5, points_viz=True)
    plot_poses(camera_dir, mesh_dir, sequence_num, interval=15, points_viz=False)
