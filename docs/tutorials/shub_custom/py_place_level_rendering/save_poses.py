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

class camera_params(object):
    """docstring for camera_params"""
    def __init__(self, camera_dir):
        super(camera_params, self).__init__()
        self.camera_dir = camera_dir
        self.img_size = None
        self.K = None
        self.RT = None
        self.model = None

    def set_intrinsics(self):
        camera_file = os.path.join(self.camera_dir, 'camera.yaml')
        #camera parsing
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
        K[2,2] = 1
        self.K = K 
        self.img_size = img_size
        self.model = model
        print("camera model:" ,model)
        print("img size", img_size)
        print("K", K)

        

def convex_hull(mesh):

    ### 1. Convex hull

    hull, _ = mesh.compute_convex_hull()
    hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
    hull_ls.paint_uniform_color((1, 0, 0))
    #o3d.visualization.draw_geometries([mesh, hull_ls])

    pcd_hull = o3d.geometry.PointCloud()
    pcd_hull.points = hull_ls.points
    pcd_hull.colors = o3d.utility.Vector3dVector(np.ones((np.asarray(hull_ls.points)).shape)*0)
    # o3d.visualization.draw_geometries([mesh, pcd_hull])

    return pcd_hull

def find_centroid_coordinates(pcd, labels):

    pcd_np = np.asarray(pcd.points)
    dict_labels = {}
    for v, k in enumerate(labels):
        dict_labels.setdefault(k, [])
        dict_labels[k].append(v)
    #print(dict_labels)

    centroids_coordinates = np.zeros((len(dict_labels), 3))

    for label_id, pointset_list in dict_labels.items():
        centroids_coordinates[label_id] = np.mean(pcd_np[pointset_list], axis=0)
    # print(centroids_coordinates)
    return centroids_coordinates

def dbscan_clustering(pcd_hull):

    #with o3d.utility.VerbosityContextManager(
    #        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd_hull.cluster_dbscan(eps=0.5, min_points=2, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")

    centroids_coordinates = find_centroid_coordinates(pcd_hull, labels)

    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd_hull.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd_hull, centroids_coordinates



def create_rt(lookat,location):
    z_axis = np.array(lookat) - np.array(location)
    z = z_axis/np.linalg.norm(z_axis)
    y = np.array([0,0,-1])
    x = np.cross(y,z)
    RT = np.zeros((4,4))
    RT[0:3,0] = x 
    RT[0:3,1] = y 
    RT[0:3,2] = z
    RT[0:3,3] = np.array(location)
    RT[3,3] = 1 
    return RT 

def convert_w_t_c(RT_ctow):
    """
    input RT is transform from camera to world
    o3d requires world to camera
    """
    RT_wtoc = np.zeros((RT_ctow.shape))
    RT_wtoc[0:3,0:3] = RT_ctow[0:3,0:3].T
    RT_wtoc[0:3,3] = - RT_ctow[0:3,0:3].T @ RT_ctow[0:3,3]
    RT_wtoc[3,3] = 1
    return RT_wtoc


        

def create_rt(lookat,location):
    '''
    Gives a rotation matrix that converts from camera to world

    Physically, this represents the transform to bring the camera to its
    position in the world frame.

    Renders do not use this, so the above function converts it
    '''
    z_axis = np.array(lookat) - np.array(location)
    z = z_axis/np.linalg.norm(z_axis)
    y = np.array([0,0,-1])
    x = np.cross(y,z)
    RT = np.zeros((4,4))
    RT[0:3,0] = x 
    RT[0:3,1] = y 
    RT[0:3,2] = z
    RT[0:3,3] = np.array(location)
    RT[3,3] = 1 
    return RT 


def load_view_point(pcd, img_size, param):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=img_size[0], width=img_size[1])
    # print(img_size[0], img_size[1])
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

    # param.extrinsic = convert_w_t_c(RT)
    param.extrinsic = RT

    load_view_point(mesh, img_size, param)



def verify_file(mesh, camera, filename):
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
        R_temp = np.zeros((4,4))
        R_temp[0:3,0:3] = rt_mat.as_matrix()[0]
        R_temp[0:3,3] = np.array([tx,ty,tz])
        R_temp[3,3] = 1
        viz_image(mesh, R_temp, camera)



def save_to_file(filename, sequence_num, list_of_rts):
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




def save_poses(mesh_dir, camera_dir, dest_dir,sequence_num, filename=None):
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

    list_of_rts = []
    for hull_point in range(len(centroids_coordinates)):
        '''
        create_rt(lookat,location)
        '''
        RT = create_rt(sphere_center_coords, centroids_coordinates[hull_point])
        # print(RT)
        RT = convert_w_t_c(RT)
        list_of_rts.append(RT)

    if filename is None:
        filename = os.path.join(dest_dir, sequence_num + '.txt')

    save_to_file(filename, sequence_num, list_of_rts)
    verify_file(mesh, camera, filename)



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

    #Reading data paths
    #To edit:
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

    save_poses(mesh_dir, camera_dir, dest_dir,sequence_num)

    # final_poses = poses_for_places(viz_pcd, True)
    # synth_image(viz_pcd, False, device)
