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

from pytorch3d_utils import render_py3d_img
from tf_helper import convert_w_t_c


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

def o3dframe_from_coords(RT, color=[1, 0.706, 0], radius = 0.2):
    frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=1)
    # sphere = copy.deepcopy(sphere_mesh).translate(coords)
    # sphere.paint_uniform_color(color)
    print("RT", RT)
    frame_mesh.transform(RT)
    return frame_mesh

def o3dsphere_from_coords(coords, color=[1, 0.706, 0], radius = 0.2):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere = copy.deepcopy(sphere_mesh).translate(coords)
    sphere.paint_uniform_color(color)

    return sphere 

def create_rt(lookat,location):
    print("TODO5: I think Rotation should be transpose of what it is. Check visually.")
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


def viz_image(RT_list, camera, dest_dir, mesh_dir, device):
    """
    Creates a bunch of images based on RT_list
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
        render_py3d_img(img_size, param, dest_file, mesh_dir, device)

def viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, camera, dest_dir, mesh_dir, device):
    # Visualise for particular centroid
    # in this case we will visualise for
    # number 10
    list_of_rts = []
    for hull_point in range(len(centroids_coordinates)):
        # print("centroid, coordinates", centroids_coordinates[hull_point], sphere_center_coords)
        #sampling points along center and convex hull point location
        #sample_poses =  np.linspace(centroids_coordinates[hull_point], sphere_center_coords, num=5)

        #list_of_rts = []
        
        #create_rt(lookat,location)
        list_of_rts.append(create_rt(sphere_center_coords, centroids_coordinates[hull_point]))
        #Below for loop is giving NaNs. Skipping for now
        # for pose in sample_poses:
        #     #Here RT describes the location of the camera
        #     RT = create_rt(sphere_center_coords, pose)
        #     list_of_rts.append(RT)

    #print(list_of_rts)
    viz_image(list_of_rts, camera, dest_dir, mesh_dir, device)
        
def synth_image(viz_pcd=False, custom_dir=False, device=None):
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

    viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, camera, dest_dir, mesh_dir,device)


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
    synth_image(viz_pcd, False, device)

