import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys

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
    # print(dict_labels)

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

def o3dsphere_from_coords(coords, color=[1, 0.706, 0], radius = 0.2):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere = copy.deepcopy(sphere_mesh).translate(coords)
    sphere.paint_uniform_color(color)

    return sphere 

def o3dglobalframe_from_coords(coords=[0,0,0]):
    coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=1)

    T = np.eye(4)
    # T[:3, :3] = coord_mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0,  np.pi ))
    T[:3, :3] = coord_mesh.get_rotation_matrix_from_xyz((np.pi / 2, -np.pi/4,  np.pi ))
    T[0:3,3] = coords
    coord_mesh.transform(T)
    # ## mesh_t = copy.deepcopy(coord_mesh).transform(T)

    # coord_mesh.translate(coords)
    # coord_mesh.rotate(R, center=coords)

    # coord_pcd =  coord_mesh.sample_points_uniformly(number_of_points=500)
    # o3d.visualization.draw_geometries([coord_mesh])
    return coord_mesh, np.linalg.inv(T)

def viz_centroids_and_center(centroids_coordinates, sphere_center_coords, mesh):
    # print(f"{centroids_coordinates.shape}")

    centroids_coordinates[:,2] = np.ones((centroids_coordinates[:,2]).shape)
    cents_list = []
    for i_cent in range(centroids_coordinates.shape[0]):
        sphere = o3dsphere_from_coords(centroids_coordinates[i_cent], color=[1, 0.706, 0.3], radius=0.3)
        cents_list.append(sphere)
    # print("viz pcd_hull")
    # o3d.visualization.draw_geometries([pcd_hull])
    # print("viz colored_pcd_hull")
    # o3d.visualization.draw_geometries([pcd_hull])


    sphere_center = o3dsphere_from_coords(sphere_center_coords, color=[0.5, 0.706, 0], radius=0.2)

    # coord_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=1)
    # coord_pcd =  coord_mesh.sample_points_uniformly(number_of_points=500)
    # o3d.visualization.draw_geometries([pcd_hull, mesh, coord_mesh,sphere])
    cents_list.extend([mesh, sphere_center])
    o3d.visualization.draw_geometries(cents_list)

def find_final_poses_from_centroids_and_center(centroids_coordinates, sphere_center_coords, mesh):
    # print(centroids_coordinates[0], sphere_center_coords)
    sample_poses =  np.linspace(centroids_coordinates[10], sphere_center_coords, num=5)
    # hi =(centroids_coordinates[0]+ sphere_center_coords ) /2
    # hii = (hi + sphere_center_coords)/2
    # hiii = (centroids_coordinates[0] + hi)/2

    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)

    poses_list = []
    for i_cent in range(sample_poses.shape[0]):
        sphere = o3dsphere_from_coords(sample_poses[i_cent], color=[1, 0.206, 0.7], radius=0.2)
        poses_list.append(sphere)

    sphere_center = o3dsphere_from_coords(sphere_center_coords, color=[0.5, 0.706, 0], radius=0.3)
    given = o3dsphere_from_coords(centroids_coordinates[0], color=[0.5, 0.706, 0], radius=0.3)

    final_tf_values = []
    for i_cent in range(sample_poses.shape[0]):
        frame, T = o3dglobalframe_from_coords(sample_poses[i_cent])
        poses_list.append(frame)
        final_tf_values.append(T)
    # poses_list=[]
    poses_list.extend([mesh, sphere_center, given])
    # o3d.visualization.draw_geometries(poses_list)
    return final_tf_values
    
def poses_for_places():
    mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/models01/seq01_01/"
   
    ada = True
    if ada==True:
        mesh_dir = "/scratch/saishubodh/RIO10_data/scene01/models01/seq01_01/"

    mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh.obj"), True)
    # o3d.visualization.draw_geometries([mesh])

    pcd_hull = convex_hull(mesh)
    colored_pcd_hull, centroids_coordinates = dbscan_clustering(pcd_hull)

    # ### fixing up coordinate for clusters
    pcd_hull_points = (np.asarray(pcd_hull.points))
    pcd_hull_points[:,2] = np.ones((pcd_hull_points[:,2]).shape)
    sphere_center_coords = np.mean(pcd_hull_points, axis=0)
    # viz_centroids_and_center(centroids_coordinates, sphere_center_coords, mesh)

    centroids_coordinates[:,2] = np.ones((centroids_coordinates[:,2]).shape)
    final_poses = find_final_poses_from_centroids_and_center(centroids_coordinates, sphere_center_coords,mesh)
    return final_poses

if __name__ == '__main__':
    final_poses = poses_for_places()
    print(final_poses)
