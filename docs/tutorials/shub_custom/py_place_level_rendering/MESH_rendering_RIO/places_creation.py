import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

#def create_rt(lookat,location):
def rt_given_lookat(lookat,location):
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
    """ 
    Given individual hull points and their labels, this function finds centroids of every label. Basically,
    if there are 19 clusters, this func returns 19 centroids of each cluster.
    """


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