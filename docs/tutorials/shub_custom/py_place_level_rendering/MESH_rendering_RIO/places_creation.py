import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import sys

#def rt_given_lookat_planar(lookat,location):
def rt_given_lookat_planar(lookat,location):
    """
    This only works For planar cases, i.e. vector joining lookat and location are parallel to ground. See Notion link for more info.
    Next function is generalization of this.

    Input: 1. lookat like sphere_center i.e. center of room, 2. your current location from where you are looking
    Both are positions.

    Output: This outputs poses in ctow format, obvious choice because
    any robot poses are naturally in ctow format. 

    See Notion for full clarity: https://www.notion.so/saishubodh/Personal-notes-ALL-coordinate-frame-conventions-Habitat-Notebook-InLoc-RIO10-Pytorch3d-Common--01ac85553c324a06a63b1821be1a463f#ea75fd7aa6394cf2b06d1e10d75348ac
    """
    z_axis = np.array(lookat) - np.array(location)
    z = z_axis/np.linalg.norm(z_axis)
    y = np.array([0,0,-1])
    x = np.cross(y,z)
    RT_ctow = np.zeros((4,4))
    RT_ctow[0:3,0] = x 
    RT_ctow[0:3,1] = y 
    RT_ctow[0:3,2] = z
    RT_ctow[0:3,3] = np.array(location)
    RT_ctow[3,3] = 1 
    return RT_ctow 

# def rt_given_lookat_new(lookat,location):
def rt_given_lookat(lookat,location):
    """
    Input: 1. lookat like sphere_center i.e. center of room, 2. your current location from where you are looking
    Both are positions.

    Output: This outputs poses in ctow format, obvious choice because
    any robot poses are naturally in ctow format. 

    See Notion for full clarity: https://www.notion.so/saishubodh/Personal-notes-ALL-coordinate-frame-conventions-Habitat-Notebook-InLoc-RIO10-Pytorch3d-Common--01ac85553c324a06a63b1821be1a463f#ea75fd7aa6394cf2b06d1e10d75348ac
    """
    z_axis = np.array(lookat) - np.array(location)
    z = z_axis/np.linalg.norm(z_axis)

    y_down = np.array([0,0,-1]) #down vector direction in world frame

    x = np.cross(y_down, z)
    y = np.cross(z,x)

    RT_ctow = np.zeros((4,4))
    RT_ctow[0:3,0] = x 
    RT_ctow[0:3,1] = y 
    RT_ctow[0:3,2] = z
    RT_ctow[0:3,3] = np.array(location)
    RT_ctow[3,3] = 1 
    return RT_ctow 

#def viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, camera, dest_dir, mesh_dir, device):
def create_list_of_rts_for_all_places(centroids_coordinates, sphere_center_coords, linspace_num):
    # 3. Adding midpoints between consecutive centroids_coordinates. So if there are X centroids_coordinates, final total will be (2X-1).
    and_midpoint_points = (centroids_coordinates[1:] + centroids_coordinates[:-1]) / 2
    centroids_coordinates_and_midpoint_points = np.vstack((centroids_coordinates, and_midpoint_points))
    centroids_coordinates = centroids_coordinates_and_midpoint_points
    #print(f"hi debug: {centroids_coordinates.shape, centroids_coordinates, centroids_coordinates_and_midpoint_points}")
    #sys.exit()
    list_of_rts = []
    for hull_point in range(len(centroids_coordinates)):
        # 1. NO SAMPLING: If you only want to render images looking from centroids_coordinates
        # list_of_rts.append(rt_given_lookat(sphere_center_coords, centroids_coordinates[hull_point]))

        # 2. SAMPLING: Sampling poses between every centroid and sphere center.
        sample_poses =  np.linspace(centroids_coordinates[hull_point], sphere_center_coords, num=linspace_num) # previously 4, 8
        sample_poses = sample_poses[:-1] #Want all poses except the last one, which is sphere_center itself.
        for i_coord in range(sample_poses.shape[0]):
            # 2. A: lookat sphere_center from sampled point
            list_of_rts.append(rt_given_lookat(sphere_center_coords, sample_poses[i_coord]))
            # 2. B: lookat sampled point from  sphere_center
            list_of_rts.append(rt_given_lookat(sample_poses[i_coord], sphere_center_coords))


        #Below for loop is giving NaNs. Skipping for now
        # for pose in sample_poses:
        #     #Here RT describes the location of the camera
        #     RT = rt_given_lookat(sphere_center_coords, pose)
        #     list_of_rts.append(RT)

    # viz_image(list_of_rts, camera, dest_dir, mesh_dir, device)
    return list_of_rts

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

def all_coords_from_mesh(mesh):
    mesh_array = np.asarray(mesh.vertices)
    room_center_z = (np.max(mesh_array[:,2]) - np.min(mesh_array[:,2])) / 2 #np.average(mesh_array[:,2]) is skewed towards floor, probably because of higher density below.


    #Apply Convex Hull
    pcd_hull = convex_hull(mesh)
    colored_pcd_hull, centroids_coordinates = dbscan_clustering(pcd_hull)

    # ### fixing up coordinate for clusters
    pcd_hull_points = (np.asarray(pcd_hull.points))
    sphere_center_height = room_center_z #1.48
    pcd_hull_points[:,2] = np.ones((pcd_hull_points[:,2]).shape) * sphere_center_height 
    sphere_center_coords = np.mean(pcd_hull_points, axis=0)

    # Automate this fix_up_coord_list using linspace and info from mesh max and min 
    # fix_up_coord_list = [-0.5] #0.5, 1, 1.5, 2, 2.5
    # fix_up_coord_list = [-0.5, 0, 0.5, 1, 1.5, 2, 2.5]
    linspace_num_across_room_height = 7 # np.min - 0.5 below because it has to look at floor. Visualize it, the ray has to pass through the floor.
    fix_up_coord_arr =  np.linspace(np.min(mesh_array[:,2]) - 0.5, np.max(mesh_array[:,2]),   num=linspace_num_across_room_height) 
    fix_up_coord_list = list(fix_up_coord_arr) 

    linspace_num_across_ray = 5
    return centroids_coordinates, sphere_center_coords, fix_up_coord_list, linspace_num_across_ray