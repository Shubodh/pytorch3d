import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
from scipy.spatial import Delaunay

from numpy.linalg import det, norm

# def rt_given_lookat(lookat,location):
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

    R = RT_ctow[0:3, 0:3]
    assert np.isclose(det(R), 1) #wrong currently
    assert np.allclose(R @ R.T, np.eye(3))

    return RT_ctow 

# def rt_given_lookat_new(lookat,location):
def rt_given_lookat(lookat,location, y_down = np.array([0,0,-1]) ):
    """
    # y_down = np.array([0,0,-1]) #down vector direction in world frame

    Input: 1. lookat like sphere_center i.e. center of room, 2. your current location from where you are looking
    Both are positions.

    Output: This outputs poses in ctow format, obvious choice because
    any robot poses are naturally in ctow format. 

    See Notion for full clarity: https://www.notion.so/saishubodh/Personal-notes-ALL-coordinate-frame-conventions-Habitat-Notebook-InLoc-RIO10-Pytorch3d-Common--01ac85553c324a06a63b1821be1a463f#ea75fd7aa6394cf2b06d1e10d75348ac
    """
    z_axis = np.array(lookat) - np.array(location)
    z = z_axis/np.linalg.norm(z_axis)

    # y_down = np.array([0,0,-1]) #down vector direction in world frame

    x_axis = np.cross(y_down, z)
    x = x_axis/np.linalg.norm(x_axis)

    y_axis = np.cross(z,x)
    y = y_axis/np.linalg.norm(y_axis)

    RT_ctow = np.zeros((4,4))
    RT_ctow[0:3,0] = x 
    RT_ctow[0:3,1] = y 
    RT_ctow[0:3,2] = z
    R = RT_ctow[0:3, 0:3]

    # assert Rotation matrix properties
    assert np.isclose(det(R), 1)
    assert np.allclose(R @ R.T, np.eye(3))
    # print(R)
    # sys.exit()

    RT_ctow[0:3,3] = np.array(location)
    RT_ctow[3,3] = 1 
    return RT_ctow 

#def viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, camera, dest_dir, mesh_dir, device):
def create_list_of_rts_for_all_places(pcd_hull, centroids_coordinates, sphere_center_coords, linspace_num):
    pcd_hull_copy = copy.deepcopy(pcd_hull)
    pcd_pts = np.asarray(pcd_hull_copy.points)

    # 3. Adding midpoints between consecutive centroids_coordinates. So if there are X centroids_coordinates, final total will be (2X-1).
    and_midpoint_points = (centroids_coordinates[1:] + centroids_coordinates[:-1]) / 2
    centroids_coordinates_and_midpoint_points = np.vstack((centroids_coordinates, and_midpoint_points))
    centroids_coordinates = centroids_coordinates_and_midpoint_points
    #print(f"hi debug: {centroids_coordinates.shape, centroids_coordinates, centroids_coordinates_and_midpoint_points}")
    list_of_rts = []
    for hull_point in range(len(centroids_coordinates)):
        # 1. NO SAMPLING: If you only want to render images looking from centroids_coordinates
        # list_of_rts.append(rt_given_lookat(sphere_center_coords, centroids_coordinates[hull_point]))

        # 2. SAMPLING: Sampling poses between every centroid and sphere center.
        sample_poses =  np.linspace(centroids_coordinates[hull_point], sphere_center_coords, num=linspace_num) # previously 4, 8
        sample_poses_except_sphere = sample_poses[:-1] #Want all poses except the last one, which is sphere_center itself.
        sample_poses_except_hull_pt = sample_poses[1:]
        # print(sample_poses, "\n", sample_poses_except_sphere, "\n", sample_poses_except_hull_pt, "\n")
        for i_coord in range(sample_poses_except_sphere.shape[0]):

            # 2. A: lookat sphere_center from sampled point
            if in_hull(sample_poses_except_sphere[i_coord], pcd_pts):
                list_of_rts.append(rt_given_lookat(sphere_center_coords, sample_poses_except_sphere[i_coord]))
            # 2. B: lookat sampled point from  sphere_center
            if in_hull(sample_poses_except_hull_pt[i_coord], pcd_pts):
                list_of_rts.append(rt_given_lookat(centroids_coordinates[hull_point], sample_poses_except_hull_pt[i_coord]))

            # OLD: Without checking if the point is in_hull
            # list_of_rts.append(rt_given_lookat(sphere_center_coords, sample_poses_except_sphere[i_coord]))
            # list_of_rts.append(rt_given_lookat(centroids_coordinates[hull_point], sample_poses_except_hull_pt[i_coord]))

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

def all_coords_from_mesh(mesh, ref_not_query):
    mesh_array = np.asarray(mesh.vertices)
    # room_center_z = (np.max(mesh_array[:,2]) - np.min(mesh_array[:,2])) / 2 # this is towards ceiling, not desirable as most images from robot are usually towards floor.
    room_center_z = np.average(mesh_array[:,2]) # is skewed towards floor, probably because of higher density below.
    print(f"Sphere center height: {room_center_z}")

    #Apply Convex Hull
    pcd_hull = convex_hull(mesh)
    colored_pcd_hull, centroids_coordinates = dbscan_clustering(pcd_hull)

    pcd_hull_copy = copy.deepcopy(pcd_hull)
    # ### fixing up coordinate for clusters
    pcd_hull_points = (np.asarray(pcd_hull_copy.points))
    sphere_center_height = room_center_z #1.48
    pcd_hull_points[:,2] = np.ones((pcd_hull_points[:,2]).shape) * sphere_center_height 
    sphere_center_coords = np.mean(pcd_hull_points, axis=0)

    # Automate this fix_up_coord_list using linspace and info from mesh max and min 
    # fix_up_coord_list = [-0.5] #0.5, 1, 1.5, 2, 2.5
    # fix_up_coord_list = [-0.5, 0, 0.5, 1, 1.5, 2, 2.5]
    if ref_not_query:
        linspace_num_across_room_height = 7 # np.min - 1.0 below because it has to look at floor. Visualize it, the ray has to pass through the floor.
        linspace_num_across_ray = 5
    else: 
        print("should change sampling later for QUERY")
        linspace_num_across_room_height = 7 # np.min - 1.0 below because it has to look at floor. Visualize it, the ray has to pass through the floor.
        linspace_num_across_ray = 5
        # linspace_num_across_room_height = 4
        # linspace_num_across_ray = 2



    fix_up_coord_arr =  np.linspace(np.min(mesh_array[:,2]) - 1.0, np.max(mesh_array[:,2]),   num=linspace_num_across_room_height) 
    fix_up_coord_list = list(fix_up_coord_arr) 


    # test_in_hull(centroids_coordinates, fix_up_coord_list, pcd_hull)

    return pcd_hull, centroids_coordinates, sphere_center_coords, fix_up_coord_list, linspace_num_across_ray
    
def test_in_hull(centroids_coordinates, fix_up_coord_list, pcd_hull):
    """
    Quick test of function in_hull(). Call this from all_coords_from_mesh() to test layer by layer. 
    So topmost and bottomost will be False, middle layers will be True.
    """
    pcd_hull_copy = copy.deepcopy(pcd_hull)
    pcd_pts = np.asarray(pcd_hull_copy.points)

    for fix_up_coord in fix_up_coord_list:
        centroids_coordinates[:,2] = np.ones((centroids_coordinates[:,2]).shape) * fix_up_coord
        # print(centroids_coordinates.shape, pcd_pts)
        
        print(in_hull(centroids_coordinates, pcd_pts))
    sys.exit()

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


if __name__ == '__main__':
    """
    Test if points in `p` are in `hull`. See in_hull() function for more details.
    """
    tested = np.random.rand(20,3)
    cloud  = np.random.rand(50,3)
    print(type(tested), type(cloud))

    print(in_hull(tested,cloud))
