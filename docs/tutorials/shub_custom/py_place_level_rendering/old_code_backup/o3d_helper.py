import open3d as o3d 
import copy

def o3dframe_from_coords(RT, color=[1, 0.706, 0], radius = 0.2):
    frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(origin=[0, 0, 0], size=1)
    # sphere = copy.deepcopy(sphere_mesh).translate(coords)
    # sphere.paint_uniform_color(color)
    #print("RT", RT)
    frame_mesh.transform(RT)
    return frame_mesh

def o3dsphere_from_coords(coords, color=[1, 0.706, 0], radius = 0.2):
    sphere_mesh = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere = copy.deepcopy(sphere_mesh).translate(coords)
    sphere.paint_uniform_color(color)
    return sphere 


def load_view_point(pcd, img_size, param):
    """
    Here, param's extrinsic is in wtoc. This is intrinsic params, basically the convention in 
    standard pinhole camera equation.
    """
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

def create_o3d_param_and_viz_image(mesh, RT_wtoc, camera):
    model = camera.model
    img_size = camera.img_size
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic.set_intrinsics(width = img_size[1],
                                                height = img_size[0],
                                                fx = model[0],
                                                fy = model[1],
                                                cx = model[2],
                                                cy = model[3])

    # param.extrinsic = convert_w_t_c(RT_wtoc)
    param.extrinsic = RT_wtoc
    load_view_point(mesh, img_size, param)


def viz_centroids_and_center(centroids_coordinates, sphere_center_coords, mesh):
    # See usage of this function in file output_places_poses_RIO_convex_hull.py
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