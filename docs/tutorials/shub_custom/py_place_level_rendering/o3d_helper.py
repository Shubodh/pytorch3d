import open3d as o3d 

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