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
    print(img_size[0], img_size[1])
    ctr = vis.get_view_control()
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    vis.run()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    return image

def convert_w_t_c(RT_ctow):
	"""
	input RT is transform from camera to world
	o3d requires world to camera
	"""
	RT_wtoc = np.zeros((RT_ctow.shape))
	RT_wtoc[0:3,0:3] = RT_ctow[0:3,0:3].T
	RT_wtoc[0:3,3] = - RT_ctow[0:3,0:3].T @ RT_ctow[0:3,3]
	return RT_wtoc

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

	param.extrinsic = convert_w_t_c(RT)
	load_view_point(mesh, img_size, param)
	



def viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, viz_pcd, camera):
	# Visualise for particular centroid
	# in this case we will visualise for
	# number 10
	test_num = 10
	print("centroid, coordinates", centroids_coordinates[test_num], sphere_center_coords)

	#sampling points along center and convex hull point location
	sample_poses =  np.linspace(centroids_coordinates[test_num], sphere_center_coords, num=4)
	

	poses_list = [mesh]
	for i_cent in range(sample_poses.shape[0]):
		sphere = o3dsphere_from_coords(sample_poses[i_cent], color=[1, 0.206, 0.7], radius=0.1)
		poses_list.append(sphere)

	#Camera is located here
	sphere_center = o3dsphere_from_coords(sphere_center_coords, color=[0.5, 0.706, 0], radius=0.1)
	
	#Camera is looking here
	given = o3dsphere_from_coords(centroids_coordinates[test_num], color=[0.5, 0.706, 0], radius=0.1)
	# poses_list.append(sphere_center)
	# poses_list.append(given)
	
	#plot as mesh. This transform is camera to world
	#for image synth purposes, we will have to do world to camera

	RT = create_rt(centroids_coordinates[test_num],sphere_center_coords)
	camera_center = o3dframe_from_coords(RT, color=[0.6, 0.706, 1], radius=0.2)
	poses_list.append(camera_center)


	# visualise_camera(mesh, K, RT)
	if viz_pcd:
		o3d.visualization.draw_geometries(poses_list)

	viz_image(mesh, RT, camera)
	
	#if we want other way round
	RT = create_rt(sphere_center_coords, centroids_coordinates[test_num])
	viz_image(mesh, RT, camera)
	return 


	

def synth_image(viz_pcd=False, custom_dir=False):
	#Reading data paths
	mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/models01/seq01_01/"
	camera_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/seq01_01/"
   
	ada = not viz_pcd
	if ada==True:
		mesh_dir = "/scratch/saishubodh/RIO10_data/scene01/models01/seq01_01/"
		camera_dir = "scratch/saishubodh/RIO10_data/scene01/seq01/seq01_01/"

	if custom_dir:
		mesh_dir = "../../../../../scene01/models01/seq01_01/"
		camera_dir = "../../../../../scene01/seq01/seq01_01/"

	mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, "mesh.obj"), True)

	#Apply Convex Hull
	pcd_hull = convex_hull(mesh)
	colored_pcd_hull, centroids_coordinates = dbscan_clustering(pcd_hull)

	# ### fixing up coordinate for clusters
	pcd_hull_points = (np.asarray(pcd_hull.points))
	pcd_hull_points[:,2] = np.ones((pcd_hull_points[:,2]).shape)
	sphere_center_coords = np.mean(pcd_hull_points, axis=0)
	
	centroids_coordinates[:,2] = np.ones((centroids_coordinates[:,2]).shape)

	camera = camera_params(camera_dir)
	camera.set_intrinsics()
	viz_points_cam(centroids_coordinates, sphere_center_coords, mesh, viz_pcd, camera)


if __name__ == '__main__':
	"""
	z - blue
	x - red
	y - green

	z - look at
	y - down
	x - cross
	"""
	viz_pcd = True
	# final_poses = poses_for_places(viz_pcd, True)
	synth_image(viz_pcd, True)