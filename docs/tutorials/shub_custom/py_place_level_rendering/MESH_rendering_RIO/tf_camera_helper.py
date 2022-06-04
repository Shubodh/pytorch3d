import numpy as np
import os
import yaml

from scipy.spatial.transform import Rotation as R
import sys

def convert_w_t_c(RT_ctow):
    """
    This is actually nothing but inverse transform. But for sake of clarity for our application,
    keeping its name w_to_c.

    input RT is transform from camera to world: i.e. robot poses.
    output: o3d visualizer requires pinhole camera parameters, which is naturally
    world to camera (standard projection equation). See load_view_point in o3d_helper.py for more details.
    """
    RT_wtoc = np.zeros((RT_ctow.shape))
    RT_wtoc[0:3,0:3] = RT_ctow[0:3,0:3].T
    RT_wtoc[0:3,3] = - RT_ctow[0:3,0:3].T @ RT_ctow[0:3,3]
    RT_wtoc[3,3] = 1
    return RT_wtoc

def moveback_tf_simple_given_pose(RT_wtoc, moveback_distance=0.5):
    """ 
    Basically, tf "current pose" by 0.5 m backward in egocentric view and return that new pose.
    "Current pose" is extrinsics in camera projection terminology, RT_wtoc.
    x = K [R t]_wtoc X_w = K X_c; where X_c = RT_wtoc @ X_w.
    RIO10 local frame convention: Z forward, X right, Y below.
    Now what we want to apply camera projection over X_c_new where 
    X_c_new = T_mb @ X_c; T_mb's R = [I] and t = [0, 0, +0.5], basically I want to see points 0.5m behind me also, so I will move them in front of me.
    X_c_new = T_mb @ RT_wtoc @ X_w. Therefore, new extrinsics would now be:
    RT_wtoc_new = T_mb @ RT_wtoc
    NOTE: DO remember X_c should be full point cloud, not trimmed point cloud.
    """
    T_mb = np.eye(4,4)
    T_mb[2,3] = moveback_distance
    RT_wtoc_new = T_mb @ RT_wtoc
    return RT_wtoc_new 

def move_rotate_leftright_tf_simple_given_pose(RT_wtoc, rotate_left_or_right="left"):
    """ 
    Basically, tf "current pose" by rotating L or R in egocentric view and return that new pose in global frame.
    "Current pose" is extrinsics in camera projection terminology, RT_wtoc.
    x = K [R t]_wtoc X_w = K X_c; where X_c = RT_wtoc @ X_w.
    RIO10 local frame convention: Z forward, X right, Y below.
    Now what we want to apply camera projection over X_c_new where 
    X_c_new = T_mb @ X_c; T_mb's R = [R(theta)] and t = [0, 0, 0], 

    R(theta) -- If turning left: Remember y-axis is facing below
        I will rotate points 90 deg such that left point are in front of me,
        So, [R.from_euler('y', 90..) @ X_c] will rotate points rightwards such that left point are in front. (Rotator-transform equivalence: 1. Vector or operator)

    X_c_new = T_mb @ RT_wtoc @ X_w. Therefore, new extrinsics would now be:
    RT_wtoc_new = T_mb @ RT_wtoc
    NOTE: DO remember X_c should be full point cloud, not trimmed point cloud.
    """
    R_left_obj = R.from_euler('y', 90, degrees=True) 
    R_left = R_left_obj.as_matrix() #should be [[0,0,1],[0,1,0],[-1,0,0]]
    R_right_obj = R.from_euler('y', -90, degrees=True)
    R_right = R_right_obj.as_matrix()

    T_mb = np.eye(4,4)
    if rotate_left_or_right == "left":
        T_mb[:3, :3] = R_left
    elif rotate_left_or_right == "right":
        T_mb[:3, :3] = R_right
    else:
        raise ValueError(f'Wrong argument {rotate_left_or_right}, should be left or right')

    RT_wtoc_new = T_mb @ RT_wtoc
    return RT_wtoc_new 

def move_rotate_leftright_tf_simple_given_pose_in_ctow(RT_ctow, rotate_left_or_right="left"):
    """ 
    Basically, tf "current pose" by rotating L or R in egocentric view and return that new pose in global frame.
    "Current pose" is extrinsics in camera projection terminology, RT_wtoc.
    x = K [R t]_wtoc X_w = K X_c; where X_c = RT_wtoc @ X_w.
    RIO10 local frame convention: Z forward, X right, Y below.
    Now what we want to apply camera projection over X_c_new where 
    X_c_new = T_mb @ X_c; T_mb's R = [R(theta)] and t = [0, 0, 0], 

    R(theta) -- If turning left: Remember y-axis is facing below
        I will rotate points 90 deg such that left point are in front of me,
        So, [R.from_euler('y', 90..) @ X_c] will rotate points rightwards such that left point are in front. (Rotator-transform equivalence: 1. Vector or operator)

    X_c_new = T_mb @ RT_wtoc @ X_w. Therefore, new extrinsics would now be:
    RT_wtoc_new = T_mb @ RT_wtoc
    NOTE: DO remember X_c should be full point cloud, not trimmed point cloud.
    """
    R_left_obj = R.from_euler('y', 90, degrees=True) 
    R_left = R_left_obj.as_matrix() #should be [[0,0,1],[0,1,0],[-1,0,0]]
    R_right_obj = R.from_euler('y', -90, degrees=True)
    R_right = R_right_obj.as_matrix()

    T_mb = np.eye(4,4)
    if rotate_left_or_right == "left":
        T_mb[:3, :3] = R_left
    elif rotate_left_or_right == "right":
        T_mb[:3, :3] = R_right
    else:
        raise ValueError(f'Wrong argument {rotate_left_or_right}, should be left or right')

    RT_wtoc = np.linalg.inv(RT_ctow)
    RT_wtoc_new = T_mb @ RT_wtoc
    return np.linalg.inv(RT_wtoc_new)

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
        # print("camera model:" ,model)
        # print("img size", img_size)
        # print("K", K)