import numpy as np
import os
import yaml


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
    """
    T_mb = np.eye(4,4)
    T_mb[2,3] = moveback_distance
    RT_wtoc_new = T_mb @ RT_wtoc
    return RT_wtoc_new 

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