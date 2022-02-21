import numpy as np
import os
import yaml


def convert_w_t_c(RT_ctow):
    """
    input RT is transform from camera to world
    o3d requires world to camera
    """
    RT_wtoc = np.zeros((RT_ctow.shape))
    RT_wtoc[0:3,0:3] = RT_ctow[0:3,0:3].T
    RT_wtoc[0:3,3] = - RT_ctow[0:3,0:3].T @ RT_ctow[0:3,3]
    RT_wtoc[3,3] = 1
    return RT_wtoc

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