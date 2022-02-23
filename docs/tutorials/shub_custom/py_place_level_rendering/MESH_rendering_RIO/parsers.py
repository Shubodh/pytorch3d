import numpy as np
import open3d as o3d
import json
from scipy.io import loadmat
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import yaml
import copy
import sys

def parse_pose_file_RIO(pose_file, param):
    # NOTE: In RIO pose files, convention is: camera to world. Obviously, it is pose of a robot,
    # so obviously it has to be (a camera) in world frame.
    # We want world to camera as our points are in world coordinates.
    #Reading the pose file
    with open(pose_file,'r') as f:
        pose_lines = f.readlines()
    # for row in pose_lines:
    #     print(row)
    pose_lines = [line.strip() for line in pose_lines]
    pose_lines = [line.split(' ') for line in pose_lines]
    pose_vals = [float(i) for line in pose_lines for i in line]
    RT_mat = np.array(pose_vals)
    RT_ctow = RT_mat.reshape((4,4))
    # NOTE: This RT is from  camera coordinate system to the world coordinate 
    # We want world to camera

    RT_wtoc = np.zeros((RT_ctow.shape))
    RT_wtoc[0:3,0:3] = RT_ctow[0:3,0:3].T
    RT_wtoc[0:3,3] = - RT_ctow[0:3,0:3].T @ RT_ctow[0:3,3]

    # print("DEBUG")
    # print(RT, RT_wtoc)
    RT_final = RT_wtoc

    param.extrinsic = RT_final 
    # print(param.extrinsic)
    return param, RT_final, RT_ctow



def parse_camera_file_RIO(camera_file):
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
    # print("camera model:" ,model)
    # print("img size", img_size)
    # print(K)
    #Set intrinsics here itself:
    param = o3d.camera.PinholeCameraParameters()
    param.intrinsic.set_intrinsics(width = img_size[1],
                                        height = img_size[0],
                                        fx = model[0],
                                        fy = model[1],
                                        cx = model[2],
                                        cy = model[3])
    #intrinsic = param.intrinsic.set_intrinsics(width = img_size[1],
    #                                                height = img_size[0],
    #                                                fx = model[0],
    #                                                fy = model[1],
    #                                                cx = model[2],
    #                                                cy = model[3])
    ## param.intrinsic = intrinsic
    ## print(img_size)
    ##print(intrinsic)
    ##print(param.intrinsic.intrinsic_matrix)
    return param, K, img_size
