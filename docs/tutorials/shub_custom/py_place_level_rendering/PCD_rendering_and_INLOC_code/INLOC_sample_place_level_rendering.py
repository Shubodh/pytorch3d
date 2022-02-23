import numpy as np
import open3d as o3d
import json
from scipy.io import loadmat
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import yaml


from utils import synthesize_img_given_viewpoint, load_pcd_mat


if __name__=='__main__':
    print("NOTE: Place level rendering not yet implemented for INLOC, this code currently is just synthesizing image")
    sample_path = "../sample_data"
    base_path = "/media/shubodh/DATA/OneDrive/rrc_projects/2021/graph-based-VPR/Hierarchical-Localization/datasets/inloc_small/cutouts_imageonly/DUC1/005/"
    json_file = os.path.join(sample_path, "p3p_005.json")
    og_img = plt.imread(os.path.join(base_path, "DUC_cutout_005_0_0.jpg"))
    pcd_file = os.path.join(base_path, "DUC_cutout_005_0_0.jpg.mat")


    pcd = load_pcd_mat(pcd_file)

    vpt_json = json.load(open(json_file))
    extrinsics = np.array(vpt_json['extrinsic']).reshape(4,4).T
    K = np.array(vpt_json['intrinsic']['intrinsic_matrix']).reshape(3,3).T

    img = synthesize_img_given_viewpoint(pcd, K, extrinsics)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(og_img)
    ax2.imshow(img)
    plt.show()