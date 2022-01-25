import numpy as np
import open3d as o3d
import json
from scipy.io import loadmat
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

def load_pcd_mat(filename):
    # For loading files like 'sample/DUC_cutout_005_0_0.jpg.mat'
    xyz_file  = loadmat(Path(filename))["XYZcut"]
    rgb_file = loadmat(Path(filename))["RGBcut"]
    xyz_sp = (xyz_file.shape)
    xyz_file = (xyz_file.reshape((xyz_sp[0]*xyz_sp[1] ,3)))
    rgb_file = (rgb_file.reshape((xyz_sp[0]*xyz_sp[1] ,3)))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_file)
    pcd.colors = o3d.utility.Vector3dVector(rgb_file/255.0)
    return pcd

def load_view_point(pcd, img_size, param):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=img_size[0], width=img_size[1])
    print(img_size[0], img_size[1])
    ctr = vis.get_view_control()
#     print(param.intrinsic)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
    vis.run()
    image = vis.capture_screen_float_buffer()
    vis.destroy_window()
    return image


def synthesize_img_given_viewpoint(pcd, K, extrinsics, save=False):
    H = 1200
    W = 1600
    #print("K")
    #print(K)
    xyz = np.asarray(pcd.points)

    rvecs = np.zeros(3)
    cv2.Rodrigues(extrinsics[0:3,0:3], rvecs)
    #cv2.Rodrigues(extrinsics[0:3,0:3].T, rvecs)
    tvecs = np.zeros(3)
    tvecs = extrinsics[0:3,3]
    #tvecs = - extrinsics[0:3,0:3].T @ extrinsics[0:3,3]


    dist = np.zeros(5)
    print("Starting cv2.project:")
    xyz_T = xyz.T
    xyz_hom1 = np.vstack((xyz_T, np.ones(xyz_T[0].shape)))
    K_hom = np.vstack((K, np.zeros(K[0].shape)))
    K_hom = np.hstack((K_hom, np.array([[0,0,0,1]]).T))

    #print("1. X_G visualization")
    #viz_with_array_inp(xyz_hom1.T[:, :3],np.asarray(pcd.colors))
    xyz_hom1 = np.matmul(extrinsics, xyz_hom1) #xyz_hom1.shape: 4 * 11520000
    #print("2. X_L visualization")
    #viz_with_array_inp(xyz_hom1.T[:, :3],np.asarray(pcd.colors))

    #tf_ex = np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    #tf_ex_hom = np.vstack((tf_ex, np.zeros(tf_ex[0].shape)))
    #tf_ex_hom = np.hstack((tf_ex_hom, np.array([[0,0,0,1]]).T))
    #xyz_hom1 = np.matmul(tf_ex_hom, xyz_hom1)
    #print("3. X_L_corr visualization")
    #viz_with_array_inp(xyz_hom1.T[:, :3],np.asarray(pcd.colors))

    xy_img = np.matmul(K_hom, xyz_hom1)
    #print(np.nanmax(xy_img[0:2,:]), np.nanmin(xy_img[0:2,:]))
    xy_img = xy_img[0:2,:] / xy_img[2:3,:] 
    xy_imgcv = np.array(xy_img.T, dtype = np.int_)

    print("Done cv2.project:")

    #xy_imgcv, jac = cv2.projectPoints(xyz, rvecs, tvecs, K, dist)
    #xy_imgcv = np.array(xy_imgcv.reshape(xy_imgcv.shape[0], 2), dtype=np.int_)


#    W_valid = (xy_imgcv[:,0] >= 0) &  (xy_imgcv[:,0] < W)
#    H_valid = (xy_imgcv[:,1] >= 0) &  (xy_imgcv[:,1] < H)
#    #print(xy_imgcv[:,0].shape,"hi", np.nanmax(xy_imgcv, axis=0), "hii", xy_imgcv[0:10])
#    #print(xy_imgcv.shape)
#    final_valid = (H_valid  & W_valid)
#    #print(xy_imgcv[final_valid])
#    print(np.nanmax(xy_imgcv[final_valid], axis=0))
#    #print(np.argwhere(final_valid==False))
#
#
    pcd_colors = np.asarray(pcd.colors) * 255
    synth_img = np.ones((H, W, 3))  * 255

#    colors_re = pcd_colors.reshape((H,W,3))
#    colors_re = colors_re.T
#    pcd_colors = colors_re.reshape((H*W, 3))
    #print(f"xy_imgcv.shape, synth_img.shape, pcd_colors.shape: {xy_imgcv.shape}, {synth_img.shape}, {pcd_colors.shape}")
    #synth_img(xy_imgcv[:]) 
    for i in range(pcd_colors.shape[0]):
        # Be careful here: For xy_imgcv, (x,y) means x right first then y down.
        # Whereas for numpy array, (x, y) means x down first then y right.

        # 1. Ignore points with negative depth, i.e. ones behind the camera. 
        if xyz_hom1[2,i] > 0: # Make sure the xyz you're checking are in ego frame
            # 2. projected pixel must be between  [{0,W},{0,H}]
            if (xy_imgcv[i,0] >= 0) & (xy_imgcv[i,0] < W):
                if (xy_imgcv[i,1] >= 0) &  (xy_imgcv[i,1] < H):
                    #print(xy_imgcv[i], i)
                    synth_img[xy_imgcv[i,1], xy_imgcv[i,0]] = pcd_colors[i] #



    img = o3d.geometry.Image((synth_img).astype(np.uint8))
    #o3d.visualization.draw_geometries([img])
    if save:
        img_dest = os.path.join("sample_data", "synth_image_inloc.png")
        o3d.io.write_image(img_dest, img)
        print(f"image written to {img_dest}")
    return img