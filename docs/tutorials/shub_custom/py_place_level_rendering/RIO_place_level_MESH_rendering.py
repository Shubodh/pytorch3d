import numpy as np
import open3d as o3d
import json
from scipy.io import loadmat
import sys
import os
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml

import torch
import torch.nn.functional as F

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

from pytorch3d.utils import cameras_from_opencv_projection

from utils import synthesize_img_given_viewpoint, load_pcd_mat, load_view_point, parse_camera_file_RIO, parse_pose_file_RIO
from output_places_poses_RIO_convex_hull import poses_for_places

#New custom utils functions
from pytorch3d_utils import RtK_in_torch_format, lights_given_position

def main(img_ids, save_imgs):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    #change file num: 
    #added format code, so just
    #integer format

    # seq01_XX = ["01", "02"]
    seq01_XX = ["01"]
    for seq_id in seq01_XX:
        for num in img_ids:

            pose_file_dir = "/scratch/saishubodh/RIO10_data/scene01/seq01/seq01_" + seq_id +"/"
            camera_file = os.path.join(pose_file_dir, 'camera.yaml')

            camera_file = 'camera.yaml'
            pose_file = 'frame-{:06d}.pose.txt'.format(num)
            rgb_file = 'frame-{:06d}.color.jpg'.format(num)

            camera_file = os.path.join(pose_file_dir, camera_file)
            pose_file = os.path.join(pose_file_dir, pose_file)
            rgb_file = os.path.join(pose_file_dir, rgb_file)

            print(camera_file, os.path.isfile(camera_file))
            print(pose_file, os.path.isfile(pose_file))
            print(rgb_file, os.path.isfile(rgb_file))
            img = plt.imread(rgb_file)
            # plt.imshow(img)
            # plt.show()

            o3d_param, K, img_size =  parse_camera_file_RIO(camera_file)
            o3d_param, RT, RT_ctow = parse_pose_file_RIO(pose_file, o3d_param)
            RT_wtoc = RT

            print(f"H & W: {img_size}, \n K:\n{K}, \n tf w to c:\n{RT} \n tf c to w:\n{RT_ctow} ")

            mesh_dir = "/scratch/saishubodh/RIO10_data/scene01/models01/seq01_01/"
            mesh_obj_file = os.path.join(mesh_dir ,"mesh.obj")
            print("Testing IO for meshes ...")
            mesh = load_objs_as_meshes([mesh_obj_file], device=device)
            print("Mesh loading done !!!")

            texture_image=mesh.textures.maps_padded()
            # plt.figure(figsize=(7,7))
            # plt.imshow(texture_image.squeeze().cpu().numpy())
            # plt.show()


            RR, tt, KK, img_size_t = RtK_in_torch_format(K, RT, img_size)

            cameras_pytorch3d = cameras_from_opencv_projection(RR.float(), tt.float(), KK.float(), img_size_t.float())
            # above line was giving dtype errors, so made everything float..
            cameras_pytorch3d  = cameras_pytorch3d.to(device)
            # (img_size_t.float().dtype, RR.dtype)
            raster_settings = RasterizationSettings(
                image_size=img_size, 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )

            #lights = lights_given_position(RT_ctow[0:3, 3], device)
            lights = lights_given_position(RT_wtoc[0:3, 3], device)

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras_pytorch3d, 
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=device,
                    cameras=cameras_pytorch3d,
                    lights=lights
                )
            )

            rendered_images = renderer(mesh)
            # plt.imshow(rendered_images[0, ..., :3].cpu().numpy())
            # plt.show()
            
            given_img = plt.imread(rgb_file)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(given_img)
            ax2.imshow(rendered_images[0, ..., :3].cpu().numpy())

            #img_type = "_true-pose-ctow"
            img_type = "_random-pose-wtoc"
            if save_imgs:
                plt.savefig("outputs/" + seq_id + "_" + str(num) + img_type + ".png")
                print(f"img saved to outputs/{seq_id + str(num)+ img_type}.png")
            plt.show()

def main_depth(img_ids, save_imgs):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    #change file num: 
    #added format code, so just
    #integer format

    # seq01_XX = ["01", "02"]
    seq01_XX = ["01"]
    for seq_id in seq01_XX:
        for num in img_ids:

            pose_file_dir = "/scratch/saishubodh/RIO10_data/scene01/seq01/seq01_" + seq_id +"/"
            camera_file = os.path.join(pose_file_dir, 'camera.yaml')

            camera_file = 'camera.yaml'
            pose_file = 'frame-{:06d}.pose.txt'.format(num)
            rgb_file = 'frame-{:06d}.color.jpg'.format(num)

            camera_file = os.path.join(pose_file_dir, camera_file)
            pose_file = os.path.join(pose_file_dir, pose_file)
            rgb_file = os.path.join(pose_file_dir, rgb_file)

            print(camera_file, os.path.isfile(camera_file))
            print(pose_file, os.path.isfile(pose_file))
            print(rgb_file, os.path.isfile(rgb_file))
            img = plt.imread(rgb_file)
            # plt.imshow(img)
            # plt.show()

            o3d_param, K, img_size =  parse_camera_file_RIO(camera_file)
            o3d_param, RT, RT_ctow = parse_pose_file_RIO(pose_file, o3d_param)
            RT_wtoc = RT

            print(f"H & W: {img_size}, \n K:\n{K}, \n tf w to c:\n{RT} \n tf c to w:\n{RT_ctow} ")

            mesh_dir = "/scratch/saishubodh/RIO10_data/scene01/models01/seq01_01/"
            mesh_obj_file = os.path.join(mesh_dir ,"mesh.obj")
            print("Testing IO for meshes ...")
            mesh = load_objs_as_meshes([mesh_obj_file], device=device)
            print("Mesh loading done !!!")

            texture_image=mesh.textures.maps_padded()
            # plt.figure(figsize=(7,7))
            # plt.imshow(texture_image.squeeze().cpu().numpy())
            # plt.show()


            RR, tt, KK, img_size_t = RtK_in_torch_format(K, RT, img_size)

            cameras_pytorch3d = cameras_from_opencv_projection(RR.float(), tt.float(), KK.float(), img_size_t.float())
            # above line was giving dtype errors, so made everything float..
            cameras_pytorch3d  = cameras_pytorch3d.to(device)
            # (img_size_t.float().dtype, RR.dtype)
            raster_settings = RasterizationSettings(
                image_size=img_size, 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )

            #lights = lights_given_position(RT_ctow[0:3, 3], device)
            lights = lights_given_position(RT_wtoc[0:3, 3], device)
            rasterizer=MeshRasterizer(
                    cameras=cameras_pytorch3d, 
                    raster_settings=raster_settings
            )
            renderer = MeshRenderer(
                rasterizer=rasterizer,
                shader=SoftPhongShader(
                    device=device,
                    cameras=cameras_pytorch3d,
                    lights=lights
                )
            )
            print("Test")


            rendered_images = renderer(mesh)


            #DEPTH EXPERIMENT
            fragments = rasterizer(mesh)
            depth_info = fragments.zbuf
            """
            fragments.zbuf is a (N, H, W, K) dimensional tensor
            #top k points. We'll take top 1
            source: https://github.com/facebookresearch/pytorch3d/issues/35#issuecomment-583870676
            """
            depth_image = depth_info[0,...,0]




            # plt.imshow(rendered_images[0, ..., :3].cpu().numpy())
            # plt.show()
            
            given_img = plt.imread(rgb_file)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(given_img)
            ax2.imshow(rendered_images[0, ..., :3].cpu().numpy())
            ax3.imshow(depth_info[0,...,0].cpu().numpy())


            #img_type = "_true-pose-ctow"
            img_type = "depth_stuff"
            if save_imgs:
                plt.savefig("outputs/" + seq_id + "_" + str(num) + img_type + ".png")
                print(f"img saved to outputs/{seq_id + str(num)+ img_type}.png")
            #plt.show()



def main_given_poses(given_poses, save_imgs):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    #change file num: 
    #added format code, so just
    #integer format

    # seq01_XX = ["01", "02"]
    seq01_XX = ["01"]
    for seq_id in seq01_XX:
        for pose_tf in given_poses: #for num in img_ids

            pose_file_dir = "/scratch/saishubodh/RIO10_data/scene01/seq01/seq01_" + seq_id +"/"
            camera_file = os.path.join(pose_file_dir, 'camera.yaml')

            camera_file = 'camera.yaml'
            # pose_file = 'frame-{:06d}.pose.txt'.format(num)
            # rgb_file = 'frame-{:06d}.color.jpg'.format(num)

            camera_file = os.path.join(pose_file_dir, camera_file)
            # pose_file = os.path.join(pose_file_dir, pose_file)
            # rgb_file = os.path.join(pose_file_dir, rgb_file)

            print(camera_file, os.path.isfile(camera_file))
            # print(pose_file, os.path.isfile(pose_file))
            # print(rgb_file, os.path.isfile(rgb_file))
            # img = plt.imread(rgb_file)
            # plt.imshow(img)
            # plt.show()

            o3d_param, K, img_size =  parse_camera_file_RIO(camera_file)
            # o3d_param, RT, RT_ctow = parse_pose_file_RIO(pose_file, o3d_param)
            RT = pose_tf
            RT_ctow = np.zeros((4,4))
            RT_wtoc = RT

            print(f"H & W: {img_size}, \n K:\n{K}, \n tf w to c:\n{RT} \n tf c to w:\n{RT_ctow} ")

            mesh_dir = "/scratch/saishubodh/RIO10_data/scene01/models01/seq01_01/"
            mesh_obj_file = os.path.join(mesh_dir ,"mesh.obj")
            print("Testing IO for meshes ...")
            mesh = load_objs_as_meshes([mesh_obj_file], device=device)
            print("Mesh loading done !!!")

            texture_image=mesh.textures.maps_padded()
            # plt.figure(figsize=(7,7))
            # plt.imshow(texture_image.squeeze().cpu().numpy())
            # plt.show()


            RR, tt, KK, img_size_t = RtK_in_torch_format(K, RT, img_size)

            cameras_pytorch3d = cameras_from_opencv_projection(RR.float(), tt.float(), KK.float(), img_size_t.float())
            # above line was giving dtype errors, so made everything float..
            cameras_pytorch3d  = cameras_pytorch3d.to(device)
            # (img_size_t.float().dtype, RR.dtype)
            raster_settings = RasterizationSettings(
                image_size=img_size, 
                blur_radius=0.0, 
                faces_per_pixel=1, 
            )

            #lights = lights_given_position(RT_ctow[0:3, 3], device)
            lights = lights_given_position(RT_wtoc[0:3, 3], device)

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras_pytorch3d, 
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=device,
                    cameras=cameras_pytorch3d,
                    lights=lights
                )
            )

            rendered_images = renderer(mesh)
            plt.imshow(rendered_images[0, ..., :3].cpu().numpy())
            plt.show()
            
            # given_img = plt.imread(rgb_file)
            # fig, (ax1, ax2) = plt.subplots(1, 2)
            # ax1.imshow(given_img)
            # ax2.imshow(rendered_images[0, ..., :3].cpu().numpy())

            #img_type = "_true-pose-ctow"
            img_type = "_random-pose-wtoc"
            if save_imgs:
                plt.savefig("outputs/" + seq_id + "_" + str(num) + img_type + ".png")
                print(f"img saved to outputs/{seq_id + str(num)+ img_type}.png")
            plt.show()


if __name__=='__main__':
    # img_ids = [131, 1992, 3530, 3622]
    img_ids = [131]
    save_imgs = True
    given_poses = poses_for_places()
    # print(given_poses)
    main_depth(img_ids, save_imgs)
    #main_given_poses(given_poses, save_imgs)
