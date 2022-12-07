import cv2
import numpy as np
import logging
from pathlib import Path
import sys
import png
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from parsers import parse_pose_file_RIO #parse_poses_from_file, 

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise ValueError(f'Cannot read image {path}.')
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image

def read_depth_image_given_depth_path(path):
    depth_file = Path(path)
    assert  depth_file.exists(), depth_file 
    
    depth_image = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
    if depth_image is None:
        raise ValueError(f'Cannot read image {path}.')

    return depth_image

def read_depth_image_given_colorimg_path(dataset_dir, r):
    """
    This function replaces color img path with depth img for same img id.
    dataset_dir: Path(datasets/InLoc_like_RIO10/scene01_synth)
    r: Path(database/cutouts/frame-001820.color.jpg)
    """
    full_prefix_path = dataset_dir / r.parents[0]
    r_stem = r.stem.replace("color", "")
    depth_file  = Path(full_prefix_path, r_stem + 'rendered.depth.png')
    assert  depth_file.exists(), depth_file 
    
    depth_image = cv2.imread(str(depth_file), cv2.IMREAD_ANYDEPTH)
    if depth_image is None:
        raise ValueError(f'Cannot read image {path}.')

    # Using Open3D
    #print("DEBUG DePTH") 
    #depth_raw = o3d.io.read_image(str(depth_file))
    #depth_img = np.asarray(depth_raw)
    #print(np.array_equal(depth_img, depth_image))

    return depth_image

def save_depth_image(np_array, save_path, depth_in_metres=False):

    """
    CONVENTION: We are representing holes in depth images with 0 as per RIO10 format (and you seem to be getting good metric results with this as well). 
    So before passing depth_array to this function, ensure you replace [-1 in pytorch3d case] with 0. 

    Given numpy array of depth in millimetres, save the png image at save_path.
    save_path must end with "_depth.png" or "rendered.depth.png"
    """

    # If np_array in metres, do np_array*1000
    if depth_in_metres:
        np_array = np_array * 1000
    depth_mm = (np_array).astype(np.uint16)

    # with open(raw_data_folder + str(obs_id) + "_depth.png", 'wb') as fdep:
    with open(save_path, 'wb') as fdep:
        writer = png.Writer(width=depth_mm.shape[1], height=depth_mm.shape[0], bitdepth=16, greyscale=True)
        depth_gray2list = depth_mm.tolist()
        writer.write(fdep, depth_gray2list)
    print(f"Saved depth image at {save_path}")

def write_individual_pose_txt_in_RIO_format(RT_ctow, dest_file_prefix):
    # print(RT_ctow)
    rows_list = [RT_ctow[0], RT_ctow[1], RT_ctow[2], RT_ctow[3]]
    # print(rows_list)

    pose_save_path = Path(str(dest_file_prefix) + ".pose.txt")
    with open(pose_save_path, 'w') as f:
        for row in rows_list:
            row = ' '.join(map(str, row))
            # name = q.split("/")[-1]
            f.write(f'{row}\n')
    print(f'Written individual pose to {pose_save_path}')


def return_individual_pose_files_as_single_list(folder_path, scene_id):
    """ 
    Given a folder as input, 
    1. read all individual pose.txt files (RIO10 format, ctow: 4*4 matrix in 4 lines) 
    2. Optional: Convert Rt to quat. Default: RT_ctow only, not quat.
    3. return it as a list in ctow
    """
    is_quat = False
    folder_path = Path(folder_path)
    pose_files = sorted(list(folder_path.glob('*pose.txt')))
    img_poses_list_final = []
    dict_final = {}
    for pose_file in pose_files:
        _, _, RT_ctow = parse_pose_file_RIO(pose_file)
        if is_quat:
            qx_c, qy_c, qz_c, qw_c = R.from_matrix(RT_ctow[0:3,0:3]).as_quat()
            tx_c, ty_c, tz_c = RT_ctow[0:3,3]
            pose_c = [qw_c, qx_c, qy_c, qz_c, tx_c, ty_c, tz_c]

        img_poses_list_final.append(RT_ctow)
        pose_file_pre = Path(pose_file.stem).stem
        dict_final[pose_file_pre] = RT_ctow
        
        # print(pose_file)
    return img_poses_list_final, dict_final

def write_individual_pose_files_to_single_output(folder_path, output_file_path, scene_id):
    """ 
    Given a folder as input, 
    1. read all individual pose.txt files (RIO10 format, ctow: 4*4 matrix in 4 lines) 
    2. Convert Rt to quat
    3. save it in a single output file (RIO10 format, ctow: # scene-id/frame-id qw qx qy qz tx ty tz)
    """
    folder_path = Path(folder_path)
    pose_files = sorted(list(folder_path.glob('*pose.txt')))
    img_poses_list_final = []
    for pose_file in pose_files:
        _, RT_ctow = parse_pose_file_RIO(pose_file)
        qx_c, qy_c, qz_c, qw_c = R.from_matrix(RT_ctow[0:3,0:3]).as_quat()
        tx_c, ty_c, tz_c = RT_ctow[0:3,3]
        pose_c = [qw_c, qx_c, qy_c, qz_c, tx_c, ty_c, tz_c]

        file_stem = str(Path(Path(pose_file).stem).stem)
        file_full = "seq" + str(scene_id) + "_02/" + file_stem
        img_poses_list_final.append((file_full, pose_c))
        
        # print(pose_file)
    
    write_results_to_file(output_file_path, img_poses_list_final)
    print(f"Converted individual pose files in {str(folder_path)} to single output file {str(output_file_path)}")
