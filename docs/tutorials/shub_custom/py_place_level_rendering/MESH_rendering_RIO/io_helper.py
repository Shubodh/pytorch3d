import cv2
import numpy as np
import logging
from pathlib import Path
import sys
import png
import open3d as o3d

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

def save_depth_image(np_array, save_path):
    """
    Given numpy array of depth in millimetres, save the png image at save_path.
    save_path must end with "_depth.png" or "rendered.depth.png"
    """
    # If np_array in metres, do np_array*1000
    depth_mm = (np_array).astype(np.uint16)

    # with open(raw_data_folder + str(obs_id) + "_depth.png", 'wb') as fdep:
    with open(save_path, 'wb') as fdep:
        writer = png.Writer(width=depth_mm.shape[1], height=depth_mm.shape[0], bitdepth=16, greyscale=True)
        depth_gray2list = depth_mm.tolist()
        writer.write(fdep, depth_gray2list)
    print(f"Saved depth image at {save_path}")