import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def image_too_close(filename, threshold, depth_value):
    """
    read this function as:
    if <threshold> % of image has at max <depth_value>
    then it is too close
    Ex: image_too_close("test.png", 70, 1000)
    returns True if test.png has 70% of pixels less than 1000
    -> lower depth value -> means object is closer to camera
    """
    img = cv2.imread(filename,  cv2.IMREAD_ANYDEPTH)
    (H,W) = img.shape 
    total = H * W 

    img[img <= depth_value] = 1
    img[img > depth_value] = 0

    total_img = np.sum(img)
    

    img_ratio = (total_img/total)*100
    # print(img_ratio)
    if img_ratio > threshold:
        return True
    return False


def depth_analysis(camera_dir, sequence_num):
    """
    prints number of files based on params
    args:
        camera_dir -> directory where sequences are recorded
        sequence_num -> directory for sequence
    """
    seq_dir = os.path.join(camera_dir, sequence_num)

    list_files = os.listdir(seq_dir)
    total_files = len(list_files)

    total_frames = total_files - 1 #1 file is camera file
    total_frames = total_frames // 3
    # print(total_frames)

    depth_files = ["frame-{:06d}.rendered.depth.png".format(i) for i in range(total_frames)]
    depth_files = [os.path.join(seq_dir, i) for i in depth_files]

    
    
    #these values to be played with
    percent = 70 
    threshold = 1500 # 1000 #
    
    count = 0
    # print(depth_files)
    for i in depth_files:
        # print(i, os.path.isfile(i))
        if image_too_close(i, percent, threshold):
            count += 1
            
    print(f"{str(sequence_num)[:-1]}:  {(count/total_frames)*100}%, i.e. {count} images out of {total_frames} images are too close: too close meaning --> {percent}% of pixels less than {threshold} mm")
    """
    uncomment below to plot random images
    """
    """
    test_idx = 250
    img = cv2.imread(depth_files[test_idx],  cv2.IMREAD_ANYDEPTH)
    img_rgb = cv2.imread(os.path.join(seq_dir,"frame-{:06d}.color.jpg".format(test_idx)))

    print(img.shape)
    plt.imshow(img)
    plt.show()
    plt.imshow(img_rgb)
    plt.show()
    """
    
def depth_analysis_for_places(folder_path):
    """
    prints number of files based on params
    args:
        folder_path -> directory where sequences are recorded
    """
    seq_dir = Path(folder_path)
    depth_files = sorted(list(seq_dir.glob('places*depth.png')))
    # rgb_files = sorted(list(seq_dir.glob('*depth.png')))
    total_frames = len(depth_files)

    # depth_files = ["frame-{:06d}.rendered.depth.png".format(i) for i in range(total_frames)]
    # depth_files = [os.path.join(seq_dir, i) for i in depth_files]

    #these values to be played with
    percent = 70 
    threshold = 1500 # 1000 #
    
    count = 0
    # print(depth_files)
    for i in depth_files:
        # print(i, os.path.isfile(i))
        if image_too_close(str(i), percent, threshold):
            count += 1
            
    print(f"{str(sequence_num)[:-1]}_rendered:  {(count/total_frames)*100}%, i.e. {count} images out of {total_frames} images are too close: too close meaning --> {percent}% of pixels less than {threshold} mm")
    """
    uncomment below to plot random images
    """
    """
    test_idx = 250
    img = cv2.imread(depth_files[test_idx],  cv2.IMREAD_ANYDEPTH)
    img_rgb = cv2.imread(os.path.join(seq_dir,"frame-{:06d}.color.jpg".format(test_idx)))

    print(img.shape)
    plt.imshow(img)
    plt.show()
    plt.imshow(img_rgb)
    plt.show()
    """

    

if __name__ == '__main__':
    """
    Gives output based on depth analysis as following:
    seq01_01: 74.1324200913242%, i.e. 3247 images out of 4380 images are too close: too close meaning 
    --> 70% of pixels less than 1500 mm
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, required=True) # 01 or 02 etc
    parser.add_argument('--on_ada', type=str, required=True,choices=['ada', 'shub_local', 'aryan_local'], help ='where running code') 
    parser.add_argument('--ref_or_query', type=str, required=True,choices=['ref', 'query'], help ='save ref vids or query') 
    parser.add_argument('--places_analysis', dest='places_analysis', default=False, action='store_true') # Just provide (next line)
    # provide "--places_analysis" on command line if you want to do depth analysis for places. Don't set it to anything for original.
    args = parser.parse_args()


    scene_id = args.scene_id
    on_ada = args.on_ada
    ref_or_query = args.ref_or_query
    places_analysis_bool = args.places_analysis

    custom_dir = (on_ada=="aryan_local") 
    ada = (on_ada=="ada")

    #Reading data paths
    mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
    camera_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"
   
    #ada = not viz_pcd
    if ada:
        mesh_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
        camera_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"

    if custom_dir:
        mesh_dir = "../../../../../scene" + scene_id + "/models" + scene_id + "/"
        camera_dir = "../../../../../scene" + scene_id + "/seq" + scene_id + "/"

    #Sequence number and where visualisations are saved:
    # MAKE SURE IT IS SAVE WITHOUT '/'
    ref_bool = (ref_or_query=="ref")
    if ref_bool:
        sequence_num = 'seq' + scene_id + '_01/'
        folder_path = "/scratch/saishubodh/InLoc_like_RIO10/sampling10/scene" +  scene_id + "_A-queryAND-ND_PLACES/database/cutouts/"
    else:
        sequence_num = 'seq' + scene_id + '_02/'
        folder_path = "/scratch/saishubodh/InLoc_like_RIO10/sampling10/scene" +  scene_id + "_A-queryAND-ND_PLACES/query/"

    if places_analysis_bool:
        print("depth analysis for places rendered images")
        depth_analysis_for_places(folder_path)
    else:
        print("depth analysis for normal images")
        depth_analysis(camera_dir, sequence_num)
    # print("What about depth analysis for places, aka, rendered images?")