import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


def image_too_close(filename, threshold, depth_value):
    """
    read this function as:
    if <threshold> % of image has at max <depth_value>
    then it is too close
    Ex: image_too_close("test.png", 70, 1000)
    returns True if test.png has 70% of pixels less than 1000
    -> lower depth value -> means object is closer to screen
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
    print(total_frames)

    depth_files = ["frame-{:06d}.rendered.depth.png".format(i) for i in range(total_frames)]
    depth_files = [os.path.join(seq_dir, i) for i in depth_files]

    
    
    #these values to be played with
    percent = 70 
    threshold = 1000
    
    count = 0
    # print(depth_files)
    for i in depth_files:
        # print(i, os.path.isfile(i))
        if image_too_close(i, percent, threshold):
            count += 1
            
    print("{i} images out of {j} images are too close".format(i=count,j=total_frames))
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
    Reads jpg files from dir, and saves a video
    """
    custom_dir = True
    ada = False 

    #Reading data paths
    mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/models01/"
    camera_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene01/seq01/"
   
    
    if ada==True:
        mesh_dir = "/scratch/saishubodh/RIO10_data/scene01/models01/"
        camera_dir = "/scratch/saishubodh/RIO10_data/scene01/seq01/"

    if custom_dir:
        mesh_dir = "../../../../../scene01/models01/"
        camera_dir = "../../../../../scene01/seq01/"

    #Sequence number and where visualisations are saved:
    # MAKE SURE IT IS SAVE WITHOUT '/'
    sequence_num = 'seq01_01'
    depth_analysis(camera_dir, sequence_num)