import os 
import cv2 
import argparse
from pathlib import Path
import sys
from tqdm import tqdm

def save_vid_from_downloads_folder_given_seq_no(camera_dir, sequence_num, fps=10, interval=5):
    """
    Creates a video in the directory where this file is run from
    args:
        camera_dir -> directory where sequences are recorded
        sequence_num -> ending dir path: 'seq' + scene_id + '_01/'
        fps -> fps of video you want
        interval -> how many frames to skip
    TODO: add a dest path option
    """
    seq_dir = os.path.join(camera_dir, sequence_num)
    list_files = os.listdir(seq_dir)
    total_files = len(list_files)

    total_frames = total_files - 1 #1 file is camera file
    total_frames = total_frames // 3
    print(total_frames)

    images_files = ["frame-{:06d}.color.jpg".format(i) for i in range(total_frames)]
    images_files = [os.path.join(seq_dir, i) for i in images_files]
    

    #Sanity check
    # print(images_files)
    # for i in images_files:
    #     print(i, os.path.isfile(i))
    vid_name = str(sequence_num)[:-1]
    img_ = cv2.imread(images_files[0])
    H, W, C = img_.shape
    # print(H,W,C)
    video=cv2.VideoWriter(vid_name +'_original' + '.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps,(W, H))
    count = 0
    print(f"writing video to {vid_name}_original.mp4")
    for image in tqdm(images_files):
        if count%interval == 0:
            img = cv2.imread(image)
            video.write(img)

        count += 1

    cv2.destroyAllWindows()
    video.release()

def save_vid_from_rgb_imgs_given_any_folder_path(folder_path, fps=10, interval=5):
    """
    Creates a video in the directory where this file is run from
    args:
        camera_dir -> directory where sequences are recorded
        sequence_num -> directory for sequence
        fps -> fps of video you want
        interval -> how many frames to skip
    TODO: add a dest path option
    """
    #seq_dir = os.path.join(camera_dir, sequence_num)
    seq_dir = camera_dir
    list_files = os.listdir(seq_dir)
    total_files = len(list_files)

    total_frames = total_files - 1 #1 file is camera file
    #total_frames = total_frames // 3
    print(total_frames)

    #images_files = ["frame-{:06d}.color.jpg".format(i) for i in range(total_frames)]
    #images_files = [os.path.join(seq_dir, i) for i in images_files]
    print(list_files)
    images_files = list_files
    

    #Sanity check
    # print(images_files)
    # for i in images_files:
    #     print(i, os.path.isfile(i))
    img_ = cv2.imread(images_files[0])
    H, W, C = img_.shape
    print(H,W,C)
    video=cv2.VideoWriter('rend_video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps,(W, H))
    count = 0
    for image in images_files:
        if count%interval == 0:
            img = cv2.imread(image)
            video.write(img)

        count += 1

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    """
    Reads jpg files from dir, and saves a video
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, required=True) # 01 or 02 etc
    parser.add_argument('--on_ada', type=str, required=True,choices=['ada', 'shub_local', 'aryan_local'], help ='where running code') 
    args = parser.parse_args()

    # final_poses = poses_for_places(viz_pcd, True)
    scene_id = args.scene_id
    on_ada = args.on_ada

    custom_dir = (on_ada=="aryan_local") 
    ada = (on_ada=="ada")
    # ada = False
    #viz_pcd = True

    mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
    camera_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"
   
    #ada = not viz_pcd
    if ada:
        #mesh_dir = "/scratch/saishubodh/RIO10_data/scene01/models01/"
        #camera_dir = "/scratch/saishubodh/RIO10_data/scene01/seq01/"
        mesh_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
        camera_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"

    if custom_dir:
        #mesh_dir = "../../../../../scene01/models01/"
        #camera_dir = "../../../../../scene01/seq01/"
        mesh_dir = "../../../../../scene" + scene_id + "/models" + scene_id + "/"
        camera_dir = "../../../../../scene" + scene_id + "/seq" + scene_id + "/"
    
    #Sequence number and where visualisations are saved:
    # MAKE SURE IT IS SAVE WITHOUT '/'
    sequence_num = 'seq' + scene_id + '_01/'

    # save_vid_from_downloads_folder_given_seq_no(camera_dir, sequence_num)

    folder_path = "/scratch/saishubodh/InLoc_like_RIO10/sampling10/scene01_AND_PLACES/database/cutouts/"
    save_vid_from_rgb_imgs_given_any_folder_path(folder_path)
