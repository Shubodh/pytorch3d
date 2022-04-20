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

def save_vid_from_rgb_imgs_given_any_folder_path(folder_path, save_rendered_vid, scene_id, fps=10, interval=5):
    """
    Creates a video in the directory where this file is run from
    args:
        folder_path -> directory where rgb files are located
        save_rendered_vid -> True if only save places images, False if all rgb images of specified `folder_path`
        fps -> fps of video you want
        interval -> how many frames to skip
    """
    seq_dir = Path(folder_path)
    if save_rendered_vid: #Just places.
        rgb_files = sorted(list(seq_dir.glob('places*color.jpg')))
    else: # all rgb images
        rgb_files = sorted(list(seq_dir.glob('*color.jpg')))
    # list_files = os.listdir(rgb_files)
    total_frames = len(rgb_files)

    img_ = cv2.imread(str(rgb_files[0]))
    H, W, C = img_.shape
    # print(H,W,C)

    vid_name = str(sequence_num)[:-1]
    fps = 2
    if save_rendered_vid: #Just places.
        video=cv2.VideoWriter(vid_name +'_places_rendered' + '.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps,(W, H))
        print(f"writing video to {vid_name}_places_rendered.mp4")
    else: # all rgb images
        video=cv2.VideoWriter(vid_name  + '.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps,(W, H))
        print(f"writing video to {vid_name}.mp4")

    count = 0
    for image in tqdm(rgb_files):
        if count%interval == 0:
            img = cv2.imread(str(image))
            video.write(img)

        count += 1

    cv2.destroyAllWindows()
    video.release()

if __name__ == '__main__':
    """
    Reads jpg files from dir, and saves a video: Two functions are there -->
    save_vid_from_rgb_imgs_given_any_folder_path()
    save_vid_from_downloads_folder_given_seq_no()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_id', type=str, required=True) # 01 or 02 etc
    parser.add_argument('--on_ada', type=str, required=True,choices=['ada', 'shub_local', 'aryan_local'], help ='where running code') 
    parser.add_argument('--ref_or_query', type=str, required=True,choices=['ref', 'query'], help ='save ref vids or query') 
    parser.add_argument('--save_rendered_vid', dest='save_rendered_vid', default=False, required=True, action='store_true') # Just provide "--save_rendered_vid" on command line if you want to save rendered vid. Don't set it to anything if you want to save normal original vids.
    args = parser.parse_args()

    # final_poses = poses_for_places(viz_pcd, True)
    scene_id = args.scene_id
    on_ada = args.on_ada
    save_rendered_vid = args.save_rendered_vid
    ref_or_query = args.ref_or_query

    custom_dir = (on_ada=="aryan_local") 
    ada = (on_ada=="ada")
    # ada = False
    #viz_pcd = True

    # mesh_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
    camera_dir = "/media/shubodh/DATA/Downloads/data-non-onedrive/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"
   
    #ada = not viz_pcd
    if ada:
        # mesh_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/models" + scene_id + "/"
        camera_dir = "/scratch/saishubodh/RIO10_data/scene" + scene_id + "/seq" + scene_id + "/"

    if custom_dir:
        # mesh_dir = "../../../../../scene" + scene_id + "/models" + scene_id + "/"
        camera_dir = "../../../../../scene" + scene_id + "/seq" + scene_id + "/"
    
    #Sequence number and where visualisations are saved:
    # MAKE SURE IT IS SAVE WITHOUT '/'
    if ref_or_query == "ref":
        sequence_num = 'seq' + scene_id + '_01/'
        folder_path = "/scratch/saishubodh/InLoc_like_RIO10/sampling10/scene" +  scene_id + "_A-queryAND-ND_PLACES/database/cutouts/"
    elif ref_or_query == "query":
        sequence_num = 'seq' + scene_id + '_02/'
        folder_path = "/scratch/saishubodh/InLoc_like_RIO10/sampling10/scene" +  scene_id + "_A-queryAND-ND_PLACES/query/"


    if save_rendered_vid:
        save_vid_from_rgb_imgs_given_any_folder_path(folder_path, save_rendered_vid, scene_id)
    else:
        save_vid_from_downloads_folder_given_seq_no(camera_dir, sequence_num)


    # IF YOU WANT TO save vid given ANY folder (not just save places but all rgb imgs),
    # call save_vid_from_rgb_imgs_given_any_folder_path() with save_rendered_vid as False BELOW.