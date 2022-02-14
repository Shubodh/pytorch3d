import os 
import cv2 

def save_vid(camera_dir, sequence_num, fps=10, interval=5):
    """
    Creates a video in the directory where this file is run from
    args:
        camera_dir -> directory where sequences are recorded
        sequence_num -> directory for sequence
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
    img_ = cv2.imread(images_files[0])
    H, W, C = img_.shape
    print(H,W,C)
    video=cv2.VideoWriter('video.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps,(W, H))
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

    save_vid(camera_dir, sequence_num)