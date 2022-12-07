import os


if __name__ == '__main__':
    #room_ids = ["1","3", "5", "7", "9"]
    room_ids = ["1", "3", "7", "9"]
    #room_ids = ["4"]
    for room_id in room_ids:
        folder_name = "/scratch/saishubodh/InLoc_like_RIO10/sampling10/scene0" + room_id + "_RRI_with_QRI/database/cutouts/"
        file_list = []
        for file_n in os.listdir(folder_name):
            #print(os.path.join(folder_name, file_n))
            file_list.append(os.path.join(folder_name, file_n))

        [os.rename(f, f.replace('frame-rendered', 'frame')) for f in file_list  if not f.startswith('.')]
        #[os.rename(f, f.replace('_', '-')) for f in os.listdir('.') if not f.startswith('.')]
