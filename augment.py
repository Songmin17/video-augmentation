import cv2
import numpy as np 
import os
from PIL import Image, ImageSequence 
from vidaug.vidaug import augmentors as va

# clear all contents within a directory
def clear_dir(dir_path):
    for item in os.listdir(dir_path):
        item_path = f'{os.path.abspath(dir_path)}/{item}'
        os.remove(item_path)

def main():
    movie_shots_path = '/data1/common_datasets_urop/MovieShots'
    data_dir = os.path.join(movie_shots_path, 'frame', 'raw')
    result_dir = os.path.join(movie_shots_path, 'frame', 'augmented')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    sometimes = lambda aug: va.Sometimes(0.5, aug)
    sigmas = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    np.random.seed(1)

    seq = va.Sequential([
        sometimes(va.HorizontalFlip()),
        sometimes(va.VerticalFlip()),
        sometimes(va.OneOf([va.GaussianBlur(np.random.choice(sigmas)), va.ElasticTransformation()])),
        sometimes(va.Add(20)),
        va.OneOf([va.Pepper(), va.Salt(), va.Add(10)]),
        sometimes(va.OneOf([va.TemporalBeginCrop(size=13), va.InverseOrder()]))
    ])

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(os.path.abspath(data_dir), folder)
        for shot_folder in os.listdir(folder_path):
            img_seq = []
            shot_folder_path = os.path.join(folder_path, shot_folder)
            for item in os.listdir(shot_folder_path):
                item_path = os.path.join(shot_folder_path, item)
                if '.jpg' in item:
                    img = cv2.imread(item_path)
                    img_seq.append(img)

            augmented_img_seq_path = os.path.join(result_dir, folder, shot_folder)
            if not os.path.exists(os.path.join(os.path.abspath(result_dir), folder)):
                os.makedirs(os.path.join(result_dir, folder))
            
            if not os.path.exists(os.path.join(os.path.abspath(result_dir), folder, shot_folder)):
                os.makedirs(os.path.join(result_dir, folder, shot_folder))

            video_aug = seq(img_seq)
            for i, aug_img in enumerate(video_aug):
                cv2.imwrite(f'{augmented_img_seq_path}/image_{i:05d}.jpg', aug_img)

# import shutil

# movie_shots_path = '/data1/common_datasets_urop/MovieShots'
# result_dir = os.path.join(movie_shots_path, 'frame', 'augmented')
# for folder in os.listdir(result_dir):
#     folder_path = os.path.join(os.path.abspath(result_dir), folder)
#     shot_folders = list(os.listdir(folder_path))
#     print(f'shot_folders:\n{shot_folders}')

#     for elem in shot_folders[1:]:
#         current_path = os.path.join(folder_path, elem)
#         shutil.rmtree(current_path) 

main()

    
