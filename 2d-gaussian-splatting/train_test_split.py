import os
import shutil
import json
import numpy as np

def copy_images(train_imgs, test_imgs, src_dir, dst_dir):
    # Create destination directories:
    # dst_dir
    # ├── train
    # └── test
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.exists(dst_dir + '/train'):
        os.makedirs(dst_dir + '/train')
    if not os.path.exists(dst_dir + '/test'):
        os.makedirs(dst_dir + '/test')

    # Copy train images
    for img_num in train_imgs:
        src_path = src_dir + f'/train/r_{img_num}.png'
        dst_path = dst_dir + f'/train/r_{img_num}.png'
        shutil.copy2(src_path, dst_path)
        print(f'Copied {src_path} to {dst_path}')

    # Copy test images
    for img_num in test_imgs:
        src_path = src_dir + f'/test/r_{img_num}.png'
        dst_path = dst_dir + f'/test/r_{img_num}.png'
        shutil.copy2(src_path, dst_path)
        print(f'Copied {src_path} to {dst_path}')


def filter_frames(frames, image_numbers):
    filtered_frames = []
    for frame in frames:
        # Extract the image number from the file path
        image_number = int(frame["file_path"].split('_')[-1])
        if image_number in image_numbers:
            filtered_frames.append(frame)
    return filtered_frames


def create_filtered_json(train_imgs, test_imgs, src_dir, dst_dir):
    # Create transofrms_train.json
    with open(os.path.join(src_dir, 'transforms_train.json'), 'r') as file:
        data = json.load(file)

    train_frames = filter_frames(data["frames"], train_imgs)
    
    train_json_data = {
        "camera_angle_x": data["camera_angle_x"],
        "frames": train_frames
    }
    
    with open(os.path.join(dst_dir, 'transforms_train.json'), 'w') as file:
        json.dump(train_json_data, file, indent=4)

    # Create transofrms_test.json
    with open(os.path.join(src_dir, 'transforms_test.json'), 'r') as file:
        data = json.load(file)
    
    test_frames = filter_frames(data["frames"], test_imgs)

    test_json_data = {
        "camera_angle_x": data["camera_angle_x"],
        "frames": test_frames
    }

    with open(os.path.join(dst_dir, 'transforms_test.json'), 'w') as file:
        json.dump(test_json_data, file, indent=4)


if __name__ == '__main__':
    for dataset in ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']:
        # SPECIFY THE TRAIN/TEST IMAGES
        train_imgs = [2, 7, 13, 16, 26, 30, 53, 54, 55, 73, 75, 78, 86, 92, 93, 95]
        test_imgs = [i for i in range(0, 200, 8)]

        # SPECIFY THE SOURCE AND DESTINATION DIRECTORIES FOR THE TRAIN/TEST IMAGES
        src_dir = f'data/nerf_synthetic/{dataset}'
        dst_dir = f'data/free_nerf_runs/{dataset}_16v'


        copy_images(train_imgs, test_imgs, src_dir, dst_dir)
        create_filtered_json(train_imgs, test_imgs, src_dir, dst_dir)

