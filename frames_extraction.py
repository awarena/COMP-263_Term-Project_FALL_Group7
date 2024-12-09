import os
import cv2
import shutil
import time
import json


def vidToFrame(vid_file, out_dir):
    video_capture = cv2.VideoCapture(vid_file)
    output_folder = out_dir
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_count += 1
        frame_path = os.path.join(output_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

    video_capture.release()
    cv2.destroyAllWindows()



file_path = './data/input/WLASL_v0.3.json'
missing_file_path = './data/input/missing.txt'
videos_dir = './data/input/videos/'

# Load the WLASL dataset
with open(file_path) as file:
    wlasl = json.load(file)

# Read the missing video IDs from the file
with open(missing_file_path, 'r') as file:
    missing_videos = file.read().splitlines()

# Specify the base directory for the dataset
dataset_dir = './data/working/Dataset/frames'

# Create necessary directories
os.makedirs(dataset_dir, exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'Train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'Test'), exist_ok=True)

# Process each class in the WLASL dataset
for class_data in wlasl:
    class_name = class_data['gloss']
    print(class_name)
    
    for instance in class_data['instances']:
        video_id = instance['video_id']
        
        if video_id not in missing_videos:
            video_file = os.path.join(videos_dir, video_id + '.mp4')
    
            if instance['split'] == 'train':
                train_dir = os.path.join(dataset_dir, 'Train', class_name, video_id)
                os.makedirs(train_dir, exist_ok=True)
                vidToFrame(video_file, train_dir)
                print('train', video_id)
            else:
                test_dir = os.path.join(dataset_dir, 'Test', class_name, video_id)
                os.makedirs(test_dir, exist_ok=True)
                vidToFrame(video_file, test_dir)
                print('test', video_id)