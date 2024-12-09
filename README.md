# COMP-263_Term-Project_FALL_Group7
## Installation Requiremnts
tensorflow
opencv-python
mediapipe
torchvision
torch
pillow

## Dataset
https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed/data

## Running Instructions
### Supervised
1. Download and extract the dataset, put it in a folder structure like this: data/input/{your_dataset}
2. Run 'frames_extraction_supervised.py'
3. Run 'landmarks_extraction_supervised.py'
4. Run 'lstm_model_processing_supervised.py'

### Self-supervised
1. have a videos folder that has all the videos run the below code to create the frames/ folder
```
import pandas as pd
import json
from pathlib import Path

# open WLASL_v0.3.json
df = pd.read_json("WLASL_v0.3.json")

# Read the missing video IDs from the file and store them in a set
with open("missing.txt", "r") as file:
    missing_video_ids = set(file.read().splitlines())

import av
import cv2
import shutil

def extract_frames(path_to_video, path_to_frames, frame_start, frame_end, fps):
    # extract frames
    "convert video to PIL images "
    
    video = av.open(str(path_to_video))
    for frame in video.decode(0):
        yield frame.to_image()
    pass

def avi2frames(video_path, path_frames, force=False):
    # check if video is already processed
    if not force and (path_frames / '00001.jpg').exists():
        return
    
    # copy the video to the frames directory and name it 00000.mp4
    # path_frames.mkdir(parents=True, exist_ok=True)
    # path_frames_video = path_frames / '00000.mp4'
    # shutil.copy(video_path, path_frames_video)
    
    # extract frames
    video = cv2.VideoCapture(str(video_path))
    frame_id = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame_id += 1
        cv2.imwrite(str(path_frames / f'{frame_id:05d}.jpg'), frame)
    video.release()

valid_video_ids = set()
folders_to_remove = set()

# apply function to each row
def process_row(gloss, instances):
    videos_for_training = [instance['video_id'] for instance in instances if instance['split'] == 'train']
    if all(videoId in missing_video_ids for videoId in videos_for_training):
        folders_to_remove.add(gloss)
        return
    for instance in instances:
        videoId = instance['video_id']
        bbox = instance['bbox']
        fps = instance['fps']
        frame_start = instance['frame_start']
        frame_end = instance['frame_end']
        split = instance['split']
        if split != 'train':
            valid_video_ids.add(videoId)
        
        path = Path(f'frames/{gloss}/{videoId}')
        path.mkdir(parents=True, exist_ok=True)
        
        path_to_video = Path(f'videos/{videoId}.mp4')
        
        # extract frames
        avi2frames(path_to_video, path, force=False)

        
# apply function to one row
glosses = df['gloss']
instances = df['instances']
for i in range(len(glosses)):
    process_row(glosses[i], instances[i])

import os

def remove_empty_folders(path):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
                if dir_path.split('/')[-1] in valid_video_ids:
                    valid_video_ids.remove(dir_path.split('/')[-1])

# Remove empty folders in the frames/ directory
remove_empty_folders('frames/')

# Remove folders that contain only missing videos
for folder in folders_to_remove:
    shutil.rmtree(f'frames/{folder}', ignore_errors=True)
```
2. Run the gesture_annotioation.py
3. Run the asl_training_self_supervised.py
4. Run the asl_inference_self.supervised.py
   
### SOTA
1. Download and extract the dataset, put it in a folder structure like this: data/input/{your_dataset}
2. Run 'frames_extraction_supervised.py'
3. Run 'sota.py'

