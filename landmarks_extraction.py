import mediapipe as mp
import random
import numpy as np
import pandas as pd
import os
import shutil
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import glob
import cv2
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
np.random.seed(42)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    if image is None:
        raise ValueError(f"Failed to load image: {image}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def adjust_landmarks(arr, center):
    arr_reshaped = arr.reshape(-1, 3)
    center_repeated = np.tile(center, (len(arr_reshaped), 1))
    arr_adjusted = arr_reshaped - center_repeated
    arr_adjusted = arr_adjusted.reshape(-1)
    return arr_adjusted

def extract_keypoints(results):

    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    nose=pose[:3]
    lh_wrist=lh[:3]
    rh_wrist=rh[:3]
    pose_adjusted = adjust_landmarks(pose,nose)
    lh_adjusted = adjust_landmarks(lh,lh_wrist)
    rh_adjusted = adjust_landmarks(rh,rh_wrist)
    return pose_adjusted, lh_adjusted, rh_adjusted


def move_train_to_test(train_path, test_path, word, percentage=0.3):
    word_train_path = os.path.join(train_path, word)
    word_test_path = os.path.join(test_path, word)
    
    if not os.path.exists(word_test_path):
        os.makedirs(word_test_path)
    
    video_files = os.listdir(word_train_path)
    num_files_to_move = int(len(video_files) * percentage)
    files_to_move = random.sample(video_files, num_files_to_move)
    
    for file in files_to_move:
        src = os.path.join(word_train_path, file)
        dst = os.path.join(word_test_path, file)
        shutil.move(src, dst)

selected_words = os.listdir('./data/working/Dataset/frames/Train')

def make_keypoint_arrays(path, split):
    os.makedirs('./data/working/Dataset/npy_arrays', exist_ok=True)
    os.makedirs(f'./data/working/Dataset/npy_arrays/{split}', exist_ok=True)
    working_path = f'./data/working/Dataset/npy_arrays/{split}'
    words_folder = os.path.join(path, split)
    selected_words1 = []
    
    for words1 in selected_words:
        npy_fold = os.listdir(os.path.join(working_path))
        if words1 not in npy_fold:
            selected_words1.append(words1)
    
    train_path = './data/working/Dataset/frames/Train'
    test_path = './data/working/Dataset/frames/Test'
    
    # Check if words in train set exist in test set, if not move 30% to test set
    for word in selected_words:
        if word not in os.listdir(test_path):
            move_train_to_test(train_path, test_path, word, percentage=0.3)
    
    # Loop through all the subfolders in the folder
    for word in tqdm(selected_words1):
        npy_fold = os.listdir(os.path.join(working_path))
        if word not in npy_fold:
            video_files = os.listdir(os.path.join(words_folder, word))
            # Loop through the video files
            for video_file in video_files:
                # Open the video file
                video = sorted(os.listdir(os.path.join(words_folder, word, video_file)))
        
                # Initialize the list of keypoints for this video
                pose_keypoints, lh_keypoints, rh_keypoints = [], [], []
                with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    # Loop through the video frames
                    for frame in video:
                        # Perform any necessary preprocessing on the frame (e.g., resizing, normalization)
                        frame = os.path.join(words_folder, word, video_file, frame)
                        frame = cv2.imread(frame)
                        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                        # Normalize pixel values to the range [0, 1]
                        # Make detections
                        image, results = mediapipe_detection(frame, holistic)
        
                        # Extract keypoints
                        pose, lh, rh = extract_keypoints(results)
                        # Add the keypoints to the list for this video
                        pose_keypoints.append(pose)
                        lh_keypoints.append(lh)
                        rh_keypoints.append(rh)
                    
                    # Save the keypoints for this video to a numpy array
                    pose_directory = os.path.join(working_path, word, 'pose_keypoints')
                    lh_directory = os.path.join(working_path, word, 'lh_keypoints')
                    rh_directory = os.path.join(working_path, word, 'rh_keypoints')
        
                    if not os.path.exists(pose_directory):
                        os.makedirs(pose_directory)
        
                    if not os.path.exists(lh_directory):
                        os.makedirs(lh_directory)
        
                    if not os.path.exists(rh_directory):
                        os.makedirs(rh_directory)
        
                    pose_path = os.path.join(pose_directory, video_file)
                    np.save(pose_path, pose_keypoints)
        
                    lh_path = os.path.join(lh_directory, video_file)
                    np.save(lh_path, lh_keypoints)
        
                    rh_path = os.path.join(rh_directory, video_file)
                    np.save(rh_path, rh_keypoints)

make_keypoint_arrays('./data/working/Dataset/frames','Train/')
make_keypoint_arrays('./data/working/Dataset/frames','Test/')

words= np.array(os.listdir('./data/working/Dataset/frames/Train'))
print(words)

label_map = {label:num for num, label in enumerate(words)}
print(label_map)