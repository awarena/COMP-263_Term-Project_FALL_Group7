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
    nose = pose[:3]
    lh_wrist = lh[:3]
    rh_wrist = rh[:3]
    pose_adjusted = adjust_landmarks(pose, nose)
    lh_adjusted = adjust_landmarks(lh, lh_wrist)
    rh_adjusted = adjust_landmarks(rh, rh_wrist)
    return pose_adjusted, lh_adjusted, rh_adjusted

def make_keypoint_arrays(path, split):
    """
    Creates and saves keypoint arrays for each split (Train, Test, Val).
    Only processes words that are present in all three splits.
    """
    os.makedirs('./data/working/Dataset/npy_arrays', exist_ok=True)
    os.makedirs(f'./data/working/Dataset/npy_arrays/{split}', exist_ok=True)
    working_path = f'./data/working/Dataset/npy_arrays/{split}'
    words_folder = os.path.join(path, split)

    # Filter words that are present in all three folders
    train_words = set(os.listdir('./data/working/Dataset/frames/Train'))
    test_words = set(os.listdir('./data/working/Dataset/frames/Test'))
    val_words = set(os.listdir('./data/working/Dataset/frames/Val'))
    common_words = train_words & test_words & val_words

    print(f"Processing words for split '{split}': {len(common_words)} common words found.")

    # Loop through the filtered words
    for word in tqdm(common_words):
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
                        if frame is None:
                            continue

                        # Perform Mediapipe detection
                        image, results = mediapipe_detection(frame, holistic)

                        # Extract keypoints
                        pose, lh, rh = extract_keypoints(results)
                        pose_keypoints.append(pose)
                        lh_keypoints.append(lh)
                        rh_keypoints.append(rh)

                    # Save the keypoints for this video to numpy arrays
                    pose_directory = os.path.join(working_path, word, 'pose_keypoints')
                    lh_directory = os.path.join(working_path, word, 'lh_keypoints')
                    rh_directory = os.path.join(working_path, word, 'rh_keypoints')

                    os.makedirs(pose_directory, exist_ok=True)
                    os.makedirs(lh_directory, exist_ok=True)
                    os.makedirs(rh_directory, exist_ok=True)

                    pose_path = os.path.join(pose_directory, video_file)
                    np.save(pose_path, pose_keypoints)

                    lh_path = os.path.join(lh_directory, video_file)
                    np.save(lh_path, lh_keypoints)

                    rh_path = os.path.join(rh_directory, video_file)
                    np.save(rh_path, rh_keypoints)

# Generate keypoint arrays for Train, Test, and Val
make_keypoint_arrays('./data/working/Dataset/frames', 'Train')
make_keypoint_arrays('./data/working/Dataset/frames', 'Test')
make_keypoint_arrays('./data/working/Dataset/frames', 'Val')

# Verify common words
common_words = set(os.listdir('./data/working/Dataset/frames/Train')) & \
               set(os.listdir('./data/working/Dataset/frames/Test')) & \
               set(os.listdir('./data/working/Dataset/frames/Val'))

words = np.array(list(common_words))
print(words)

label_map = {label: num for num, label in enumerate(words)}
print(label_map)
