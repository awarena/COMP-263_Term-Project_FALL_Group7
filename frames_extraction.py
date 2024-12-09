import os
import cv2
import shutil
import json
import random

def vid_to_frames(vid_file, out_dir):
    """Extract frames from a video and save them to the specified directory."""
    video_capture = cv2.VideoCapture(vid_file)
    os.makedirs(out_dir, exist_ok=True)

    frame_count = 0
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame_count += 1
        frame_path = os.path.join(out_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

    video_capture.release()
    cv2.destroyAllWindows()


def process_wlasl_data(file_path, missing_file_path, videos_dir, dataset_dir, move_percentage=0.3):
    """Process WLASL dataset, handling missing videos and ensuring balanced splits."""
    # Load the WLASL dataset
    with open(file_path) as file:
        wlasl = json.load(file)

    # Read the missing video IDs from the file
    with open(missing_file_path, 'r') as file:
        listed_missing_videos = set(file.read().splitlines())

    # Identify missing videos by checking their existence
    all_missing_videos = set(listed_missing_videos)
    for class_data in wlasl:
        for instance in class_data["instances"]:
            video_id = instance["video_id"]
            video_file = os.path.join(videos_dir, video_id + '.mp4')
            if not os.path.exists(video_file):
                all_missing_videos.add(video_id)

    # Create necessary directories
    os.makedirs(dataset_dir, exist_ok=True)
    for split in ["Train", "Test", "Val"]:
        os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)

    # Process each class
    for class_data in wlasl:
        class_name = class_data["gloss"]
        train_instances = []
        test_instances = []
        val_instances = []

        for instance in class_data["instances"]:
            video_id = instance["video_id"]
            if video_id in all_missing_videos:
                continue  # Skip missing videos

            split = instance["split"].capitalize()  # Convert split to match folder naming
            video_file = os.path.join(videos_dir, video_id + '.mp4')
            output_dir = os.path.join(dataset_dir, split, class_name, video_id)

            vid_to_frames(video_file, output_dir)

            if split == "Train":
                train_instances.append(video_id)
            elif split == "Test":
                test_instances.append(video_id)
            elif split == "Val":
                val_instances.append(video_id)

        # Ensure all splits have at least one instance
        if not test_instances or not val_instances:
            num_to_move = max(int(len(train_instances) * move_percentage), 1)
            if len(train_instances) == 1:
                # Clone the single instance to Test and Val
                video_id = train_instances[0]
                train_path = os.path.join(dataset_dir, "Train", class_name, video_id)
                for split in ["Test", "Val"]:
                    clone_dir = os.path.join(dataset_dir, split, class_name, video_id)
                    shutil.copytree(train_path, clone_dir)
            else:
                # Move instances to Test and Val
                random.shuffle(train_instances)
                for i, video_id in enumerate(train_instances[:num_to_move]):
                    train_path = os.path.join(dataset_dir, "Train", class_name, video_id)
                    target_split = "Test" if i % 2 == 0 else "Val"
                    target_dir = os.path.join(dataset_dir, target_split, class_name, video_id)
                    shutil.move(train_path, target_dir)

    print("Dataset processing and balancing completed.")


# Parameters
file_path = './data/input/WLASL_v0.3.json'
missing_file_path = './data/input/missing.txt'
videos_dir = './data/input/videos'
dataset_dir = './data/working/Dataset/frames'

# Run the processing function
process_wlasl_data(file_path, missing_file_path, videos_dir, dataset_dir, move_percentage=0.3)
