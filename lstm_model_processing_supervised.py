import os
import cv2
import json
import pandas as pd
import datetime as dt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
np.random.seed(42)
# import os
# import shutil
# import random
# import numpy as np

# def is_valid_npy(file_path):
#     """
#     Check if the given .npy file has valid dimensions (2D).
#     """
#     try:
#         data = np.load(file_path)
#         return data.ndim == 2
#     except Exception as e:
#         print(f"Error reading {file_path}: {e}")
#         return False

# def balance_train_test_by_npy(train_path, test_path, move_percentage=0.3):
#     """
#     Validate .npy files in Test directories. For invalid words, balance Train and Test directories
#     by moving/copying files from Train to Test.
#     """
#     train_words = os.listdir(train_path)
#     test_words = os.listdir(test_path)

#     for word in train_words:
#         train_word_path = os.path.join(train_path, word)
#         test_word_path = os.path.join(test_path, word)
#         os.makedirs(test_word_path, exist_ok=True)

#         # Get npy files in Test directory
#         test_lh_dir = os.path.join(test_word_path, "lh_keypoints")
#         test_rh_dir = os.path.join(test_word_path, "rh_keypoints")
#         test_pose_dir = os.path.join(test_word_path, "pose_keypoints")
#         invalid_files = []

#         for test_dir in [test_lh_dir, test_rh_dir, test_pose_dir]:
#             if os.path.exists(test_dir):
#                 for file in os.listdir(test_dir):
#                     file_path = os.path.join(test_dir, file)
#                     if not is_valid_npy(file_path):
#                         invalid_files.append(file)

#         # If invalid files found, balance Train and Test
#         if invalid_files:
#             print(f"Invalid files found for word '{word}' in Test: {invalid_files}")

#             # Move files from Train to Test
#             train_lh_dir = os.path.join(train_word_path, "lh_keypoints")
#             train_rh_dir = os.path.join(train_word_path, "rh_keypoints")
#             train_pose_dir = os.path.join(train_word_path, "pose_keypoints")

#             for train_dir, test_dir in zip(
#                 [train_lh_dir, train_rh_dir, train_pose_dir],
#                 [test_lh_dir, test_rh_dir, test_pose_dir]
#             ):
#                 if not os.path.exists(train_dir):
#                     continue
#                 train_files = os.listdir(train_dir)

#                 # Determine how many files to move
#                 num_to_move = max(int(len(train_files) * move_percentage), 1)
#                 if len(train_files) == 1:
#                     # Copy the single file to Test
#                     file_to_copy = train_files[0]
#                     src = os.path.join(train_dir, file_to_copy)
#                     dst = os.path.join(test_dir, file_to_copy)
#                     shutil.copy(src, dst)
#                 else:
#                     # Move a percentage of files
#                     files_to_move = random.sample(train_files, num_to_move)
#                     for file in files_to_move:
#                         src = os.path.join(train_dir, file)
#                         dst = os.path.join(test_dir, file)
#                         shutil.move(src, dst)

#     print("Balancing based on npy files completed.")

# # Example Usage
# train_dir = "./data/working/Dataset/npy_arrays/Train"
# test_dir = "./data/working/Dataset/npy_arrays/Test"
# balance_train_test_by_npy(train_dir, test_dir, move_percentage=0.3)


words = sorted(np.array(os.listdir(os.path.join("./data/working/Dataset/frames", 'Train'))))  # Ensure consistent order
label_map = {label: num for num, label in enumerate(words) if label in words}

import os
import numpy as np
from tqdm import tqdm

def preprocess_data(data_path, split, f_avg):
    sequences = []
    labels = []
    
    words = np.array(os.listdir(os.path.join(data_path, split)))
    label_map = {label: num for num, label in enumerate(words)}
    
    for word in tqdm(words):
        word_path = os.path.join(data_path, split, word)
        lh_keypoints_folder = os.path.join(word_path, "lh_keypoints")
        rh_keypoints_folder = os.path.join(word_path, "rh_keypoints")
        pose_keypoints_folder = os.path.join(word_path, "pose_keypoints")
        
        # Check if all required folders exist
        if not all(os.path.exists(folder) for folder in [lh_keypoints_folder, rh_keypoints_folder, pose_keypoints_folder]):
            print(f"Skipping word '{word}' due to missing keypoints folder(s).")
            continue
        
        for sequence in os.listdir(lh_keypoints_folder):
            try:
                # Load left-hand keypoints
                res_lh = np.load(os.path.join(lh_keypoints_folder, sequence))
                if res_lh.ndim != 2:
                    continue
                num_frames = min(res_lh.shape[0], f_avg)
                res_lh = res_lh[:num_frames, :]
                while num_frames < f_avg:
                    res_lh = np.concatenate((res_lh, np.expand_dims(res_lh[-1, :], axis=0)), axis=0)
                    num_frames += 1

                # Load right-hand keypoints
                res_rh = np.load(os.path.join(rh_keypoints_folder, sequence))
                if res_rh.ndim != 2:
                    continue
                num_frames = min(res_rh.shape[0], f_avg)
                res_rh = res_rh[:num_frames, :]
                while num_frames < f_avg:
                    res_rh = np.concatenate((res_rh, np.expand_dims(res_rh[-1, :], axis=0)), axis=0)
                    num_frames += 1

                # Load pose keypoints
                res_pose = np.load(os.path.join(pose_keypoints_folder, sequence))
                if res_pose.ndim != 2:
                    continue
                num_frames = min(res_pose.shape[0], f_avg)
                res_pose = res_pose[:num_frames, :]
                while num_frames < f_avg:
                    res_pose = np.concatenate((res_pose, np.expand_dims(res_pose[-1, :], axis=0)), axis=0)
                    num_frames += 1

                # Concatenate all features
                sequences.append(np.concatenate((res_pose, res_lh, res_rh), axis=1))
                labels.append(label_map[word])
            
            except FileNotFoundError as e:
                print(f"Skipping file due to missing file: {e.filename}")
                continue
            
            except Exception as e:
                print(f"An unexpected error occurred while processing {sequence}: {e}")
                continue
    
    return np.array(sequences), np.array(labels)




data_path = './data/working/Dataset/npy_arrays'
X_train, y_train = preprocess_data(data_path, 'Train', 48)
X_test, y_test = preprocess_data(data_path, 'Test', 48)

from collections import Counter

# Count instances per class
class_counts = Counter(y_train)
rare_classes = {cls for cls, count in class_counts.items() if count == 1}

# Separate rare classes
rare_indices = [i for i, label in enumerate(y_train) if label in rare_classes]
common_indices = [i for i, label in enumerate(y_train) if label not in rare_classes]

X_rare = X_train[rare_indices]
y_rare = y_train[rare_indices]

X_common = X_train[common_indices]
y_common = y_train[common_indices]

# Split common classes
X_common_train, X_val, y_common_train, y_val = train_test_split(
    X_common, y_common, test_size=0.35, random_state=42, stratify=y_common
)

# Add rare classes to training set
X_train = np.concatenate([X_common_train, X_rare], axis=0)
y_train = np.concatenate([y_common_train, y_rare], axis=0)

# Ensure one-hot encoding of the labels
y_train = to_categorical(y_train, num_classes=len(label_map))
y_val = to_categorical(y_val, num_classes=len(label_map))
y_test = to_categorical(y_test, num_classes=len(label_map))

# Define the Bidirectional LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=False)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(len(words), activation='softmax')
])

# Compile the model

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Set up early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',  # Metric to monitor for early stopping
    mode='min',  # Set mode to 'min' for minimizing the metric
    patience=15,  # Number of epochs with no improvement before stopping
    restore_best_weights=True,  # Restore the best model weights
    verbose=1
)

model_training_history = model.fit(X_train, y_train, batch_size=32, 
                                   validation_data=(X_val,y_val), validation_batch_size=32, 
                                   epochs=150)

# Evaluate the model on train data
model_evaluation_history = model.evaluate(X_train, y_train)

# Evaluate the model on test data
model_evaluation_history = model.evaluate(X_test, y_test)

def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):


    # Get metric values using metric names as identifiers.
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_value_1))

    # Plot the Graph.
    plt.plot(epochs, metric_value_1, 'blue', label = metric_name_1, linestyle = 'solid')
    plt.plot(epochs, metric_value_2, 'red', label = metric_name_2, linestyle = 'dotted')

    # Add title to the plot.
    plt.title(str(plot_name))

    # Add legend to the plot.
    plt.legend()

# Visualize the training and validation loss metrices.
plot_metric(model_training_history, 'loss', 'val_loss', 'Total Loss vs Total Validation Loss')

# Visualize the training and validation accuracy metrices.
plot_metric(model_training_history, 'categorical_accuracy', 'val_categorical_accuracy', 'Total Accuracy vs Total Validation Accuracy')

#Predicted sign
res = model.predict(X_test)
words[np.argmax(res[1])]

#Real sign
words[np.argmax(y_test[1])]

# Get the loss and accuracy from model_evaluation_history.
model_evaluation_loss, model_evaluation_accuracy = model_evaluation_history

# Define the string date format.
# Get the current Date and Time in a DateTime Object.
# Convert the DateTime object to string according to the style mentioned in date_time_format string.
date_time_format = '%Y_%m_%d__%H_%M_%S'
current_date_time_dt = dt.datetime.now()
current_date_time_string = dt.datetime.strftime(current_date_time_dt, date_time_format)

# Define a useful name for our model to make it easy for us while navigating through multiple saved models.
model_file_name = f'ASL_Date_Time_{current_date_time_string}___Loss_{model_evaluation_loss}___Accuracy_{model_evaluation_accuracy}.h5'

os.makedirs('./models/working/Model',exist_ok=True)
# Save your Model.
model.save(f'./models/working/Model/{model_file_name}')

yhat = model.predict(X_test)

def get_key_by_value(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

y = []
for v in ytrue:
    y.append(get_key_by_value(label_map, v))
print(y)



ypred = []
for v in yhat:
    ypred.append(get_key_by_value(label_map, v))
print(ypred)



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y and ypred are your target labels and predicted labels, respectively

# Select the first 20 classes
y_subset = y[:200]
ypred_subset = ypred[:200]

# Get unique class labels
class_labels = np.unique(y_subset)

# Compute confusion matrix
cm = confusion_matrix(y_subset, ypred_subset, labels=class_labels)

# Create a DataFrame from the confusion matrix
df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.3)  # for label size
sns.heatmap(df_cm, cmap="Blues", annot=True, fmt="d", annot_kws={"size": 12})
plt.title("Confusion Matrix - First 20 Classes")
plt.show()

# save model
model.save('./models/model.h5')