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

# Data paths and preprocessing
data_path = './data/working/Dataset/npy_arrays'

words = sorted([folder for folder in os.listdir(os.path.join(data_path, 'Train')) 
                if os.path.isdir(os.path.join(data_path, 'Train', folder))])
label_map = {label: num for num, label in enumerate(words)}


# Define the function to preprocess data
def preprocess_data(data_path, split, f_avg):
    sequences = []
    labels = []
    
    # Load all words in the split directory
    words = np.array(os.listdir(os.path.join(data_path, split)))
    label_map = {label: num for num, label in enumerate(words)}
    
    for word in tqdm(words, desc=f"Processing {split} data"):
        word_path = os.path.join(data_path, split, word)
        lh_keypoints_folder = os.path.join(word_path, "lh_keypoints")
        rh_keypoints_folder = os.path.join(word_path, "rh_keypoints")
        pose_keypoints_folder = os.path.join(word_path, "pose_keypoints")
        
        # Skip words with missing keypoints folders
        if not all(os.path.exists(folder) for folder in [lh_keypoints_folder, rh_keypoints_folder, pose_keypoints_folder]):
            print(f"Skipping word '{word}' due to missing keypoints folder(s).")
            continue
        
        # Process each sequence in the word directory
        for sequence in os.listdir(lh_keypoints_folder):
            try:
                # Load and process left-hand keypoints
                res_lh = np.load(os.path.join(lh_keypoints_folder, sequence))
                res_lh = pad_or_truncate(res_lh, f_avg)
                
                # Load and process right-hand keypoints
                res_rh = np.load(os.path.join(rh_keypoints_folder, sequence))
                res_rh = pad_or_truncate(res_rh, f_avg)
                
                # Load and process pose keypoints
                res_pose = np.load(os.path.join(pose_keypoints_folder, sequence))
                res_pose = pad_or_truncate(res_pose, f_avg)
                
                # Concatenate features and append to dataset
                sequences.append(np.concatenate((res_pose, res_lh, res_rh), axis=1))
                labels.append(label_map[word])
            
            except Exception as e:
                print(f"Error processing sequence {sequence}: {e}")
                continue
    
    return np.array(sequences), np.array(labels)

# Define helper function to pad or truncate sequences
def pad_or_truncate(keypoints, f_avg):
    """
    Pads or truncates a keypoint sequence to the desired number of frames.
    """
    num_frames = min(keypoints.shape[0], f_avg)
    keypoints = keypoints[:num_frames, :]
    while num_frames < f_avg:
        keypoints = np.concatenate((keypoints, np.expand_dims(keypoints[-1, :], axis=0)), axis=0)
        num_frames += 1
    return keypoints


# Preprocess Train, Test, and Val datasets
X_train, y_train = preprocess_data(data_path, 'Train', 48)
X_val, y_val = preprocess_data(data_path, 'Val', 48)
X_test, y_test = preprocess_data(data_path, 'Test', 48)

# Ensure one-hot encoding of the labels
num_classes = len(os.listdir(os.path.join(data_path, 'Train')))
y_train = to_categorical(y_train, num_classes=num_classes)
y_val = to_categorical(y_val, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

print("Data preprocessing complete!")
print(f"Train samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}, Test samples: {X_test.shape[0]}")


# Define the Bidirectional LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=False)),   
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
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