import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === GPU CONFIGURATION ===
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    print("Using GPU for training!")
else:
    print("No GPU found. Using CPU.")

# === CONFIGURATION ===
SEQUENCE_LENGTH = 64
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
EPOCHS = 20
TRAIN_DIR = './data/working/Dataset/frames/Train'
TEST_DIR = './data/working/Dataset/frames/Test'

# Create 'results' directory for outputs
RESULTS_DIR = './results'
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, 'i3d_model.h5')
TFLITE_SAVE_PATH = os.path.join(RESULTS_DIR, 'i3d_model.tflite')
CONFUSION_MATRIX_PATH = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
TRAINING_HISTORY_PATH = os.path.join(RESULTS_DIR, 'training_history.png')

# === 1. DATA LOADING AND PREPROCESSING ===
def load_frames(sequence_dir, sequence_length=64, img_size=(224, 224)):
    """Load and preprocess video frames."""
    video_frames = sorted(os.listdir(sequence_dir))
    frames = []
    for frame_file in video_frames:
        frame_path = os.path.join(sequence_dir, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, img_size) / 255.0  # Normalize to [0, 1]
        frames.append(frame)

    # Pad or truncate to fixed length
    if len(frames) < sequence_length:
        padding = [np.zeros((img_size[0], img_size[1], 3))] * (sequence_length - len(frames))
        frames += padding
    else:
        frames = frames[:sequence_length]

    return np.array(frames)

def create_tf_dataset(data_dirs, labels, batch_size=8, sequence_length=64, img_size=(224, 224)):
    """Create a TensorFlow dataset from frame sequences and labels."""
    X, y = [], []
    for i, video_dir in enumerate(data_dirs):
        frames = load_frames(video_dir, sequence_length=sequence_length, img_size=img_size)
        X.append(frames)
        y.append(labels[i])

    X, y = np.array(X), np.array(y)
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# Load data directories and labels
def get_data_and_labels(base_dir):
    """Retrieve video directories and their corresponding labels."""
    class_labels = os.listdir(base_dir)
    data_dirs, labels = [], []
    for label, class_name in enumerate(class_labels):
        class_dir = os.path.join(base_dir, class_name)
        video_dirs = [os.path.join(class_dir, video) for video in os.listdir(class_dir)]
        data_dirs.extend(video_dirs)
        labels.extend([label] * len(video_dirs))
    return data_dirs, labels, class_labels

train_dirs, train_labels, class_labels = get_data_and_labels(TRAIN_DIR)
test_dirs, test_labels, _ = get_data_and_labels(TEST_DIR)

# Create datasets
train_dataset = create_tf_dataset(train_dirs, train_labels, batch_size=BATCH_SIZE)
test_dataset = create_tf_dataset(test_dirs, test_labels, batch_size=BATCH_SIZE)

# === 2. MODEL CREATION ===
def create_i3d_model(input_shape, num_classes):
    """Create and compile the I3D model."""
    i3d_layer = hub.KerasLayer("https://tfhub.dev/deepmind/i3d-kinetics-400/1", trainable=True)

    inputs = layers.Input(shape=input_shape)
    x = i3d_layer(inputs)
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(1024, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

input_shape = (SEQUENCE_LENGTH, IMG_SIZE[0], IMG_SIZE[1], 3)
model = create_i3d_model(input_shape, len(class_labels))

# === 3. MODEL TRAINING ===
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
    ]
)

# === 4. EVALUATION ===
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Confusion Matrix
y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)
y_pred = np.argmax(model.predict(test_dataset), axis=1)

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(CONFUSION_MATRIX_PATH)  # Save confusion matrix
plt.show()

# Plot Training History
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig(TRAINING_HISTORY_PATH)  # Save training history
plt.show()

# === 5. MODEL DEPLOYMENT ===
# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TensorFlow Lite model
with open(TFLITE_SAVE_PATH, 'wb') as f:
    f.write(tflite_model)

print(f"Model saved to {MODEL_SAVE_PATH}")
print(f"TensorFlow Lite model saved to {TFLITE_SAVE_PATH}")
print("Confusion matrix and training history saved in 'results' folder.")
