import os
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_dataset_efficientnet(base_dir, img_size):
    images = []
    labels_left = []
    labels_right = []

    for label_type in ['negative', 'positive']:
        dir_path = os.path.join(base_dir, label_type)
        for file_name in os.listdir(dir_path):
            if not file_name.endswith('.npy'):
                continue

            file_path = os.path.join(dir_path, file_name)
            try:
                image_data = np.load(file_path, allow_pickle=True)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            base_name, left_label, right_label = file_name.split('.')[0].split('_')
            if 'x' in left_label or 'x' in right_label:
                continue

            images.append(image_data)
            labels_left.append(int(left_label))
            labels_right.append(int(right_label))

    images = np.array(images)
    labels_left = np.array(labels_left)
    labels_right = np.array(labels_right)

    return images, labels_left, labels_right

def preprocess_data(images, labels_left, labels_right, num_classes):
    images = images.astype('float32') / 255.0
    labels_left = to_categorical(labels_left, num_classes)
    labels_right = to_categorical(labels_right, num_classes)
    return images, labels_left, labels_right
