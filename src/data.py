import csv
import os

import tensorflow as tf

from model import IMAGE_SIZE

DATA_DIR = "data/data_set"
LABELS_FILE = os.path.join(DATA_DIR, "annotations", "classification.csv")
BATCH_SIZE = 8

# Turn the label words into numbers: no cliff = 0, cliff = 1
LABEL_NUMBERS = {"no_cliff_drop": 0, "cliff_drop": 1}


def read_labels(split):
    """Read the list file and keep only the pictures in one split."""
    paths = []
    labels = []
    with open(LABELS_FILE, newline="") as f:
        for row in csv.DictReader(f):
            image = row["image"]   # like images/train/img_000001.jpg
            label = row["label"]   # like cliff_drop

            if label not in LABEL_NUMBERS:
                if label != "unclear":
                    print(f"Skipping unknown label '{label}' for {image}")
                continue

            # The middle part of the path is the split: train, validate, or test
            if image.split("/")[1] != split:
                continue

            paths.append(os.path.join(DATA_DIR, image))
            labels.append([float(LABEL_NUMBERS[label])])
    return paths, labels


def load_image(path, label):
    """Open one picture and resize it for the model."""
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image.set_shape((IMAGE_SIZE, IMAGE_SIZE, 3))
    return image, label


def load_data(split, shuffle=False):
    """Build the pictures and labels for one split."""
    paths, labels = read_labels(split)
    data = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        data = data.shuffle(len(paths), seed=42)  # Mix up the order for training
    return data.map(load_image).batch(BATCH_SIZE)


if __name__ == "__main__":
    for split in ["train", "validate", "test"]:
        paths, labels = read_labels(split)
        cliffs = sum(label[0] for label in labels)
        print(f"{split}: {len(paths)} images ({int(cliffs)} cliff)")

    for images, labels in load_data("train").take(1):
        print("Images shape:", images.shape)