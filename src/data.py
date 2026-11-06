import keras

from model import IMAGE_SIZE

DATA_DIR = "data/practice"
BATCH_SIZE = 8


def load_data():
    train_data, val_data = keras.utils.image_dataset_from_directory(
        DATA_DIR,
        label_mode="binary",
        class_names=["no_cliff_drop", "cliff_drop"],  # no cliff = 0, cliff = 1
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        validation_split=0.25,  # Keep 25% of the images for checking the model
        subset="both",
        seed=42,  # Split the images the same way every time
    )
    return train_data, val_data


if __name__ == "__main__":
    train_data, val_data = load_data()

    # Look at one group of training images
    for images, labels in train_data.take(1):
        print("Images shape:", images.shape)
        print("Labels:", labels.numpy().flatten())