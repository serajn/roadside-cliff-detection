import sys

import keras
import numpy as np

from model import IMAGE_SIZE


def main():
    image_path = sys.argv[1]  # The image path you type in the command

    model = keras.saving.load_model("models/practice_classifier.keras")

    # Load the image and resize it
    image = keras.utils.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))

    # Turn it into numbers, as a batch of 1 image
    batch = np.expand_dims(keras.utils.img_to_array(image), axis=0)

    # Get the hazard score
    score = float(model.predict(batch, verbose=0)[0][0])

    label = "cliff_drop" if score >= 0.5 else "no_cliff_drop"
    print(f"Hazard score: {score:.3f}")
    print(f"Prediction: {label}")


if __name__ == "__main__":
    main()