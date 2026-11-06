import keras
import numpy as np

IMAGE_SIZE = 224


def build_model():
    # The input is one color image: 224 x 224 pixels with 3 colors
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    # MobileNetV2 expects pixel values from -1 to 1, not 0 to 255
    x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)

    # Load the pretrained model without its original final layer
    base = keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Freeze it so its weights don't change yet

    x = base(x, training=False)

    # Shrink what it found into one list of numbers
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Our new layer outputs one hazard score from 0 to 1
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs, outputs)


if __name__ == "__main__":
    model = build_model()

    # Make one random fake image to test the model
    fake_image = np.random.randint(0, 256, size=(1, IMAGE_SIZE, IMAGE_SIZE, 3))
    score = model.predict(fake_image, verbose=0)[0][0]
    print("Hazard score:", score)