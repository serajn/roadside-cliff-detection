import keras

from data import load_data
from model import build_model

EPOCHS = 10


def main():
    train_data = load_data("train", shuffle=True)
    val_data = load_data("val")
    model = build_model()

    # Set up how the model learns and what we want to measure
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Recall(name="recall")],
    )

    # Train the model
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

    # Save the trained model
    model.save("models/practice_classifier.keras")
    print("Model saved!")


if __name__ == "__main__":
    main()