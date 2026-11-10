import keras

from data import load_data, read_labels
from model import build_model

EPOCHS = 10

def get_class_weights():
    """Make cliffs count more, since there are fewer of them."""
    _, labels = read_labels("train")
    total = len(labels)
    cliffs = sum(label[0] for label in labels)
    no_cliffs = total - cliffs
    weights = {0: total / (2 * no_cliffs), 1: total / (2 * cliffs)}
    print(f"Class weights: no cliff = {weights[0]:.2f}, cliff = {weights[1]:.2f}")
    return weights

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
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS,
              class_weight=get_class_weights())
    # Save the trained model
    model.save("models/practice_classifier.keras")
    print("Model saved!")


if __name__ == "__main__":
    main()