import keras
import numpy as np

from data import load_data

THRESHOLD = 0.5  # A score at or above this counts as a cliff


def main():
    val_data = load_data("test")  # The final exam: pictures it never saw
    model = keras.saving.load_model("models/practice_classifier.keras")

    all_labels = []
    all_scores = []

    # Go through the validation images one batch at a time
    for images, labels in val_data:
        scores = model.predict(images, verbose=0)
        all_labels.extend(labels.numpy().flatten())
        all_scores.extend(scores.flatten())

    labels = np.array(all_labels) == 1          # True = really a cliff
    guesses = np.array(all_scores) >= THRESHOLD  # True = model said cliff

    # Count the four kinds of results
    tp = np.sum(guesses & labels)    # Said cliff, really a cliff
    fp = np.sum(guesses & ~labels)   # Said cliff, really no cliff (false alarm)
    tn = np.sum(~guesses & ~labels)  # Said no cliff, really no cliff
    fn = np.sum(~guesses & labels)   # Said no cliff, really a cliff (missed cliff)

    accuracy = (tp + tn) / len(labels)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    print(f"Images tested: {len(labels)}")
    print(f"Confusion matrix:  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"Missed cliffs: {fn}")


if __name__ == "__main__":
    main()