
import json
import math

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.fruit_classifier.config import (
    MODEL_PATH,
    OUTPUT_DIR,
    FIGURES_DIR,
)
from src.fruit_classifier.data import create_data_generators
import tensorflow as tf


def main():
    _, _, test_generator = create_data_generators()

    print(f"Test images: {test_generator.samples}")
    print(f"Number of classes: {test_generator.num_classes}")

    # Calculate steps required to cover the entire test dataset
    test_steps = math.ceil(
        test_generator.samples / test_generator.batch_size
    )

    print(f"Evaluation steps: {test_steps}")

    print(f"\nLoading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("\nEvaluating entire test dataset...")

    test_loss, test_accuracy = model.evaluate(
        test_generator,
        steps=test_steps,
        verbose=1
    )

    print("\nFull Test Results")
    print("=" * 40)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Reset generator before predictions
    test_generator.reset()

    predictions = model.predict(
        test_generator,
        steps=test_steps,
        verbose=1
    )

    predicted_classes = np.argmax(predictions, axis=1)

    # Ensure predictions don't exceed actual test sample count
    predicted_classes = predicted_classes[:test_generator.samples]

    true_classes = test_generator.classes[:len(predicted_classes)]

    class_labels = list(test_generator.class_indices.keys())

    # Classification report
    report = classification_report(
        true_classes,
        predicted_classes,
        labels=list(range(len(class_labels))),
        target_names=class_labels,
        zero_division=0
    )

    print("\nClassification Report")
    print("=" * 40)
    print(report)

    # Save classification report
    report_path = OUTPUT_DIR / "classification_report.txt"

    with open(report_path, "w") as file:
        file.write(report)

    # Confusion matrix
    cm = confusion_matrix(
        true_classes,
        predicted_classes,
        labels=list(range(len(class_labels)))
    )

    fig, ax = plt.subplots(figsize=(16, 14))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_labels
    )

    display.plot(
        ax=ax,
        xticks_rotation=90,
        colorbar=True,
        values_format="d"
    )

    plt.title("VGG16 Fruit Classification - Full Test Confusion Matrix")
    plt.tight_layout()

    confusion_matrix_path = FIGURES_DIR / "confusion_matrix.png"

    plt.savefig(
        confusion_matrix_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # Save evaluation metrics
    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "test_accuracy_percent": float(test_accuracy * 100),
        "test_samples_evaluated": int(len(true_classes)),
        "total_test_samples": int(test_generator.samples),
        "num_classes": int(test_generator.num_classes),
    }

    metrics_path = OUTPUT_DIR / "evaluation_metrics.json"

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\nClassification report saved: {report_path}")
    print(f"Confusion matrix saved: {confusion_matrix_path}")
    print(f"Metrics saved: {metrics_path}")
    print("\nFull test evaluation completed.")


if __name__ == "__main__":
    main()
