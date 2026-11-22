
import json
import matplotlib.pyplot as plt

from src.fruit_classifier.config import (
    OUTPUT_DIR,
    FIGURES_DIR,
    INITIAL_EPOCHS,
    ACCURACY_PLOT_PATH,
    LOSS_PLOT_PATH,
)


def load_training_history():
    """Load saved training history from JSON."""
    history_path = OUTPUT_DIR / "training_history.json"

    if not history_path.exists():
        raise FileNotFoundError(
            f"Training history not found: {history_path}"
        )

    with open(history_path, "r") as file:
        history = json.load(file)

    return history


def plot_accuracy(history):
    """Plot training and validation accuracy."""

    accuracy = history.get("accuracy", [])
    val_accuracy = history.get("val_accuracy", [])

    if not accuracy or not val_accuracy:
        raise ValueError(
            "Accuracy metrics are missing from training history."
        )

    epochs = range(1, len(accuracy) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        accuracy,
        marker="o",
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        val_accuracy,
        marker="o",
        label="Validation Accuracy"
    )

    if INITIAL_EPOCHS < len(accuracy):
        plt.axvline(
            x=INITIAL_EPOCHS + 0.5,
            linestyle="--",
            label="Fine-tuning begins"
        )

    plt.title("VGG16 Fruit Classification - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(ACCURACY_PLOT_PATH, dpi=300)
    plt.show()

    print(f"Accuracy plot saved to: {ACCURACY_PLOT_PATH}")


def plot_loss(history):
    """Plot training and validation loss."""

    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])

    if not loss or not val_loss:
        raise ValueError(
            "Loss metrics are missing from training history."
        )

    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        loss,
        marker="o",
        label="Training Loss"
    )

    plt.plot(
        epochs,
        val_loss,
        marker="o",
        label="Validation Loss"
    )

    if INITIAL_EPOCHS < len(loss):
        plt.axvline(
            x=INITIAL_EPOCHS + 0.5,
            linestyle="--",
            label="Fine-tuning begins"
        )

    plt.title("VGG16 Fruit Classification - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xticks(list(epochs))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(LOSS_PLOT_PATH, dpi=300)
    plt.show()

    print(f"Loss plot saved to: {LOSS_PLOT_PATH}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    history = load_training_history()

    plot_accuracy(history)
    plot_loss(history)

    print("\nTraining visualization completed successfully.")


if __name__ == "__main__":
    main()
