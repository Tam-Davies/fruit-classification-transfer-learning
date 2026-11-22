import json
import tensorflow as tf

from .config import (
    INITIAL_EPOCHS,
    FINE_TUNE_EPOCHS,
    STEPS_PER_EPOCH,
    VALIDATION_STEPS,
    INITIAL_MODEL_PATH,
    MODEL_PATH,
    EARLY_STOPPING_PATIENCE,
    LR_PATIENCE,
    LR_REDUCTION_FACTOR,
    MIN_LEARNING_RATE,
    RANDOM_SEED,
)

from .data import create_data_generators
from .model import build_model, prepare_fine_tuning


def create_callbacks():
    """Create callbacks for each training phase."""

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=LR_REDUCTION_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
        verbose=1,
    )

    return [early_stopping, reduce_lr]


def save_history(history, filename):
    """Save training history to JSON."""

    with open(filename, "w") as file:
        json.dump(history.history, file, indent=4)


def main():

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # --------------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------------
    print("\nLoading datasets...")

    train_generator, val_generator, test_generator = (
        create_data_generators()
    )

    num_classes = len(train_generator.class_indices)

    print(f"\nNumber of fruit classes: {num_classes}")
    print(f"Class mapping: {train_generator.class_indices}")

    # --------------------------------------------------------
    # 2. BUILD MODEL
    # --------------------------------------------------------
    print("\nBuilding VGG16 model...")

    model, base_model = build_model(
        num_classes=num_classes
    )

    model.summary()

    # --------------------------------------------------------
    # 3. INITIAL TRAINING
    # --------------------------------------------------------
    print("\nStarting initial training phase...")

    initial_history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_generator,
        validation_steps=VALIDATION_STEPS,
        epochs=INITIAL_EPOCHS,
        callbacks=create_callbacks(),
        verbose=1,
    )

    # Save initial model
    model.save(INITIAL_MODEL_PATH)

    print(f"\nInitial model saved to: {INITIAL_MODEL_PATH}")

    # --------------------------------------------------------
    # 4. FINE-TUNING
    # --------------------------------------------------------
    print("\nPreparing model for fine-tuning...")

    model = prepare_fine_tuning(
        model=model,
        base_model=base_model,
    )

    model.summary()

    print("\nStarting fine-tuning phase...")

    fine_tune_history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_generator,
        validation_steps=VALIDATION_STEPS,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=create_callbacks(),
        verbose=1,
    )

    # --------------------------------------------------------
    # 5. SAVE FINAL MODEL
    # --------------------------------------------------------
    model.save(MODEL_PATH)

    print(f"\nFinal model saved to: {MODEL_PATH}")

    # --------------------------------------------------------
    # 6. SAVE TRAINING HISTORY
    # --------------------------------------------------------
    combined_history = {}

    for metric in initial_history.history:
        combined_history[metric] = (
            initial_history.history[metric]
            + fine_tune_history.history[metric]
        )

    save_history(
        type("HistoryObject", (), {
            "history": combined_history
        })(),
        "outputs/training_history.json",
    )

    print("\nTraining history saved to outputs/training_history.json")

    # --------------------------------------------------------
    # 7. RETURN TRAINED MODEL AND DATA
    # --------------------------------------------------------
    return model, test_generator, combined_history


if __name__ == "__main__":
    main()