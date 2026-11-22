from tensorflow.keras.preprocessing.image import ImageDataGenerator

from .config import (
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
    ROTATION_RANGE,
    WIDTH_SHIFT_RANGE,
    HEIGHT_SHIFT_RANGE,
    SHEAR_RANGE,
    ZOOM_RANGE,
    HORIZONTAL_FLIP,
    FILL_MODE,
)


def create_data_generators():
    """
    Create training, validation, and testing data generators.

    Augmentation is applied only to training data.
    Validation and test data are only rescaled.
    """

    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        fill_mode=FILL_MODE,
    )

    # Validation preprocessing
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    # Test preprocessing
    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255
    )

    train_generator = train_datagen.flow_from_directory(
        directory=str(TRAIN_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=RANDOM_SEED,
    )

    val_generator = val_datagen.flow_from_directory(
        directory=str(VAL_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    test_generator = test_datagen.flow_from_directory(
        directory=str(TEST_DIR),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    # Ensure class indices are consistent
    if train_generator.class_indices != val_generator.class_indices:
        raise ValueError("Training and validation class mappings do not match.")

    if train_generator.class_indices != test_generator.class_indices:
        raise ValueError("Training and test class mappings do not match.")

    return train_generator, val_generator, test_generator