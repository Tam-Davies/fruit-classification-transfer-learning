from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam

from .config import (
    INPUT_SHAPE,
    DENSE_UNITS,
    DROPOUT_RATE,
    INITIAL_LEARNING_RATE,
    FINE_TUNE_LEARNING_RATE,
    FINE_TUNE_LAST_LAYERS,
)


def build_model(num_classes):
    """
    Build VGG16 transfer-learning model.

    Phase 1:
        Train the custom classification head with frozen VGG16 layers.

    Phase 2:
        Unfreeze the last configured VGG16 layers for fine-tuning.
    """

    # Input layer
    inputs = Input(shape=INPUT_SHAPE)

    # Pretrained VGG16 feature extractor
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=INPUT_SHAPE,
    )

    # Freeze the pretrained feature extractor initially
    base_model.trainable = False

    # Feature extraction
    x = base_model(inputs, training=False)

    # Classification head
    x = Flatten()(x)

    x = Dense(
        DENSE_UNITS,
        activation="relu",
    )(x)

    x = Dropout(
        DROPOUT_RATE,
    )(x)

    outputs = Dense(
        num_classes,
        activation="softmax",
    )(x)

    # Complete model
    model = Model(
        inputs=inputs,
        outputs=outputs,
        name="fruit_vgg16_classifier",
    )

    # Initial training configuration
    model.compile(
        optimizer=Adam(
            learning_rate=INITIAL_LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model, base_model


def prepare_fine_tuning(model, base_model):
    """
    Unfreeze the last configured layers of VGG16
    and recompile the model with a lower learning rate.
    """

    base_model.trainable = True

    # Freeze all but the final configured layers
    for layer in base_model.layers[:-FINE_TUNE_LAST_LAYERS]:
        layer.trainable = False

    # Recompile after changing trainability
    model.compile(
        optimizer=Adam(
            learning_rate=FINE_TUNE_LEARNING_RATE
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model