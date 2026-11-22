
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.fruit_classifier.config import MODEL_PATH, IMAGE_SIZE
from src.fruit_classifier.data import create_data_generators


def predict_image(image_path, top_k=3):
    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Get class labels from training dataset
    train_generator, _, _ = create_data_generators()
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())

    # Load and preprocess image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Make prediction
    predictions = model.predict(image_array, verbose=0)[0]

    top_indices = np.argsort(predictions)[::-1][:top_k]

    print("\nFruit Classification Results")
    print("=" * 35)

    for rank, index in enumerate(top_indices, start=1):
        print(
            f"{rank}. {class_names[index]}: "
            f"{predictions[index] * 100:.2f}% confidence"
        )

    # Display image and prediction
    plt.figure(figsize=(7, 7))
    plt.imshow(image)
    plt.axis("off")

    predicted_class = class_names[top_indices[0]]
    confidence = predictions[top_indices[0]] * 100

    plt.title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence:.2f}%"
    )

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Predict fruit class using trained VGG16 model"
    )

    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the image to classify"
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of top predictions to display"
    )

    args = parser.parse_args()

    predict_image(args.image_path, args.top_k)


if __name__ == "__main__":
    main()
