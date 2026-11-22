# Fruit Classification Using Transfer Learning

## Overview

This project implements a multiclass fruit image classification model using TensorFlow/Keras and transfer learning with VGG16 pretrained on ImageNet.

The model is trained to recognize **24 fruit and vegetable classes** from images using a two-stage training approach: feature extraction followed by fine-tuning.

## Project Objectives

- Implement transfer learning using VGG16.
- Apply image augmentation to training data.
- Train a custom classification head.
- Fine-tune selected pretrained layers.
- Evaluate model performance on a test dataset.
- Visualize training performance and predictions.

## Model Architecture

- **Base model:** VGG16
- **Pretrained weights:** ImageNet
- **Input dimensions:** 64 × 64 × 3
- **Classification head:** Flatten → Dense (256, ReLU) → Dropout (0.3) → Softmax
- **Number of classes:** 24
- **Optimizer:** Adam
- **Loss function:** Categorical Crossentropy

### Training Strategy

**Phase 1 — Feature Extraction**
- Freeze the pretrained VGG16 backbone.
- Train the custom classification head.

**Phase 2 — Fine-tuning**
- Unfreeze selected VGG16 layers.
- Fine-tune using a lower learning rate of 0.00001.

## Dataset

Fruit-360 Original Size dataset.

Dataset structure:

```text
data/
└── fruits-360-original-size/
    ├── Training/
    ├── Validation/
    └── Test/
```

The dataset is not included in this repository.

## Results

Full test-set evaluation:

| Metric | Result |
|---|---:|
| Test Accuracy | 79.26% |
| Test Loss | 0.5512 |
| Weighted Precision | 0.85 |
| Weighted Recall | 0.79 |
| Weighted F1-score | 0.77 |
| Macro F1-score | 0.80 |
| Test images evaluated | 3,110 |
| Classes | 24 |

The model demonstrated strong performance on several classes, while visually similar fruit varieties—particularly apple varieties—were more challenging to distinguish.

## Training Visualizations

Generated visualizations include:

- Training and validation accuracy
- Training and validation loss
- Confusion matrix

Figures are saved under `reports/figures/`.

## Sample Prediction

The trained model supports image-level inference, displaying:

- Predicted class
- Confidence score
- Top three predictions

## Project Structure

```text
fruit-classification-transfer-learning/
├── data/
├── models/
├── notebooks/
├── outputs/
├── reports/
│   └── figures/
├── src/
│   └── fruit_classifier/
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── predict.py
│       └── visualize.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Running the Project

### Train

```bash
python -m src.fruit_classifier.train
```

### Visualize Training

```bash
python -m src.fruit_classifier.visualize
```

### Evaluate

```bash
python -m src.fruit_classifier.evaluate
```

### Predict an Image

```bash
python -m src.fruit_classifier.predict path/to/image.jpg
```

## Limitations

- Training was performed using a limited number of batches per epoch to reduce computational requirements.
- Fine-tuning and model performance may vary with full-dataset training.
- Visually similar classes may be confused.
- Prediction confidence is not equivalent to guaranteed correctness.

## Technologies

Python · TensorFlow · Keras · VGG16 · NumPy · Matplotlib · Scikit-learn

## License

This project is licensed under the MIT License.
