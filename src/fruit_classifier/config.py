from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

DATASET_DIR = DATA_DIR / "fruits-360-original-size"

TRAIN_DIR = DATASET_DIR / "Training"
VAL_DIR = DATASET_DIR / "Validation"
TEST_DIR = DATASET_DIR / "Test"

MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

IMAGE_SIZE = (64, 64)
IMAGE_CHANNELS = 3

INPUT_SHAPE = (
    IMAGE_SIZE[0],
    IMAGE_SIZE[1],
    IMAGE_CHANNELS,
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 16

INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5

STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 25
TEST_STEPS = 50

RANDOM_SEED = 42


# ============================================================
# DATA AUGMENTATION
# ============================================================

ROTATION_RANGE = 20
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.2
SHEAR_RANGE = 0.2
ZOOM_RANGE = 0.2

HORIZONTAL_FLIP = True
FILL_MODE = "nearest"


# ============================================================
# MODEL CONFIGURATION
# ============================================================

BASE_MODEL_NAME = "VGG16"
PRETRAINED_WEIGHTS = "imagenet"

FINE_TUNE_LAST_LAYERS = 5

DENSE_UNITS = 256
DROPOUT_RATE = 0.3


# ============================================================
# OPTIMIZATION CONFIGURATION
# ============================================================

INITIAL_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.00001

LR_REDUCTION_FACTOR = 0.2
LR_PATIENCE = 2
MIN_LEARNING_RATE = 1e-6

EARLY_STOPPING_PATIENCE = 5


# ============================================================
# MODEL OUTPUT
# ============================================================

MODEL_FILENAME = "fruit_classifier_vgg16.keras"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

INITIAL_MODEL_FILENAME = "fruit_classifier_initial.keras"
INITIAL_MODEL_PATH = MODEL_DIR / INITIAL_MODEL_FILENAME


# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================

ACCURACY_PLOT_PATH = FIGURES_DIR / "accuracy.png"
LOSS_PLOT_PATH = FIGURES_DIR / "loss.png"
PREDICTION_PLOT_PATH = FIGURES_DIR / "sample_predictions.png"


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)