import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
LOGS_DIR = RESULTS_DIR / "logs"

# Reproducibility
RANDOM_SEED = 42

# Image Preprocessing & Model Input Specifications
# EfficientNet-B0 default input resolution is 224x224
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_WORKERS = 2

# Training Hyperparameters
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 10

# Data Splits Ratio
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# WHO Anemia Thresholds (Adults >= 15 years, Hb in g/dL)
WHO_MALE_THRESHOLD = 13.0
WHO_FEMALE_THRESHOLD = 12.0
WHO_CHILD_THRESHOLD = 11.0  # For pediatric datasets like CP-AnemiC

# Supported Modalities
MODALITIES = ["conjunctiva", "cp_anemic", "palm", "fingernail"]
