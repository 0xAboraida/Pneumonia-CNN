"""
Centralized configuration for the PneumoNet project.

All paths, hyperparameters, and constants are defined here
so that no hardcoded values appear in other modules.
"""

from pathlib import Path


class PathConfig:
    """All filesystem paths used across the project."""

    # Project root (parent of src/)
    ROOT_DIR = Path(__file__).resolve().parent.parent

    # Raw data
    DATASET_DIR = ROOT_DIR / "data" / "raw" / "Chest_X-ray_Dataset"

    # Processed data
    SPLIT_DIR = ROOT_DIR / "data" / "processed" / "Chest_X-ray_Split"
    RESIZED_DIR = ROOT_DIR / "data" / "processed" / "Chest_X-ray_Split_Resized"
    BALANCED_DIR = ROOT_DIR / "data" / "processed" / "chest_X-ray_balanced"

    # Model artifacts
    MODEL_PATH = ROOT_DIR / "models" / "PneumoNet_Final.keras"
    CHECKPOINT_PATH = ROOT_DIR / "models" / "PneumoNet_Final_checkpoint.keras"
    WEIGHTS_PATH = ROOT_DIR / "models" / "PneumoNet_Final_Best.weights.h5"
    HISTORY_DIR = ROOT_DIR / "history"
    HISTORY_PATH = HISTORY_DIR / "PneumoNet_Final_History.json"


class ImageConfig:
    """Image preprocessing settings."""

    IMG_SIZE = (150, 150)
    CHANNELS = 1  # Grayscale


class DataConfig:
    """Data splitting and balancing settings."""

    SPLIT_RATIOS = (0.7, 0.15, 0.15)
    SEED = 42
    LABELS = ["NORMAL", "PNEUMONIA"]
    IMAGE_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png")


class TrainingConfig:
    """Training hyperparameters."""

    BATCH_SIZE = 32
    EPOCHS = 15
    OPTIMIZER = "rmsprop"
    LOSS = "binary_crossentropy"
    METRICS = ["accuracy"]

    # ReduceLROnPlateau
    LR_MONITOR = "val_loss"
    LR_FACTOR = 0.4
    LR_PATIENCE = 2
    LR_MIN = 1e-6

    # EarlyStopping
    ES_MONITOR = "val_loss"
    ES_PATIENCE = 4
    ES_RESTORE_BEST = True
