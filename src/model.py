"""
CNN model architecture for Pneumonia detection.

Defines the PneumoNet model — a Sequential CNN with
Conv2D → BatchNorm → MaxPool → Dropout blocks.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPool2D,
)

from src.config import ImageConfig


class PneumoNet:
    """
    Builds the PneumoNet CNN architecture for binary classification
    of chest X-ray images (NORMAL vs PNEUMONIA).

    Architecture:
        3 × [Conv2D → BatchNorm → Conv2D → BatchNorm → MaxPool → Dropout(0.25)]
        → Flatten → Dense(256) → BatchNorm → Dropout(0.5) → Dense(1, sigmoid)
    """

    def __init__(self, input_shape: tuple | None = None):
        """
        Args:
            input_shape: Shape of input images. Defaults to (150, 150, 1).
        """
        if input_shape is None:
            input_shape = (*ImageConfig.IMG_SIZE, ImageConfig.CHANNELS)
        self.input_shape = input_shape

    def build(self) -> Sequential:
        """
        Build and return the PneumoNet Keras Sequential model.

        Returns:
            A compiled-ready Keras Sequential model.
        """
        model = Sequential(name="PneumoNet")

        # Input layer
        model.add(Input(shape=self.input_shape))

        # ── Block 1 ──────────────────────────────
        model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # ── Block 2 ──────────────────────────────
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # ── Block 3 ──────────────────────────────
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        model.add(BatchNormalization())
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # ── Classifier Head ──────────────────────
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        return model
