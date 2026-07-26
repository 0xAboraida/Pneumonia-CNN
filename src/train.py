"""
Training pipeline for the PneumoNet model.

Handles model compilation, callback setup, training, and artifact saving.
Can be run as a standalone script: python -m src.train
"""

import json
import os

import numpy as np
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from tensorflow.keras.models import Sequential

from src.config import PathConfig, TrainingConfig
from src.model import PneumoNet
from src.data_loader import ImageLoader


class Trainer:
    """
    Orchestrates model compilation, training, and artifact saving.

    Usage:
        trainer = Trainer()
        model = trainer.build_and_compile()
        history = trainer.train(model, X_train, y_train, X_val, y_val)
        trainer.save(model, history)
    """

    def __init__(self, config: type = TrainingConfig):
        """
        Args:
            config: Training configuration class with hyperparameters.
        """
        self.config = config

    def build_and_compile(self) -> Sequential:
        """Build the PneumoNet model and compile it with configured settings."""
        model = PneumoNet().build()
        model.compile(
            optimizer=self.config.OPTIMIZER,
            loss=self.config.LOSS,
            metrics=self.config.METRICS,
        )
        print("✅ Model built and compiled successfully!")
        model.summary()
        return model

    def get_callbacks(self) -> list:
        """
        Create and return the list of Keras callbacks for training.

        Returns:
            List of [EarlyStopping, ReduceLROnPlateau, ModelCheckpoint].
        """
        early_stop = EarlyStopping(
            monitor=self.config.ES_MONITOR,
            patience=self.config.ES_PATIENCE,
            restore_best_weights=self.config.ES_RESTORE_BEST,
        )

        reduce_lr = ReduceLROnPlateau(
            monitor=self.config.LR_MONITOR,
            factor=self.config.LR_FACTOR,
            patience=self.config.LR_PATIENCE,
            min_lr=self.config.LR_MIN,
            verbose=1,
        )

        checkpoint = ModelCheckpoint(
            str(PathConfig.CHECKPOINT_PATH),
            monitor=self.config.ES_MONITOR,
            save_best_only=True,
            verbose=1,
        )

        return [early_stop, reduce_lr, checkpoint]

    def train(
        self,
        model: Sequential,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """
        Train the model.

        Args:
            model: Compiled Keras model.
            X_train, y_train: Training data.
            X_val, y_val: Validation data.

        Returns:
            Keras History object.
        """
        callbacks = self.get_callbacks()

        history = model.fit(
            X_train,
            y_train,
            batch_size=self.config.BATCH_SIZE,
            epochs=self.config.EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            shuffle=True,
        )

        print("✅ Training complete!")
        return history

    @staticmethod
    def save(model: Sequential, history) -> None:
        """
        Save the trained model and training history.

        Args:
            model: Trained Keras model.
            history: Keras History object from model.fit().
        """
        # Save model
        model.save(str(PathConfig.MODEL_PATH))
        model.save_weights(str(PathConfig.WEIGHTS_PATH))
        print(f"✅ Model saved to {PathConfig.MODEL_PATH}")

        # Save history
        os.makedirs(str(PathConfig.HISTORY_DIR), exist_ok=True)
        with open(str(PathConfig.HISTORY_PATH), "w") as f:
            json.dump(history.history, f, indent=2)
        print(f"✅ History saved to {PathConfig.HISTORY_PATH}")


# ─── CLI Entry Point ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PneumoNet — Training Pipeline")
    print("=" * 60)

    # Load data
    print("\n📂 Loading data...")
    data = ImageLoader.load_all_splits()

    # Build & compile
    print("\n🏗️  Building model...")
    trainer = Trainer()
    model = trainer.build_and_compile()

    # Train
    print("\n🚀 Starting training...")
    history = trainer.train(
        model,
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
    )

    # Save
    print("\n💾 Saving artifacts...")
    trainer.save(model, history)

    print("\n✅ Done!")
