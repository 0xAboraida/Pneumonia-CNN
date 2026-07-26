"""
Evaluation utilities for the PneumoNet model.

Provides metrics, classification reports, confusion matrices,
and training history visualization.
"""

import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Model

from src.config import PathConfig, DataConfig


class Evaluator:
    """
    Evaluates a trained PneumoNet model on given data.

    Usage:
        evaluator = Evaluator(model)
        evaluator.evaluate(X_test, y_test, "Test")
        evaluator.classification_report(X_test, y_test)
        evaluator.confusion_matrix(X_test, y_test)
    """

    def __init__(self, model: Model):
        """
        Args:
            model: A trained Keras model.
        """
        self.model = model

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str = "Test",
    ) -> tuple[float, float]:
        """
        Print and return loss/accuracy for the given dataset.

        Returns:
            (loss, accuracy)
        """
        loss, accuracy = self.model.evaluate(X, y, verbose=0)
        print(f"\n📊 {dataset_name} Results:")
        print(f"   Loss:     {loss:.4f}")
        print(f"   Accuracy: {accuracy * 100:.2f}%")
        return loss, accuracy

    def classification_report(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> str:
        """
        Print and return the sklearn classification report.

        Returns:
            Classification report string.
        """
        y_pred = (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()
        report = classification_report(
            y,
            y_pred,
            target_names=DataConfig.LABELS,
        )
        print("\n📋 Classification Report:")
        print(report)
        return report

    def confusion_matrix(
        self,
        X: np.ndarray,
        y: np.ndarray,
        save_path: str | None = None,
    ) -> np.ndarray:
        """
        Plot and optionally save the confusion matrix.

        Returns:
            Confusion matrix as numpy array.
        """
        y_pred = (self.model.predict(X, verbose=0) > 0.5).astype(int).flatten()
        cm = confusion_matrix(y, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=DataConfig.LABELS,
            yticklabels=DataConfig.LABELS,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✅ Confusion matrix saved to {save_path}")

        plt.show()
        return cm


class HistoryPlotter:
    """
    Visualizes training history (loss and accuracy curves).

    Usage:
        plotter = HistoryPlotter()
        plotter.plot()  # loads from default history path
    """

    def __init__(self, history_path: str | None = None):
        """
        Args:
            history_path: Path to the JSON history file.
                          Defaults to PathConfig.HISTORY_PATH.
        """
        self.history_path = history_path or str(PathConfig.HISTORY_PATH)

    def load_history(self) -> dict:
        """Load training history from JSON file."""
        with open(self.history_path, "r") as f:
            return json.load(f)

    def plot(self, save_path: str | None = None) -> None:
        """
        Plot loss and accuracy curves from the training history.

        Args:
            save_path: If provided, saves the figure to this path.
        """
        history = self.load_history()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # ── Loss ──
        axes[0].plot(history["loss"], label="Train Loss", linewidth=2)
        axes[0].plot(history["val_loss"], label="Val Loss", linewidth=2)
        axes[0].set_title("Loss Over Epochs", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ── Accuracy ──
        axes[1].plot(history["accuracy"], label="Train Accuracy", linewidth=2)
        axes[1].plot(history["val_accuracy"], label="Val Accuracy", linewidth=2)
        axes[1].set_title("Accuracy Over Epochs", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✅ Training curves saved to {save_path}")

        plt.show()
