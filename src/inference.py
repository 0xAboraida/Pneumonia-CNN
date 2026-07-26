"""
Inference module for single-image pneumonia prediction.

Loads a trained PneumoNet model and predicts whether
a chest X-ray shows NORMAL or PNEUMONIA.
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model, Model

from src.config import PathConfig, ImageConfig, DataConfig


class Predictor:
    """
    Loads a trained model and performs inference on chest X-ray images.

    Usage:
        predictor = Predictor()
        result = predictor.predict("path/to/xray.jpg")
        print(result)
        # {'label': 'PNEUMONIA', 'confidence': 0.97, 'class_index': 1}
    """

    def __init__(self, model_path: str | None = None):
        """
        Args:
            model_path: Path to the trained .keras model file.
                        Defaults to PathConfig.MODEL_PATH.
        """
        self.model_path = model_path or str(PathConfig.MODEL_PATH)
        self.model: Model | None = None
        self.target_size = ImageConfig.IMG_SIZE

    def load_model(self) -> Model:
        """Load the trained model from disk."""
        self.model = load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")
        return self.model

    def ensure_model_loaded(self) -> None:
        """Load model if not already loaded."""
        if self.model is None:
            self.load_model()

    @staticmethod
    def preprocess_image(
        image_path: str,
        target_size: tuple = ImageConfig.IMG_SIZE,
    ) -> np.ndarray:
        """
        Read and preprocess a single image for prediction.

        Args:
            image_path: Path to the chest X-ray image.
            target_size: Target (width, height) for resizing.

        Returns:
            Preprocessed image as numpy array with shape (1, H, W, 1).
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = img[np.newaxis, ..., np.newaxis]  # Add batch and channel dims
        return img

    @staticmethod
    def preprocess_image_bytes(
        image_bytes: bytes,
        target_size: tuple = ImageConfig.IMG_SIZE,
    ) -> np.ndarray:
        """
        Preprocess an image from raw bytes (for API uploads).

        Args:
            image_bytes: Raw bytes of the uploaded image.
            target_size: Target (width, height) for resizing.

        Returns:
            Preprocessed image as numpy array with shape (1, H, W, 1).
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image from bytes")

        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = img[np.newaxis, ..., np.newaxis]
        return img

    def predict(self, image_path: str) -> dict:
        """
        Predict the class of a chest X-ray image.

        Args:
            image_path: Path to the chest X-ray image.

        Returns:
            dict with keys: label, confidence, class_index
        """
        self.ensure_model_loaded()

        img = self.preprocess_image(image_path, self.target_size)
        prob = float(self.model.predict(img, verbose=0)[0][0])
        class_index = int(prob > 0.5)
        label = DataConfig.LABELS[class_index]
        confidence = prob if class_index == 1 else 1 - prob

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "class_index": class_index,
        }

    def predict_from_bytes(self, image_bytes: bytes) -> dict:
        """
        Predict from raw image bytes (used by the API).

        Args:
            image_bytes: Raw bytes of the uploaded image.

        Returns:
            dict with keys: label, confidence, class_index
        """
        self.ensure_model_loaded()

        img = self.preprocess_image_bytes(image_bytes, self.target_size)
        prob = float(self.model.predict(img, verbose=0)[0][0])
        class_index = int(prob > 0.5)
        label = DataConfig.LABELS[class_index]
        confidence = prob if class_index == 1 else 1 - prob

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "class_index": class_index,
        }


# ─── CLI Entry Point ─────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.inference <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predictor = Predictor()
    result = predictor.predict(image_path)

    print(f"\n🔍 Prediction Result:")
    print(f"   Label:      {result['label']}")
    print(f"   Confidence: {result['confidence'] * 100:.1f}%")
    print(f"   Class:      {result['class_index']}")
