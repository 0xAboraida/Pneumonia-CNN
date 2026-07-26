"""
Data loading, splitting, resizing, and balancing utilities.

Handles the full data pipeline from raw images to training-ready NumPy arrays.
"""

import os
import shutil
import random
import glob as gb
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.config import PathConfig, ImageConfig, DataConfig


class DataSplitter:
    """
    Splits a flat dataset (NORMAL/PNEUMONIA folders)
    into train/val/test directories.
    """

    def __init__(
        self,
        input_dir: str | Path = PathConfig.DATASET_DIR,
        output_dir: str | Path = PathConfig.SPLIT_DIR,
        split_ratios: tuple = DataConfig.SPLIT_RATIOS,
        seed: int = DataConfig.SEED,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.split_ratios = split_ratios
        self.seed = seed

    def split(self) -> None:
        """
        Split images within label folders (NORMAL, PNEUMONIA)
        into train/val/test directories.
        """
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        # Create folder structure
        for split in ["train", "val", "test"]:
            for label in DataConfig.LABELS:
                (self.output_dir / split / label).mkdir(parents=True, exist_ok=True)

        for label in DataConfig.LABELS:
            images = list((self.input_dir / label).glob("*.*"))

            # Train + temp (val+test)
            train_imgs, temp_imgs = train_test_split(
                images, test_size=1 - self.split_ratios[0], random_state=self.seed
            )

            # Val + Test
            val_ratio = self.split_ratios[1] / (self.split_ratios[1] + self.split_ratios[2])
            val_imgs, test_imgs = train_test_split(
                temp_imgs, test_size=1 - val_ratio, random_state=self.seed
            )

            for img in train_imgs:
                shutil.copy(img, self.output_dir / "train" / label / img.name)
            for img in val_imgs:
                shutil.copy(img, self.output_dir / "val" / label / img.name)
            for img in test_imgs:
                shutil.copy(img, self.output_dir / "test" / label / img.name)

        print("✅ Images have been successfully split!")


class ImageResizer:
    """
    Resizes all images in a directory tree to a target size.

    Uses INTER_AREA for downscaling and INTER_CUBIC for upscaling.
    Images are loaded in grayscale (X-ray style).
    """

    def __init__(
        self,
        size: tuple = ImageConfig.IMG_SIZE,
    ):
        self.size = size

    def resize(self, input_dir: str | Path, output_dir: str | Path) -> None:
        """Resize all images from input_dir and save to output_dir."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)

        for label in os.listdir(input_dir):
            folder_in = input_dir / label
            folder_out = output_dir / label
            folder_out.mkdir(parents=True, exist_ok=True)

            for img_path in tqdm(
                list(folder_in.glob("*.*")),
                desc=f"Resizing {input_dir.name}/{label}",
            ):
                try:
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    if img.shape[0] > self.size[0] or img.shape[1] > self.size[1]:
                        interpolation = cv2.INTER_AREA  # best for shrinking
                    else:
                        interpolation = cv2.INTER_CUBIC  # best for enlarging

                    resized = cv2.resize(img, self.size, interpolation=interpolation)
                    cv2.imwrite(str(folder_out / img_path.name), resized)

                except Exception as e:
                    print(f"Error in {img_path}: {e}")

    def resize_all_splits(
        self,
        split_dir: str | Path = PathConfig.SPLIT_DIR,
        output_dir: str | Path = PathConfig.RESIZED_DIR,
    ) -> None:
        """Resize train/val/test splits in one call."""
        split_dir = Path(split_dir)
        output_dir = Path(output_dir)

        for split_name in ["train", "val", "test"]:
            self.resize(
                input_dir=split_dir / split_name,
                output_dir=output_dir / split_name,
            )


class ImageLoader:
    """
    Loads preprocessed images from disk into NumPy arrays
    ready for model training/evaluation.
    """

    @staticmethod
    def load(folder_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """
        Load images from a folder (train, val, or test) and return X, y.

        Returns:
            X: numpy array of shape (N, H, W, 1)
            y: numpy array of labels (0=NORMAL, 1=PNEUMONIA)
        """
        folder_path = Path(folder_path)
        X, y = [], []

        for label, folder_name in enumerate(sorted(os.listdir(folder_path))):
            folder = folder_path / folder_name
            files = [
                f
                for f in os.listdir(folder)
                if f.lower().endswith((".jpeg", ".jpg", ".png"))
            ]

            for file in tqdm(files, desc=f"Loading {folder_name} images"):
                img_path = folder / file
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    X.append(img)
                    y.append(label)

        X = np.array(X)[..., np.newaxis]  # Add channel dimension
        y = np.array(y)

        return X, y

    @staticmethod
    def load_all_splits(
        resized_dir: str | Path = PathConfig.RESIZED_DIR,
    ) -> dict:
        """
        Load all three splits at once.

        Returns:
            dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
        """
        resized_dir = Path(resized_dir)
        loader = ImageLoader()

        X_train, y_train = loader.load(resized_dir / "train")
        X_val, y_val = loader.load(resized_dir / "val")
        X_test, y_test = loader.load(resized_dir / "test")

        print("✅ Loading complete!\n\nDataset shapes:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val:   {X_val.shape},   y_val:   {y_val.shape}")
        print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
        }


class DataBalancer:
    """
    Balances an imbalanced dataset by oversampling the minority class.

    Copies random samples from the minority class until both classes
    have equal counts.
    """

    @staticmethod
    def balance(
        source_dir: str | Path = PathConfig.RESIZED_DIR,
        output_dir: str | Path = PathConfig.BALANCED_DIR,
        seed: int = DataConfig.SEED,
    ) -> None:
        """
        Balance the training set by oversampling the minority class.

        Copies from source_dir/train to output_dir/train_balanced,
        and links val/test as-is.
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        random.seed(seed)

        train_source = source_dir / "train"

        # Count images per label
        counts = {}
        for label in DataConfig.LABELS:
            files = list((train_source / label).glob("*.*"))
            counts[label] = files

        max_count = max(len(v) for v in counts.values())

        # Create balanced training directory
        balanced_train = output_dir / "train_balanced"
        for label in DataConfig.LABELS:
            dest = balanced_train / label
            dest.mkdir(parents=True, exist_ok=True)

            files = counts[label]
            # Copy all original files
            for f in files:
                shutil.copy(f, dest / f.name)

            # Oversample if needed
            if len(files) < max_count:
                deficit = max_count - len(files)
                extra = random.choices(files, k=deficit)
                for i, f in enumerate(extra):
                    new_name = f"{f.stem}_dup{i}{f.suffix}"
                    shutil.copy(f, dest / new_name)

        # Copy val and test as-is
        for split in ["val", "test"]:
            src = source_dir / split
            dst = output_dir / split
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

        print("✅ Dataset balanced successfully!")


class ImageCounter:
    """Utility to count and display images per class in a directory."""

    @staticmethod
    def count(base_path: str | Path, dataset_name: str) -> None:
        """Print image counts for each label folder."""
        base_path = Path(base_path)
        print(f"\n===== {dataset_name} data =====")
        for folder in sorted(os.listdir(base_path)):
            files = []
            for ext in DataConfig.IMAGE_EXTENSIONS:
                files.extend(gb.glob(str(base_path / folder / ext)))
            print(f"  Found {len(files)} images in folder {folder}")
