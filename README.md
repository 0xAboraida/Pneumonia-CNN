# 🫁 PneumoNet — Pneumonia Detection from Chest X-rays

A deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a Convolutional Neural Network (CNN). Built with TensorFlow/Keras and deployed via FastAPI.

## 📊 Model Performance

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | 98.29% | 95.90% |
| Loss | 0.0537 | 0.1025 |

## 🏗️ Project Structure

```
PneumoNet/
├── src/                          # Source code package
│   ├── __init__.py               # Package init
│   ├── config.py                 # Centralized configuration
│   ├── data_loader.py            # Data pipeline (split, resize, load, balance)
│   ├── model.py                  # CNN architecture (PneumoNet)
│   ├── train.py                  # Training pipeline
│   ├── evaluate.py               # Evaluation & visualization
│   └── inference.py              # Single-image prediction
├── app.py                        # FastAPI web deployment
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── PneumoNet_Final.keras         # Pre-trained model (~29 MB)
├── PneumoNet_Final_Best.weights.h5
├── history/                      # Training history
│   └── PneumoNet_Final_History.json
├── Chest_X-ray_Dataset/          # Raw data (NORMAL / PNEUMONIA)
├── Chest_X-ray_Split/            # Train/Val/Test split
├── Chest_X-ray_Split_Resized/    # Resized to 150x150
└── chest_X-ray_balanced/         # Balanced training set
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000/docs** for the interactive Swagger UI.

### 3. Make a Prediction (CLI)

```bash
python -m src.inference path/to/chest_xray.jpg
```

### 4. Train the Model (from scratch)

```bash
python -m src.train
```

## 🏛️ Architecture

The PneumoNet CNN follows this architecture:

```
Input (150×150×1)
  → [Conv2D(32) → BatchNorm → Conv2D(32) → BatchNorm → MaxPool → Dropout(0.25)]
  → [Conv2D(64) → BatchNorm → Conv2D(64) → BatchNorm → MaxPool → Dropout(0.25)]
  → [Conv2D(128) → BatchNorm → Conv2D(128) → BatchNorm → MaxPool → Dropout(0.25)]
  → Flatten → Dense(256, ReLU) → BatchNorm → Dropout(0.5)
  → Dense(1, Sigmoid)
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Upload X-ray image for prediction |
| `GET` | `/docs` | Swagger UI documentation |

### Example Response

```json
{
  "filename": "chest_xray_001.jpg",
  "label": "PNEUMONIA",
  "confidence": 0.9712,
  "class_index": 1
}
```

## 🧩 Module Overview

| Module | Class | Description |
|--------|-------|-------------|
| `config.py` | `PathConfig`, `ImageConfig`, `DataConfig`, `TrainingConfig` | All configuration |
| `data_loader.py` | `DataSplitter`, `ImageResizer`, `ImageLoader`, `DataBalancer`, `ImageCounter` | Data pipeline |
| `model.py` | `PneumoNet` | CNN architecture |
| `train.py` | `Trainer` | Training pipeline |
| `evaluate.py` | `Evaluator`, `HistoryPlotter` | Evaluation & visualization |
| `inference.py` | `Predictor` | Single-image prediction |

## 📝 License

This project is for educational purposes.
