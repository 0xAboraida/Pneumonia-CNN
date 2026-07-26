"""
FastAPI web application for PneumoNet — Pneumonia Detection API.

Provides REST endpoints for:
  - POST /predict  → Upload a chest X-ray image, get prediction
  - GET  /health   → Health check
  - GET  /         → API info

Run with:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.inference import Predictor
from src.config import DataConfig

# ─── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="PneumoNet API",
    description=(
        "🫁 **Pneumonia Detection from Chest X-rays**\n\n"
        "Upload a chest X-ray image and get a prediction of whether "
        "the image shows **NORMAL** lungs or signs of **PNEUMONIA**.\n\n"
        "Built with a CNN trained on the Chest X-ray dataset."
    ),
    version="1.0.0",
)

# Lazy-loaded predictor (model loads on first request)
predictor = Predictor()


# ─── Endpoints ────────────────────────────────────────────────


@app.get("/", tags=["Info"])
async def root():
    """API root — returns basic information about PneumoNet."""
    return {
        "name": "PneumoNet API",
        "version": "1.0.0",
        "description": "Pneumonia Detection from Chest X-rays using CNN",
        "endpoints": {
            "POST /predict": "Upload an X-ray image for prediction",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation (Swagger UI)",
        },
        "labels": DataConfig.LABELS,
    }


@app.get("/health", tags=["Info"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": predictor.model is not None}


@app.post("/predict", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image and receive a pneumonia prediction.

    **Accepted formats:** JPEG, JPG, PNG

    **Response:**
    - `label`: "NORMAL" or "PNEUMONIA"
    - `confidence`: Prediction confidence (0.0 to 1.0)
    - `class_index`: 0 for NORMAL, 1 for PNEUMONIA
    - `filename`: Original uploaded filename
    """
    # Validate file type
    allowed_types = {"image/jpeg", "image/jpg", "image/png"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. "
                   f"Accepted types: {', '.join(allowed_types)}",
        )

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Predict
        result = predictor.predict_from_bytes(image_bytes)

        return JSONResponse(
            content={
                "filename": file.filename,
                **result,
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
