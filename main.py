"""
Customer Churn Prediction API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import logging

from src.schemas import CustomerFeatures, PredictionResponse, BatchRequest, BatchResponse, HealthResponse
from src.predictor import ChurnPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = ChurnPredictor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model on startup...")
    predictor.load_model()
    logger.info("Model loaded successfully.")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict whether a customer will churn based on account & service usage features.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Returns API and model health status."""
    return {
        "status": "ok",
        "model_loaded": predictor.is_loaded(),
        "model_version": predictor.model_version,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Returns:
    - **churn**: Boolean prediction (True = will churn)
    - **churn_probability**: Probability of churn (0–1)
    - **risk_level**: LOW / MEDIUM / HIGH
    - **top_risk_factors**: Top features driving the prediction
    """
    if not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first via /train.")
    try:
        return predictor.predict(customer)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """
    Predict churn for multiple customers in one request (max 500).
    """
    if not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(request.customers) > 500:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 500.")
    try:
        return predictor.predict_batch(request.customers)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", tags=["Model Management"])
def train(data_path: str = "data/Telco-Customer-Churn.csv"):
    """
    Retrain the model on the given CSV path. Saves model artifacts to disk.
    """
    try:
        metrics = predictor.train(data_path)
        return {"message": "Model trained successfully.", "metrics": metrics}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Data file not found: {data_path}")
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model Management"])
def model_info():
    """Returns model metadata, feature importances, and training metrics."""
    if not predictor.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return predictor.get_model_info()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
