from fastapi import FastAPI, HTTPException
import numpy as np
import logging
import os
import time

from fastapi import Request
from app.schemas import PredictRequest, PredictResponse
from app.model_loader import load_model
from app.logging_config import setup_logging
from contextlib import asynccontextmanager


MODEL_VERSION = os.getenv("MODEL_VERSION", "0.0.0")
APP_ENV = os.getenv("APP_ENV", "development")




setup_logging()
logger = logging.getLogger("ml-api")

app = FastAPI(title="ML API Starter", version="0.2.0")

model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model()
    logger.info("Model loaded successfully.")
    yield

app = FastAPI(
    title="ML API Starter",
    version="0.2.0",
    lifespan=lifespan
)  

# @app.on_event("startup")
# def startup_event():
#     global model
#     try:
#         model = load_model()
#         logger.info("Model loaded successfully.")
#     except Exception as e:
#         logger.exception("Failed to load model.")
#         raise e


@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)

    logger.info(
    f"request_complete path={request.url.path} method={request.method} "
    f"status={response.status_code} duration_ms={duration}"
            )

    return response

@app.get("/")
def root():
    return {
        "message": "ML API Starter is live",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        X = np.array(payload.features, dtype=float).reshape(1, -1)

        # Safety: confirm expected feature count
        if X.shape[1] != 30:
            raise HTTPException(status_code=400, detail="Expected 30 features")

        proba = float(model.predict_proba(X)[0, 1])
        pred = int(model.predict(X)[0])

        logger.info(
            "Prediction made | pred=%s proba=%.4f", pred, proba
        )

        return PredictResponse(
            prediction=pred,
            probability_class_1=proba,
            model_version=MODEL_VERSION,
            environment=APP_ENV,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed") from e
    
    


