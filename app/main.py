from fastapi import FastAPI, HTTPException
import numpy as np
import logging

from app.schemas import PredictRequest, PredictResponse
from app.model_loader import load_model
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger("ml-api")

app = FastAPI(title="ML API Starter", version="0.2.0")

model = None


@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load model.")
        raise e


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

        return PredictResponse(prediction=pred, probability_class_1=proba)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed.")
        raise HTTPException(status_code=500, detail="Prediction failed") from e

#         return {
#     "prediction": pred,
#     "probability_class_1": proba,
#     "model_version": "0.1.0"
# }