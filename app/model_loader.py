from pathlib import Path
import joblib

MODEL_PATH = Path("model") / "model.joblib"

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run: python model/train.py"
        )
    return joblib.load(MODEL_PATH)