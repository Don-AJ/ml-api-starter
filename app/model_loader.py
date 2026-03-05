from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model" / "model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)
