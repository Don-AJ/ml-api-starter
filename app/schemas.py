from pydantic import BaseModel, Field, field_validator

N_FEATURES = 30  # defensive engineering


class PredictRequest(BaseModel):
    # Exact number of features required
    features: list[float] = Field(..., min_length=N_FEATURES, max_length=N_FEATURES)

    @field_validator("features")
    @classmethod
    def no_nan_or_inf(cls, v: list[float]):
        for i, x in enumerate(v):
            if x != x:  # NaN check
                raise ValueError(f"features[{i}] is NaN")
            if x in (float("inf"), float("-inf")):
                raise ValueError(f"features[{i}] is inf/-inf")
        return v


class PredictResponse(BaseModel):
    prediction: int
    probability_class_1: float
    model_version: str
    environment: str
