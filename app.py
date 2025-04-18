# app.py

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Initialize FastAPI
app = FastAPI(
    title="Ad Performance Forecast API",
    description="Predict Impressions, Clicks, and Conversions for a new ad",
    version="1.0"
)

# 2. Load your trained pipelines at startup
rf_imp_pipe  = joblib.load("best_rf_impressions.pkl")
rf_clk_pipe  = joblib.load("best_rf_clicks.pkl")
rf_conv_pipe = joblib.load("best_rf_conversions.pkl")

# 3. Define the request schema
class AdFeatures(BaseModel):
    Platform: str = Field(..., example="Facebook")
    Budget: float = Field(..., example=500.0)
    Ad_Type: str = Field(..., example="Image")
    Target_Age: str = Field(..., example="18-25")
    Target_Gender: str = Field(..., example="Male")
    CTR: float = Field(..., example=0.08)
    ConversionRate: float = Field(..., example=0.05)
    Impressions_7d_avg: float = Field(..., example=12000.0)
    Clicks_7d_avg: float = Field(..., example=800.0)
    Conversions_7d_avg: float = Field(..., example=50.0)

# 4. Define the response schema
class Forecast(BaseModel):
    Impressions: float
    Clicks: float
    Conversions: float

# 5. Prediction endpoint
@app.post("/predict", response_model=Forecast)
def predict_ad_performance(features: AdFeatures):
    """
    Returns forecasted Impressions, Clicks, and Conversions for a given ad.
    """
    # 5.1 Convert incoming data to a single-row DataFrame
    df = pd.DataFrame([features.dict()])

    try:
        # 5.2 Produce predictions using each pipeline
        pred_imp  = rf_imp_pipe.predict(df)[0]
        pred_clk  = rf_clk_pipe.predict(df)[0]
        pred_conv = rf_conv_pipe.predict(df)[0]

        return Forecast(
            Impressions = float(pred_imp),
            Clicks      = float(pred_clk),
            Conversions = float(pred_conv)
        )

    except Exception as e:
        # If anything goes wrong, return a 400 with the error message
        raise HTTPException(status_code=400, detail=str(e))
