from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import joblib
import pandas as pd
import numpy as np
import time
import os

# Saat startup (di luar endpoint, saat file diload):
# - Load model dari "models/model.pkl"
# - Load feature columns dari "models/feature_columns.pkl"
try:
    model = joblib.load("models/model.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
except Exception as e:
    model = None
    feature_columns = None
    print(f"Warning: Model or feature columns not found. {e}")

app = FastAPI()

# Tambahkan CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Latency of prediction requests in seconds"
)

# Pydantic input schema
class PredictInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(input_data: PredictInput):
    start_time = time.time()
    
    if model is None or feature_columns is None:
        raise HTTPException(status_code=500, detail="Model or feature columns are not loaded.")

    # Convert input ke DataFrame 1 baris
    input_dict = input_data.model_dump()
    df = pd.DataFrame([input_dict])
    
    # Terapkan pd.get_dummies(drop_first=True) pada kolom object
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Reindex kolom DataFrame menggunakan feature_columns
    df_reindexed = df_encoded.reindex(columns=feature_columns, fill_value=0)
    
    # Prediksi
    prediction = model.predict(df_reindexed)[0]
    probabilities = model.predict_proba(df_reindexed)[0]
    
    # Tambah PREDICTION_COUNTER
    PREDICTION_COUNTER.inc()
    
    # Hitung latency dan observe
    latency = time.time() - start_time
    PREDICTION_LATENCY.observe(latency)
    
    return {
        "prediction": int(prediction),
        "probability_churn": float(probabilities[1]),
        "probability_no_churn": float(probabilities[0])
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
