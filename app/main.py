from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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
    survival_model = joblib.load("models/survival_model.pkl")
except Exception as e:
    model = None
    feature_columns = None
    survival_model = None
    print(f"Warning: Models not found. {e}")

app = FastAPI()

# Tambahkan CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup UI (Ponytail rule: reuse FastAPI, no new dependency)
os.makedirs("app/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("app/static/index.html")


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

    # Convert input to dictionary
    input_dict = input_data.model_dump()
    
    # Ponytail rule: pd.get_dummies on a 1-row DataFrame drops categories if drop_first=True. 
    # We map directly to feature_columns for a robust, bug-free O(1) mapping.
    processed_row = {col: 0 for col in feature_columns}
    
    for key, value in input_dict.items():
        if key in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
            processed_row[key] = value
        else:
            dummy_name = f"{key}_{value}"
            if dummy_name in processed_row:
                processed_row[dummy_name] = 1
                
    df_reindexed = pd.DataFrame([processed_row])
    
    # Prediksi Churn (Random Forest)
    prediction = model.predict(df_reindexed)[0]
    probabilities = model.predict_proba(df_reindexed)[0]
    
    # Survival Analysis & LTV Prediction
    remaining_tenure = 0
    ltv = 0.0
    if survival_model is not None:
        try:
            expected_total_tenure = survival_model.predict_expectation(df_reindexed).iloc[0]
            current_tenure = input_dict.get('tenure', 0)
            monthly_charges = input_dict.get('MonthlyCharges', 0)
            
            remaining_tenure = max(0, expected_total_tenure - current_tenure)
            ltv = remaining_tenure * monthly_charges
        except Exception as e:
            print(f"Survival prediction failed: {e}")
    
    # Tambah PREDICTION_COUNTER
    PREDICTION_COUNTER.inc()
    
    # Hitung latency dan observe
    latency = time.time() - start_time
    PREDICTION_LATENCY.observe(latency)
    
    return {
        "prediction": int(prediction),
        "probability_churn": float(probabilities[1]),
        "probability_no_churn": float(probabilities[0]),
        "expected_remaining_months": round(float(remaining_tenure), 1),
        "estimated_ltv_loss": round(float(ltv), 2)
    }

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
