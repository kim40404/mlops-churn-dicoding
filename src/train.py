import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys

# Langkah 2: Tambahkan sys.path agar import preprocess bisa jalan
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess

# Langkah 3: Pastikan folder models/ ada
os.makedirs("models", exist_ok=True)

if __name__ == "__main__":
    # Langkah 4: Load dataset
    df = pd.read_csv("data/telco_churn.csv")

    # Langkah 5: Panggil fungsi preprocess(df)
    X, y = preprocess(df)

    # Langkah 6: Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Langkah 7: Set MLflow tracking uri and experiment
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("churn-prediction-experiment")

    # Ponytail rule: The best code is the code never written. Let MLflow do the heavy lifting.
    mlflow.sklearn.autolog()

    # Langkah 8: Mulai MLflow run
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prediksi & Hitung metrics
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average="weighted"),
            "recall": recall_score(y_test, y_pred, average="weighted"),
            "f1_score": f1_score(y_test, y_pred, average="weighted")
        }
        
        # Log metrics in one line
        mlflow.log_metrics(metrics)

        # Simpan model ke file lokal untuk dipakai FastAPI
        joblib.dump(model, "models/model.pkl")

        # Print hasil
        for k, v in metrics.items():
            print(f"{k.capitalize():<10} : {v:.4f}")
            
        print("\nModel saved to models/model.pkl")
        print("Feature columns saved to models/feature_columns.pkl")
