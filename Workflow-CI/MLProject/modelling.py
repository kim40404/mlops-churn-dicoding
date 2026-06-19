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

    # Langkah 8: Mulai MLflow run
    with mlflow.start_run():
        # a) Definisikan parameter
        n_estimators = 100
        random_state = 42
        test_size = 0.2

        # b) Log parameter ke MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", test_size)

        # c) Train model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)

        # d) Prediksi
        y_pred = model.predict(X_test)

        # e) Hitung metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall    = recall_score(y_test, y_pred, average="weighted")
        f1        = f1_score(y_test, y_pred, average="weighted")

        # f) Log semua metrics ke MLflow
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # g) Simpan model ke MLflow artifacts
        mlflow.sklearn.log_model(model, "random_forest_model")

        # h) Simpan model ke file lokal
        joblib.dump(model, "models/model.pkl")

        # i) Print semua hasil ke terminal
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print("Model saved to models/model.pkl")
        print("Feature columns saved to models/feature_columns.pkl")
