import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
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
    mlflow.autolog()

    # Langkah 8: Mulai MLflow run
    with mlflow.start_run():
        
        # --- SMOTE ---
        print("Menerapkan SMOTE untuk menyeimbangkan kelas...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        # --- XGBoost & Tuning ---
        print("Memulai Hyperparameter Tuning dengan XGBoost...")
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
        }
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train_res, y_train_res)
        
        model = grid_search.best_estimator_
        print(f"Parameter Terbaik: {grid_search.best_params_}")

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

        # --- SURVIVAL ANALYSIS ---
        from lifelines import CoxPHFitter
        df_cox = X_train.copy()
        df_cox["Churn"] = y_train
        
        # Penalizer 0.1 untuk mencegah error collinearity (Ponytail rule: robust edge-case)
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(df_cox, duration_col="tenure", event_col="Churn")
        joblib.dump(cph, "models/survival_model.pkl")
        print("\nSurvival Model trained successfully.")

        # Print hasil
        for k, v in metrics.items():
            print(f"{k.capitalize():<10} : {v:.4f}")
            
        print("\nModel saved to models/model.pkl")
        print("Survival model saved to models/survival_model.pkl")
        print("Feature columns saved to models/feature_columns.pkl")
