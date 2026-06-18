import pandas as pd
import joblib
import os

def preprocess(df):
    # Langkah 1: Drop kolom "customerID" karena tidak relevan sebagai fitur
    df = df.drop(columns=["customerID"])
    
    # Langkah 2: Filter baris yang memiliki nilai TotalCharges berupa string spasi
    df = df[df["TotalCharges"] != " "]
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    
    # Langkah 3: Encode kolom target "Churn"
    y = df["Churn"].map({"Yes": 1, "No": 0})
    
    # Langkah 4: One-hot encode semua kolom bertipe object/string KECUALI kolom "Churn"
    X = df.drop(columns=["Churn"])
    X = pd.get_dummies(X, drop_first=True)
    
    # Langkah 5: Simpan daftar nama kolom X ke file models/feature_columns.pkl
    os.makedirs("models", exist_ok=True)
    joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")
    
    # Langkah 6: Return X dan y
    return X, y
