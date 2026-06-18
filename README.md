# MLOps Churn Dicoding

## Overview
Sistem Machine Learning end-to-end untuk prediksi churn pelanggan
menggunakan prinsip MLOps Level 1.

## Tech Stack
- Python 3.10
- Scikit-learn (Random Forest)
- MLflow (experiment tracking)
- FastAPI (model serving)
- Docker & Docker Compose
- Prometheus & Grafana (monitoring)
- GitHub Actions (CI)
- DVC (data versioning)

## Cara Setup
1. Clone repo
2. pip install -r requirements.txt
3. Taruh dataset di data/telco_churn.csv
4. dvc init

## Cara Run Training
python src/train.py

## Cara Buka MLflow UI
mlflow ui
Buka: http://localhost:5000

## Cara Run Docker
docker-compose up --build
- API:        http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana:    http://localhost:3000 (admin/admin)

## Submission Checklist
- [x] Eksperimen dataset dengan MLflow
- [x] Model Machine Learning (Random Forest)
- [x] Workflow CI (GitHub Actions)
- [x] Sistem Monitoring (Prometheus + Grafana)
