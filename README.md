# MLOps Churn Prediction Project (Dicoding)

## 📌 Overview
An end-to-end Machine Learning Operations (MLOps) pipeline for predicting Telecommunications Customer Churn. Built as a submission for the Dicoding "Membangun Sistem Machine Learning" course.

## 💼 Business Value (Why this matters to HR & Investors)
- **Revenue Protection:** Customer churn costs telecommunication companies millions annually. This AI model predicts *which* customers are likely to cancel their subscriptions, allowing businesses to offer targeted promotions and retain them.
- **Production-Ready Engineering:** Unlike typical Data Science projects that end in a Jupyter Notebook, this project demonstrates **industry-level deployment**. The model is wrapped in a scalable REST API (FastAPI), containerized (Docker), and monitored in real-time (Prometheus/Grafana). This proves the ability to deliver tangible business tools, not just theoretical models.

## 🛠️ Technology Stack
- **Language**: Python 3.10+
- **Model**: Scikit-learn (Random Forest Classifier)
- **Experiment Tracking**: MLflow (autolog)
- **Model Serving**: FastAPI
- **Frontend**: Custom Glassmorphism UI (HTML/CSS/JS) served via FastAPI
- **Containerization**: Docker & Docker Compose
- **Monitoring**: Prometheus & Grafana
- **Data Versioning**: DVC
- **CI/CD**: GitHub Actions

---

## 🚀 How to Run the Project Locally

Follow these step-by-step instructions to run the complete MLOps pipeline on your local machine.

### 1. Setup Environment
First, install the required Python dependencies:
```bash
pip install -r requirements.txt
```
Make sure you have the dataset placed at `data/telco_churn.csv`.

### 2. Train the Model (Experiment Tracking)
Run the training script. This script will preprocess the data, train a Random Forest model, and log all metrics (Accuracy, F1, etc.) into MLflow.
```bash
python src/train.py
```
*(This will automatically generate the model artifacts in the `models/` directory)*

### 3. View MLflow UI (Metrics & Parameters)
To view the recorded experiments, open the MLflow UI. 
**Note for Windows users:** You must set the environment variable first to allow local file tracking.
Run this in your PowerShell terminal:
```powershell
$env:MLFLOW_ALLOW_FILE_STORE="true"
python -m mlflow ui
```
Then open your browser to: [http://localhost:5000](http://localhost:5000)

### 4. Serve API and Start Monitoring (Docker)
Ensure Docker Desktop is running on your machine. We will spin up the FastAPI server, Prometheus, and Grafana simultaneously.
```bash
docker-compose up --build -d
```

### 5. Generate Traffic (Inference)
To see the monitoring graphs move, generate some dummy traffic by hitting the API. You can run the provided inference script:
```bash
python "SUBMISSION-MSML\Monitor dan Logging\7.inference.py"
```

### 6. Access the Dashboards
Once Docker is running, you can access the following services in your browser:
- **API Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Web UI (Investor Demo):** [http://localhost:8000](http://localhost:8000)
- **Grafana Dashboard:** [http://localhost:3000](http://localhost:3000) (Login: `admin` / `admin`), password: `admin`)*
- **Prometheus (Metrics Scraper)**: [http://localhost:9090](http://localhost:9090)
