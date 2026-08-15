<div align="center">

🌍 Read this in: [English](README.md) | [Bahasa Indonesia](README.id.md)

<br>
  
# 🚀 Enterprise MLOps: Telecommunications Churn & LTV Prediction
*Sebuah pipeline Machine Learning Operations (MLOps) yang dapat diskalakan dari ujung ke ujung (end-to-end) dengan Web Dashboard bergaya Quiet Luxury.*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.5-FF9900?style=for-the-badge&logo=xgboost)](https://xgboost.ai/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?style=for-the-badge&logo=mlflow)](https://mlflow.org/)
[![Grafana](https://img.shields.io/badge/Grafana-Monitoring-F46800?style=for-the-badge&logo=grafana)](https://grafana.com/)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?style=for-the-badge&logo=github-actions)](https://github.com/features/actions)

</div>

---

## 📑 Daftar Isi
- [Ringkasan Eksekutif](#-ringkasan-eksekutif)
- [Alur Arsitektur Sistem](#-alur-arsitektur-sistem)
- [Visual Showcase & Kapabilitas](#-visual-showcase--kapabilitas)
- [Alur Kerja MLOps Step-by-Step](#-alur-kerja-mlops-step-by-step)
- [Panduan Memulai Cepat](#-panduan-memulai-cepat)
- [Struktur Direktori Proyek](#-struktur-direktori-proyek)

---

## 💼 Ringkasan Eksekutif

Kehilangan pelanggan (*customer churn*) merugikan industri telekomunikasi miliaran dolar setiap tahunnya. Proyek ini direkayasa sebagai **Solusi Enterprise Siap Produksi** yang ditargetkan langsung pada *Revenue Protection* (Perlindungan Pendapatan) dan Mitigasi Risiko.

Dengan mengintegrasikan **Algoritma XGBoost** yang disetel secara presisi bersama **SMOTE (Synthetic Minority Over-sampling Technique)**, mesin AI ini secara dinamis menghitung Probabilitas Churn dan memprediksi **Estimasi Kerugian Pendapatan (LTV)** secara seketika (*real-time*). Seluruh infrastruktur dikontainerisasi menggunakan Docker dan dipantau menggunakan telemetri standar industri (Prometheus & Grafana), menjadikannya prototipe SaaS yang sangat tangguh, mudah diskalakan, dan siap dipresentasikan kepada investor.

---

## 🏗️ Alur Arsitektur Sistem

Pipeline ini sangat terpisah (*decoupled*), memastikan bahwa eksperimen *Data Science* terisolasi dari proses penyajian API dan Telemetri.

```mermaid
graph TD
    %% Styling configurations for an Enterprise "Quiet Luxury" look
    classDef dataBox fill:#1e293b,stroke:#334155,stroke-width:2px,color:#f8fafc,rx:8px,ry:8px;
    classDef aiEngine fill:#0f172a,stroke:#6366f1,stroke-width:2px,color:#e0e7ff,rx:8px,ry:8px;
    classDef apiLayer fill:#0f172a,stroke:#10b981,stroke-width:2px,color:#d1fae5,rx:8px,ry:8px;
    classDef uiLayer fill:#161b22,stroke:#4f46e5,stroke-width:2px,color:#ffffff,rx:12px,ry:12px;
    classDef monitor fill:#1e1e1e,stroke:#f59e0b,stroke-width:2px,color:#fef3c7,rx:8px,ry:8px;

    %% Data Pipeline
    subgraph Data Engineering & Training
        A[(Data Telco CSV)]:::dataBox --> B(SMOTE Imbalance Handler):::dataBox
        B --> C{XGBoost + GridSearchCV}:::aiEngine
        C -->|Pencatatan Metrik| D[(Registri MLflow)]:::aiEngine
        C -->|Ekspor| E[Model Artifacts: pkl]:::dataBox
    end

    %% Deployment Pipeline
    subgraph Production Serving
        E -.->|Dimuat ke| F(FastAPI Backend):::apiLayer
        F --> G{Mesin Inferensi}:::apiLayer
        G <-->|Payload JSON| H[Quiet Luxury UI Dashboard]:::uiLayer
    end

    %% Telemetry Pipeline
    subgraph Infrastructure Telemetry
        F -->|/metrics| I(Prometheus Scraper):::monitor
        I --> J[Dashboard Grafana]:::monitor
    end
```

---

## ✨ Visual Showcase & Kapabilitas

<div align="center">
  <table style="width:100%; text-align:center;">
    <tr>
      <td width="50%">
        <b>1. The Executive Dashboard (Status Siap)</b><br><br>
        <i>Antarmuka utama bergaya "Quiet Luxury" yang menanti input data profil pelanggan.</i><br>
        <img src="assets/The_Executive_Dashboard.png" alt="Idle Dashboard" style="max-width: 100%; height: auto;">
      </td>
      <td width="50%">
        <b>2. High-Risk Churn Detection</b><br><br>
        <i>Deteksi otomatis pelanggan berisiko tinggi beserta proyeksi kerugian pendapatan (LTV Loss).</i><br>
        <img src="assets/High-Risk%20Churn%20Detection.png" alt="High Risk Detection" style="max-width: 100%; height: auto;">
      </td>
    </tr>
    <tr>
      <td width="50%">
        <b>3. Safe / Loyal Customer Assessment</b><br><br>
        <i>Evaluasi pelanggan dengan tingkat retensi tinggi yang diproyeksikan stabil.</i><br>
        <img src="assets/Loyal%20Customer%20Assessment.png" alt="Loyal Profile" style="max-width: 100%; height: auto;">
      </td>
      <td width="50%">
        <b>4. Explainable AI (SHAP Impact)</b><br><br>
        <i>Grafik SHAP Impact yang memberikan transparansi alasan matematis di balik prediksi AI.</i><br>
        <img src="assets/Inference%20Driver%20Weights.png" alt="SHAP Impact" style="max-width: 100%; height: auto;">
      </td>
    </tr>
    <tr>
      <td width="50%">
        <b>5. Executive Retention Queue</b><br><br>
        <i>Notifikasi interaktif saat eksekutif menekan tombol mitigasi pada antrean akun berisiko.</i><br>
        <img src="assets/Executive%20Retention%20Queue.png" alt="Queue Notification" style="max-width: 100%; height: auto;">
      </td>
      <td width="50%">
        <b>6. REST API Documentation</b><br><br>
        <i>Dokumentasi API otomatis (Swagger) untuk melayani proses inferensi dengan latensi rendah.</i><br>
        <img src="assets/REST%20API%20Documentation.png" alt="FastAPI Swagger" style="max-width: 100%; height: auto;">
      </td>
    </tr>
    <tr>
      <td width="50%">
        <b>7. Real-Time Telemetry</b><br><br>
        <i>Dashboard Grafana yang memvisualisasikan metrik infrastruktur secara real-time.</i><br>
        <img src="assets/Real-Time%20Telemetry%20Grafana.png" alt="Grafana Telemetry" style="max-width: 100%; height: auto;">
      </td>
      <td width="50%">
        <b>8. Experiment Tracking</b><br><br>
        <i>Tracking performa AI secara otomatis dan komparasi model XGBoost menggunakan antarmuka MLflow.</i><br>
        <img src="assets/Experiment%20Tracking.png" alt="MLflow Tracker" style="max-width: 100%; height: auto;">
      </td>
    </tr>
  </table>
</div>

---

## ⚙️ Alur Kerja MLOps Step-by-Step

Proyek ini tidak sekadar menghasilkan prediksi, melainkan mematuhi siklus hidup *Machine Learning* yang matang dan tersandardisasi.

| Tahap | Proses yang Dieksekusi | Teknologi Inti |
| :--- | :--- | :--- |
| **1. Data Preprocessing** | Pembersihan data, pemetaan kamus `O(1)` untuk latensi rendah, One-Hot Encoding. | `pandas`, `scikit-learn` |
| **2. Resampling & Balancing** | Mengimplementasikan SMOTE untuk meningkatkan sensitivitas AI terhadap deteksi ancaman kelas minoritas (Churn). | `imbalanced-learn` |
| **3. Model Training** | Pelatihan **XGBoost Classifier** dan **CoxPHFitter** (Survival Analysis untuk sisa Tenure). | `xgboost`, `lifelines` |
| **4. Experiment Tracking** | Secara otomatis merekam Hyperparameter, accuracy, f1-score, dan menyimpan artifak model (`.pkl`). | `mlflow` |
| **5. Continuous Integration** | Otomatisasi proses Linting (Flake8) dan Pengujian (Pytest) melalui GitHub Actions setiap ada perubahan kode. | `GitHub Actions` |
| **6. Containerized Deployment** | Backend FastAPI menyajikan model dan antarmuka (UI) di dalam kontainer docker yang terisolasi penuh. | `Docker`, `Docker Compose` |

---

## 🚀 Panduan Memulai Cepat

Proses *deploy* (penerapan) seluruh infrastruktur (Model, API, UI, dan Telemetri) di mesin lokal Anda sangat ringkas.

### Prasyarat
- [Python 3.10+](https://www.python.org/downloads/)
- [Docker & Docker Compose](https://www.docker.com/products/docker-desktop/)

### Langkah 1: Inisialisasi Environment & Training
Pertama, instal dependensi lokal dan latih algoritma XGBoost untuk menghasilkan artifak `.pkl`.
```bash
# Instal seluruh dependensi
pip install -r requirements.txt

# Jalankan skrip training
python src/train.py
```

### Langkah 2: Menjalankan Enterprise Stack (Docker)
Setelah model berhasil dilatih, bangun dan jalankan seluruh arsitektur *microservices*.
```bash
docker-compose up --build -d
```

### Langkah 3: Akses Platform
Tunggu beberapa detik hingga seluruh kontainer berjalan stabil, kemudian buka tautan berikut di browser Anda:
- 💎 **Web Dashboard (UI):** [http://localhost:8000](http://localhost:8000)
- ⚙️ **FastAPI Swagger:** [http://localhost:8000/docs](http://localhost:8000/docs)
- 📈 **Grafana Monitoring:** [http://localhost:3000](http://localhost:3000) (Login: `admin` / `admin`)
- 🧪 **Experiment Tracking (MLflow):** 
  - Windows: Jalankan `$env:MLFLOW_ALLOW_FILE_STORE="true"; python -m mlflow ui` di terminal PowerShell.
  - Mac/Linux: Jalankan `MLFLOW_ALLOW_FILE_STORE=true python -m mlflow ui` di terminal.
  - Lalu buka [http://localhost:5000](http://localhost:5000)

---

## 📁 Struktur Direktori Proyek

```text
mlops-churn-dicoding/
├── .github/workflows/       # Pipeline CI/CD otomatis (Linting & Pytest)
├── app/
│   ├── main.py              # Server FastAPI (Inferensi & Serving File Statis)
│   ├── models_schemas.py    # Skema validasi input Pydantic
│   └── static/              
│       └── index.html       # UI Enterprise Dashboard "Quiet Luxury"
├── assets/                  # Kumpulan gambar visualisasi untuk Repositori
├── data/                    # Direktori dataset (CSV)
├── models/                  # Artifak hasil pelatihan model (.pkl)
├── monitoring/
│   ├── grafana/             # Konfigurasi penyediaan dashboard Grafana
│   └── prometheus/          # Skrip scraping Prometheus
├── src/
│   ├── preprocess.py        # Logika pipeline transformasi data
│   └── train.py             # Skrip inti pelatihan XGBoost, SMOTE, dan MLflow
├── docker-compose.yml       # Orkestrasi multi-container
├── Dockerfile               # Skrip pembangunan container image FastAPI
└── requirements.txt         # Daftar spesifikasi versi library Python
```

---

<div align="center">
  <i>Diarsiteki untuk melampaui standar industri sebagai pemenuhan Sertifikasi "Membangun Sistem Machine Learning" Dicoding.</i>
</div>
