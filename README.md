# 🏠 Housing Price Prediction Platform

End-to-end ML system for housing price prediction with automated deployment pipeline.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Production ML pipeline that predicts housing prices through feature engineering, ensemble modeling, and real-time inference API. Built for scalability with containerized microservices and serverless deployment on Google Cloud Platform.

**Live Demo**: [Streamlit Dashboard](https://housing-streamlit-65985976686.europe-west1.run.app/dashboard) | [API Docs](https://housing-api-65985976686.europe-west1.run.app/docs)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │    FastAPI      │    │   XGBoost       │
│   (Frontend)    │◄──►│    (Backend)    │◄──►│   Model         │
│                 │    │                 │    │                 │
│ • Data Viz      │    │ • /predict      │    │ • Inference     │
│ • User Filters  │    │ • Batch Predict │    │ • Feature Eng   │
│ • Performance   │    │ • Health Check  │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   Google Cloud  │
                    │   Platform      │
                    │                 │
                    │ • Cloud Run     │
                    │ • Cloud Storage │
                    │ • Artifact Reg  │
                    │ • Workload ID   │
                    └─────────────────┘
```

**Stack**: Python • FastAPI • Streamlit • XGBoost • Docker • Cloud Run • Artifact Registry • GitHub Actions



## Features

- **Data Engineering**: Automated ETL with feature engineering pipeline
- **ML Model**: XGBoost ensemble achieving 90%+ prediction accuracy (MAE ~$50K, RMSE ~$70K on validation)
- **REST API**: FastAPI backend with batch prediction support
- **Interactive UI**: Streamlit dashboard with visualizations
- **CI/CD**: Automated Docker builds and deployments via GitHub Actions
- **Cloud Infrastructure**: Serverless deployment with auto-scaling on GCP
- **Security**: Workload Identity Federation (keyless authentication)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker
- Google Cloud account with billing enabled
- [uv](https://github.com/astral-sh/uv) package manager
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (for local development)

#### System Dependencies

**macOS**: Install OpenMP runtime (required for XGBoost):
```bash
brew install libomp
```

**Linux**: Usually pre-installed, but if needed:
```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# CentOS/RHEL
sudo yum install libomp-devel
```

**Windows**: OpenMP is included with Visual Studio Build Tools.

After installing system dependencies, reinstall Python packages:
```bash
uv sync --reinstall
```

#### Install uv (if not already installed)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip (alternative)
pip install uv
```

### Environment Variables

Set these environment variables for local development:

```bash
# Required for API and Streamlit
export GCS_BUCKET="your-gcs-bucket-name"  # Your Google Cloud Storage bucket
export API_URL="http://127.0.0.1:8000/predict"  # API endpoint URL

# Optional (for GCP operations)
export GOOGLE_CLOUD_PROJECT="your-project-id"  # GCP project ID

# Optional (with defaults)
export MODEL_PATH="models/xgboost_model.pkl"  # Local model path (fallback)
export LOG_LEVEL="INFO"  # Logging level
```

### Local Development

#### Prerequisites
- Ensure you have trained a model and uploaded data to GCS (see training section below)

#### Data Setup
The project expects processed data in `data/processed/` directory. If starting fresh:

1. Place raw data files in `data/raw/`:
   - `train.csv`
   - `eval.csv` 
   - `holdout.csv`
   - `usmetros.csv`

2. Run the preprocessing pipeline (see Training section below)

3. Upload processed data and trained model to Google Cloud Storage

#### Running the Application
```bash
# Clone repository
git clone https://github.com/xinchengppy/Housing_Price_ML_E2E.git
cd Housing_Price_ML_E2E

# Install dependencies
uv sync

# Set environment variables (optional, defaults provided)
export GCS_BUCKET="your-bucket-name"
export API_URL="http://127.0.0.1:8000/predict"

# Run API server
uv run uvicorn src.api.main:app --port 8000

# Run Streamlit dashboard (separate terminal)
streamlit run app.py

# Run tests
uv run pytest

# Run API with auto-reload (development)
uv run uvicorn src.api.main:app --reload --port 8000
```
```

Visit:
- API: http://localhost:8000/docs
- Dashboard: http://localhost:8501

#### Training a Model (Required before using API)
```bash
# 1. Load and split raw data
python src/feature_pipeline/load.py

# 2. Preprocess splits
python -m src.feature_pipeline.preprocess

# 3. Feature engineering
python -m src.feature_pipeline.feature_engineering

# 4. Train baseline model
python src/training_pipeline/train.py

# 5. Hyperparameter tuning (optional but recommended)
python src/training_pipeline/tune.py

# 6. Model evaluation
python src/training_pipeline/eval.py

# 7. Upload to GCS
# Run notebooks/07_Google_GCP_push_datasets.ipynb to upload model and data
```


## Cloud Deployment

### GCP Setup

**1. Create Workload Identity Pool**
```bash
gcloud iam workload-identity-pools create github-pool \
  --location=global \
  --project=YOUR_PROJECT_ID
```

**2. Configure OIDC Provider**
```bash
gcloud iam workload-identity-pools providers create-oidc github-actions \
  --location=global \
  --workload-identity-pool=github-pool \
  --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping=google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository \
  --attribute-condition="assertion.repository_owner == 'YOUR_GITHUB_USERNAME'" \
  --project=YOUR_PROJECT_ID
```

**3. Create Service Account**
```bash
gcloud iam service-accounts create github-actions-sa \
  --display-name="GitHub Actions" \
  --project=YOUR_PROJECT_ID
```

**4. Get Project Number**
```bash
gcloud projects describe YOUR_PROJECT_ID --format="value(projectNumber)"
```
*Note: Save this number as YOUR_PROJECT_NUMBER for the next step.*

**5. Bind Workload Identity**
```bash
gcloud iam service-accounts add-iam-policy-binding \
  github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/YOUR_PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME" \
  --project=YOUR_PROJECT_ID
```

**6. Grant Permissions**
```bash
# Artifact Registry
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

# Cloud Storage
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Cloud Run
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/run.admin"

# Service Account User
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
  --member="serviceAccount:github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Compute Engine Default Account
gcloud iam service-accounts add-iam-policy-binding \
  YOUR_PROJECT_NUMBER-compute@developer.gserviceaccount.com \
  --member="serviceAccount:github-actions-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser" \
  --project=YOUR_PROJECT_ID
```

**7. Create Artifact Registry**
```bash
gcloud artifacts repositories create YOUR_REPO_NAME \
  --repository-format=docker \
  --location=europe-west1 \
  --project=YOUR_PROJECT_ID
```

**8. Create GCS Bucket and Upload Data**
```bash
# Create bucket
gsutil mb -p YOUR_PROJECT_ID -l europe-west1 gs://YOUR_GCS_BUCKET/

# Upload processed data (run locally after training)
Run [notebooks/07_Google_GCP_push_datasets.ipynb](https://github.com/xinchengppy/Housing_Price_ML_E2E/blob/main/notebooks/07_Google_GCP_push_datasets.ipynb) to upload data and model to GCS.
```


### CI/CD Pipeline

Push to `main` branch triggers automated:
1. Docker image builds (API + Streamlit)
2. Push to Artifact Registry
3. Deploy to Cloud Run (europe-west1 in my case)

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml) for details.


## Project Structure
```
.
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI application with prediction endpoints
│   ├── batch/                   # Batch processing scripts for monthly predictions
│   ├── feature_pipeline/        # Data preprocessing and feature engineering
│   │   ├── load.py              # Load and split raw data
│   │   ├── preprocess.py        # Clean and preprocess data splits
│   │   └── feature_engineering.py # Create ML features
│   ├── inference_pipeline/      # Model inference logic and utilities
│   │   └── inference.py         # Core prediction function
│   └── training_pipeline/       # Model training and evaluation
│       ├── train.py             # Train baseline models
│       ├── tune.py              # Hyperparameter tuning with MLflow
│       └── eval.py              # Model evaluation and metrics
├── app.py                       # Streamlit dashboard application
├── main.py                      # Project entry point (simple script)
├── Dockerfile                   # Container for API service
├── Dockerfile.streamlit         # Container for Streamlit UI
├── data/
│   ├── raw/                     # Original datasets (train.csv, eval.csv, etc.)
│   ├── processed/               # Cleaned and engineered features
│   └── predictions/             # Batch prediction outputs
├── notebooks/                   # Jupyter notebooks for exploration and GCP ops
│   ├── 00_data_split.ipynb      # Data exploration
│   ├── 01_EDA_cleaning.ipynb    # Exploratory data analysis
│   ├── 02_feature_eng_encoding.ipynb # Feature engineering
│   ├── 03_baseline.ipynb        # Baseline model training
│   ├── 04_linear_regression_regularization.ipynb # Linear models
│   ├── 05_XGBoost.ipynb         # XGBoost model development
│   ├── 06_hyperparameter_tuning_MFLow.ipynb # MLflow experiments
│   └── 07_Google_GCP_push_datasets.ipynb # Upload to GCS
├── tests/                       # Unit tests and integration tests
├── logs/                        # Application and training logs
├── mlruns/                      # MLflow experiment tracking
├── models/                      # Trained model artifacts
├── .github/workflows/
│   └── ci.yml                   # GitHub Actions CI/CD pipeline
├── pyproject.toml               # Python dependencies and project config
└── README.md                    # This file
```

## Troubleshooting

### Common Issues

**API fails to start**
- Ensure model and data are uploaded to GCS
- Check environment variables: `GCS_BUCKET`, `API_URL`
- Verify Google Cloud credentials are configured

**Streamlit can't connect to API**
- Ensure API is running on port 8000
- Check `API_URL` environment variable
- Verify firewall settings allow local connections

**Model prediction errors**
- Ensure training data schema matches input data
- Check for missing features in prediction payload
- Verify model file exists in GCS bucket

**GCS upload fails**
- Authenticate with `gcloud auth login`
- Set correct project with `gcloud config set project YOUR_PROJECT_ID`
- Ensure bucket exists and permissions are correct

**Docker build fails**
- Check Docker daemon is running
- Ensure all dependencies are in `pyproject.toml`
- Verify base image is accessible

### Getting Help
- Check [API documentation](http://localhost:8000/docs) when running locally
- Review logs in `logs/` directory
- Check GitHub Actions workflow runs for deployment issues
- Open an [issue](https://github.com/xinchengppy/Housing_Price_ML_E2E/issues) for bugs or feature requests

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

*Built with Python, FastAPI, and a healthy dose of caffeine ☕*