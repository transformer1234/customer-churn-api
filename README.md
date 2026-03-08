# 🔁 Customer Churn Prediction — REST API

A production-ready FastAPI service for predicting customer churn.

---

## 🚀 Quick Start

### Local (Python)
```bash
pip install -r requirements.txt

# 1. Train the model (point to your CSV)
curl -X POST "http://localhost:8000/train?data_path=data/Telco-Customer-Churn.csv"

# 2. Start the server
uvicorn main:app --reload
```

### Docker
```bash
docker-compose up --build
```

The API will be live at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## 📡 Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + model status |
| POST | `/predict` | Single customer prediction |
| POST | `/predict/batch` | Batch prediction (up to 500) |
| POST | `/train` | Retrain model on new data |
| GET | `/model/info` | Feature importances + metrics |

---

## 📥 Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.20
  }'
```

### Example Response

```json
{
  "churn": true,
  "churn_probability": 0.7823,
  "risk_level": "HIGH",
  "top_risk_factors": [
    {"feature": "Contract", "importance": 0.142, "value": "Month-to-month"},
    {"feature": "tenure", "importance": 0.138, "value": "12"},
    {"feature": "MonthlyCharges", "importance": 0.121, "value": "70.35"}
  ]
}
```

---

## ☁️ Deployment Options

### Option 1: Railway / Render (Easiest)
1. Push this folder to a GitHub repo
2. Connect to [Railway](https://railway.app) or [Render](https://render.com)
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Done — public URL provided automatically

### Option 2: AWS / GCP / Azure (Production)
- Build Docker image → push to ECR/GCR/ACR
- Deploy on ECS Fargate / Cloud Run / Azure Container Apps
- Mount a persistent volume for `model_artifacts/`

### Option 3: Heroku
```bash
heroku create churn-api
heroku container:push web
heroku container:release web
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 🔧 Improvements Over Original

| Original | This API |
|----------|----------|
| Single script, no serving | Full REST API with FastAPI |
| LabelEncoder refit on predict | Encoders saved & reused (no data leakage) |
| No probability calibration | CalibratedClassifierCV (isotonic) |
| No class imbalance handling | `class_weight="balanced"` |
| Accuracy only | Accuracy + F1 + ROC-AUC + CV scores |
| No input validation | Pydantic v2 strict validation + enums |
| No batch support | Batch endpoint (up to 500) |
| No feature importance output | Top risk factors per prediction |
| No persistence | Model versioned & saved to disk |
| No tests | pytest test suite |
| Not containerized | Dockerfile + docker-compose |
