"""
ChurnPredictor: handles training, persistence, and inference.

Improvements over original model.py:
  - Saves LabelEncoders per-column so inference uses identical encoding
  - Uses calibrated probabilities (CalibratedClassifierCV)
  - Returns feature importance alongside predictions
  - Stores model version as timestamp
  - Prevents data leakage: encoders fit only on training data
"""

import os
import pickle
import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder

from src.schemas import CustomerFeatures, PredictionResponse, BatchPrediction, BatchResponse

logger = logging.getLogger(__name__)

MODEL_DIR = "model_artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
META_PATH = os.path.join(MODEL_DIR, "meta.pkl")

CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]

FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def _risk_level(prob: float) -> str:
    if prob < 0.35:
        return "LOW"
    elif prob < 0.65:
        return "MEDIUM"
    return "HIGH"


class ChurnPredictor:
    def __init__(self):
        self._model: Optional[CalibratedClassifierCV] = None
        self._encoders: dict[str, LabelEncoder] = {}
        self._feature_names: list[str] = []
        self._feature_importances: dict[str, float] = {}
        self._training_metrics: dict = {}
        self.model_version: Optional[str] = None

    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ #
    #  Training                                                            #
    # ------------------------------------------------------------------ #
    def train(self, data_path: str) -> dict:
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path, index_col="customerID")

        # --- Clean ---
        df.replace(" ", np.nan, inplace=True)
        df.dropna(inplace=True)
        df["TotalCharges"] = df["TotalCharges"].astype(float)

        # --- Encode target ---
        target_le = LabelEncoder()
        y = target_le.fit_transform(df["Churn"])  # No -> 0, Yes -> 1

        X = df[FEATURE_COLS].copy()

        # --- Fit encoders on training split only (no leakage) ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = X_test[col].astype(str).map(
                lambda v, le=le: le.transform([v])[0]
                if v in le.classes_
                else -1
            )
            encoders[col] = le

        # --- Train ---
        base_rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",   # handles class imbalance
            random_state=42,
            n_jobs=-1,
        )
        model = CalibratedClassifierCV(base_rf, cv=5, method="isotonic")
        model.fit(X_train, y_train)

        # --- Evaluate ---
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        cv_scores = cross_val_score(base_rf, X_train, y_train, cv=5, scoring="roc_auc")

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1_score": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std": round(cv_scores.std(), 4),
            "classification_report": classification_report(y_test, y_pred),
        }

        # Feature importances from underlying RF
        importances = np.mean([
        est.estimator.feature_importances_
        for est in model.calibrated_classifiers_
        ], axis=0)
        feat_imp = dict(zip(FEATURE_COLS, importances.tolist()))

        # --- Persist ---
        os.makedirs(MODEL_DIR, exist_ok=True)
        version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)
        with open(ENCODERS_PATH, "wb") as f:
            pickle.dump(encoders, f)
        with open(META_PATH, "wb") as f:
            pickle.dump({"version": version, "metrics": metrics, "feature_importances": feat_imp}, f)

        # Update in-memory state
        self._model = model
        self._encoders = encoders
        self._feature_names = FEATURE_COLS
        self._feature_importances = feat_imp
        self._training_metrics = metrics
        self.model_version = version

        logger.info(f"Model trained. AUC={metrics['roc_auc']}, version={version}")
        return metrics

    # ------------------------------------------------------------------ #
    #  Load                                                                #
    # ------------------------------------------------------------------ #
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            logger.warning("No saved model found. Call /train first.")
            return
        with open(MODEL_PATH, "rb") as f:
            self._model = pickle.load(f)
        with open(ENCODERS_PATH, "rb") as f:
            self._encoders = pickle.load(f)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        self.model_version = meta["version"]
        self._feature_importances = meta["feature_importances"]
        self._training_metrics = meta["metrics"]
        self._feature_names = FEATURE_COLS
        logger.info(f"Loaded model version {self.model_version}")

    # ------------------------------------------------------------------ #
    #  Inference helpers                                                   #
    # ------------------------------------------------------------------ #
    def _build_input_df(self, customer: CustomerFeatures) -> pd.DataFrame:
        row = customer.model_dump()
        df = pd.DataFrame([row])[FEATURE_COLS]
        for col in CATEGORICAL_COLS:
            le = self._encoders[col]
            val = str(df.at[0, col])
            df[col] = le.transform([val])[0] if val in le.classes_ else -1
        return df

    def predict(self, customer: CustomerFeatures) -> PredictionResponse:
        X = self._build_input_df(customer)
        prob = float(self._model.predict_proba(X)[0, 1])
        churn = prob >= 0.5

        # Top 3 risk factors
        sorted_features = sorted(
            self._feature_importances.items(), key=lambda x: x[1], reverse=True
        )[:3]
        raw = customer.model_dump()
        top_factors = [
            {"feature": f, "importance": round(imp, 4), "value": str(raw.get(f, ""))}
            for f, imp in sorted_features
        ]

        return PredictionResponse(
            churn=churn,
            churn_probability=round(prob, 4),
            risk_level=_risk_level(prob),
            top_risk_factors=top_factors,
        )

    def predict_batch(self, customers: list[CustomerFeatures]) -> BatchResponse:
        results = []
        for i, c in enumerate(customers):
            X = self._build_input_df(c)
            prob = float(self._model.predict_proba(X)[0, 1])
            results.append(BatchPrediction(
                index=i,
                churn=prob >= 0.5,
                churn_probability=round(prob, 4),
                risk_level=_risk_level(prob),
            ))
        churn_count = sum(1 for r in results if r.churn)
        return BatchResponse(
            total=len(results),
            churn_count=churn_count,
            churn_rate=round(churn_count / len(results), 4),
            predictions=results,
        )

    def get_model_info(self) -> dict:
        return {
            "model_version": self.model_version,
            "algorithm": "RandomForest + CalibratedClassifierCV (isotonic)",
            "features": self._feature_names,
            "feature_importances": dict(
                sorted(self._feature_importances.items(), key=lambda x: x[1], reverse=True)
            ),
            "training_metrics": {
                k: v for k, v in self._training_metrics.items()
                if k != "classification_report"
            },
        }
