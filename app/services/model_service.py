import pickle
from pathlib import Path
import pandas as pd
import numpy as np

class ModelService:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.engine = None

    def load(self):
        with open(self.model_path, "rb") as f:
            self.engine = pickle.load(f)

        # Expected keys from your Step C:
        # engine['models'], engine['encoders'], engine['scaler'], engine['feature_names']
        return self

    @property
    def feature_names(self):
        return self.engine["feature_names"]

    def _encode_and_align(self, payload: dict):
        encoders = self.engine["encoders"]
        scaler = self.engine["scaler"]
        feature_names = self.engine["feature_names"]

        # Build a single-row DataFrame with all expected features
        row = {f: payload.get(f, None) for f in feature_names}
        X = pd.DataFrame([row])

        # Encode categoricals using saved encoders
        for col, le in encoders.items():
            if col == "affordability":
                continue
            if col in X.columns:
                X[col] = X[col].fillna("Unknown").astype(str)
                # unseen categories -> "Unknown" if present, else first class
                classes = set(le.classes_.tolist())
                if "Unknown" in classes:
                    X[col] = X[col].apply(lambda v: v if v in classes else "Unknown")
                else:
                    X[col] = X[col].apply(lambda v: v if v in classes else le.classes_[0])
                X[col] = le.transform(X[col])

        # Fill numerics
        for col in X.columns:
            if X[col].dtype == "object":
                # if still object, coerce to numeric if possible, else 0
                X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].fillna(0)

        # Scale
        X_scaled = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
        return X_scaled

    def score(self, payload: dict):
        models = self.engine["models"]
        X = self._encode_and_align(payload)

        # Default risk
        dr_model = models["default_risk"]
        dr_proba = float(dr_model.predict_proba(X)[:, 1][0])

        # Eligibility (your model predicts eligibility where 1=eligible)
        el_model = models["eligibility"]
        el_proba = float(el_model.predict_proba(X)[:, 1][0])

        # Affordability class
        af_model = models["affordability"]
        af_enc = self.engine["encoders"]["affordability"]
        af_pred = int(af_model.predict(X)[0])
        af_label = str(af_enc.inverse_transform([af_pred])[0])

        return {
            "eligibility": {"probability": el_proba, "label": "Eligible" if el_proba >= 0.5 else "Not Eligible"},
            "default_risk": {"probability": dr_proba, "label": "High Risk" if dr_proba >= 0.5 else "Lower Risk"},
            "affordability": {"band": af_label},
        }, X
