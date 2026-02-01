"""
Step D: Explainable AI with Feature Importances
Generate global and local explanations for the default risk model.

Notes:
- This version intentionally avoids SHAP to keep the demo lightweight and stable.
- Uses model-native feature_importances_ (e.g., RandomForest) for global importance.
- Uses a simple local explanation proxy: standardized feature value × global importance.
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


class ExplainabilityGenerator:
    """Generate explainability artifacts for credit models"""

    def __init__(self, base_dir=None):
        # Prefer project root as base_dir:
        # If this file is scripts/step_d_explainability.py -> parent.parent is project root.
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)

        self.data_dir = self.base_dir / "outputs" / "datasets"
        self.models_dir = self.base_dir / "outputs" / "models"
        self.figures_dir = self.base_dir / "outputs" / "figures"
        self.reports_dir = self.base_dir / "reports"

        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_models(self):
        """Load trained credit engine package"""
        model_file = self.models_dir / "credit_engine.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model file: {model_file}")

        with open(model_file, "rb") as f:
            engine_package = pickle.load(f)

        # Minimal validation
        if "models" not in engine_package:
            raise ValueError("credit_engine.pkl missing key: 'models'")
        if "default_risk" not in engine_package["models"]:
            raise ValueError("credit_engine.pkl missing models['default_risk']")
        if "encoders" not in engine_package or "scaler" not in engine_package or "feature_names" not in engine_package:
            raise ValueError("credit_engine.pkl missing one of: encoders, scaler, feature_names")

        return engine_package

    def load_data(self):
        """Load safe training dataset used for modeling"""
        data_file = self.data_dir / "tunisia_loan_data_train.csv"
        if not data_file.exists():
            raise FileNotFoundError(f"Missing training data file: {data_file}")

        df = pd.read_csv(data_file)
        return df

    def prepare_features(self, df: pd.DataFrame, encoders: dict, feature_names: list):
        """
        Prepare features consistent with training:
        - Select columns used during training (feature_names)
        - Encode categoricals using stored encoders
        - Fill numeric missing values with median
        """
        # Select expected features; if any are missing, create them with safe defaults
        X = df.copy()

        missing = [c for c in feature_names if c not in X.columns]
        for c in missing:
            # conservative defaults: numeric -> 0, object -> "Unknown"
            X[c] = 0

        X = X[feature_names].copy()

        # Encode categorical features
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        for col in categorical_cols:
            if col in encoders:
                le = encoders[col]
                X[col] = X[col].fillna("Unknown").astype(str)

                # handle unseen categories by mapping to "Unknown" if present, else first class
                if "Unknown" in getattr(le, "classes_", []):
                    fallback = "Unknown"
                else:
                    fallback = le.classes_[0] if len(le.classes_) > 0 else "0"

                X[col] = X[col].apply(lambda v: v if v in le.classes_ else fallback)
                X[col] = le.transform(X[col])
            else:
                # If encoder missing, do a stable fallback encoding
                X[col] = X[col].fillna("Unknown").astype("category").cat.codes

        # Fill missing numeric values
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        return X

    def generate_feature_importances(self, model, feature_names):
        """Extract feature importances from model (must have feature_importances_)"""
        if not hasattr(model, "feature_importances_"):
            raise TypeError("Model does not expose feature_importances_. Use a tree-based model or update explainability.")

        importances = np.array(model.feature_importances_, dtype=float)

        if len(importances) != len(feature_names):
            raise ValueError(
                f"Importances length ({len(importances)}) != feature_names length ({len(feature_names)})"
            )

        return importances

    def plot_global_summary(self, importances, feature_names):
        """Create global feature importance bar chart (top 15)"""
        print("Creating global importance plot...")

        importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=True)
            .tail(15)
        )

        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df["importance"])
        plt.yticks(range(len(importance_df)), importance_df["feature"])
        plt.xlabel("Feature Importance", fontsize=12)
        plt.title("Global Feature Importance - Default Risk Model", fontsize=14, fontweight="bold")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        output_file = self.figures_dir / "global_feature_importance.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved: {output_file}")

    def plot_local_explanation(self, model, importances, X_sample_scaled: pd.DataFrame, feature_names, idx=0):
        """
        Create a local explanation proxy for a specific applicant:
        contribution ≈ standardized_feature_value × global_importance

        Inputs:
        - X_sample_scaled must already be scaled with the training scaler
        """
        print(f"Creating local explanation for applicant at index {idx}...")

        if idx < 0 or idx >= len(X_sample_scaled):
            idx = 0

        sample = X_sample_scaled.iloc[idx : idx + 1]
        pred_proba = float(model.predict_proba(sample)[0][1]) if hasattr(model, "predict_proba") else float(model.predict(sample)[0])
        pred_class = int(model.predict(sample)[0]) if hasattr(model, "predict") else int(pred_proba >= 0.5)

        feature_values = sample.iloc[0].values.astype(float)
        contributions = feature_values * np.array(importances, dtype=float)

        contrib_df = pd.DataFrame(
            {"feature": feature_names, "value(z)": feature_values, "contribution": contributions}
        )
        contrib_df["abs_contribution"] = contrib_df["contribution"].abs()
        contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).head(12)

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(contrib_df)), contrib_df["contribution"])
        plt.yticks(
            range(len(contrib_df)),
            [f"{feat} = {val:.2f}" for feat, val in zip(contrib_df["feature"], contrib_df["value(z)"])],
        )
        plt.xlabel("Contribution (z-score × importance)", fontsize=11)
        plt.title(
            f"Local Explanation (Proxy) - Applicant {idx}\nPredicted Default Probability: {pred_proba:.2%} (Class: {pred_class})",
            fontsize=12,
            fontweight="bold",
        )
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        output_file = self.figures_dir / "local_explanation_example.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✓ Saved: {output_file}")

    def get_top_features(self, importances, feature_names, top_n=10):
        """Get top features by global importance"""
        feature_importance = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        return feature_importance

    def generate_notes(self, top_features: pd.DataFrame):
        """Generate explainability notes (markdown)"""
        notes = [
            "# Explainability Analysis - Default Risk Model",
            "",
            "## Overview",
            "This report explains how the default risk model makes predictions using **model-native feature importance**.",
            "",
            "## Global Feature Importance",
            "Top features driving model predictions (by global importance):",
            "",
        ]

        for i, row in top_features.iterrows():
            notes.append(f"{i+1}. **{row['feature']}**: {row['importance']:.4f}")

        notes.extend(
            [
                "",
                "## How to Interpret",
                "- **Higher importance** means the model relies more on that feature overall.",
                "- Importance does **not** indicate direction (increase/decrease risk); it indicates influence strength.",
                "",
                "## Local Explanations (Proxy)",
                "The local chart uses a simple proxy contribution:",
                "- Contribution ≈ **standardized feature value (z-score)** × **global importance**",
                "- This is not SHAP, but it gives an intuitive demo-friendly explanation of what is pushing a specific case.",
                "",
                "## Recommendations",
                "1. Add true SHAP later for stronger, direction-aware explanations (requires `shap` dependency).",
                "2. Monitor importance drift over time (data distribution changes).",
                "3. Validate key features with ATB domain experts for business sanity checks.",
            ]
        )

        return "\n".join(notes)

    def run(self):
        """Execute explainability pipeline"""
        print("=" * 80)
        print("STEP D: EXPLAINABILITY (FEATURE IMPORTANCE)")
        print("=" * 80)

        # Load models
        print("\nLoading trained models...")
        engine_package = self.load_models()
        model = engine_package["models"]["default_risk"]
        encoders = engine_package["encoders"]
        scaler = engine_package["scaler"]
        feature_names = engine_package["feature_names"]

        print(f"✓ Loaded default risk model with {len(feature_names)} features")

        # Load and prepare data
        print("\nLoading data for explainability...")
        df = self.load_data()

        X = self.prepare_features(df, encoders, feature_names)

        # Scale features (consistent with training)
        X_scaled = scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

        # Sample to keep plots readable
        sample_size = min(500, len(X_scaled))
        X_sample = X_scaled.sample(n=sample_size, random_state=42)

        print(f"✓ Using {len(X_sample)} samples for explainability")

        # Generate feature importances
        importances = self.generate_feature_importances(model, feature_names)

        # Visualizations
        print("\nCreating explainability visualizations...")
        self.plot_global_summary(importances, feature_names)
        self.plot_local_explanation(model, importances, X_sample, feature_names, idx=0)

        # Top features table
        top_features = self.get_top_features(importances, feature_names, top_n=10)
        print("\nTop 10 features by importance:")
        print(top_features.to_string(index=False))

        # Notes
        print("\nGenerating explainability notes...")
        notes = self.generate_notes(top_features)
        notes_file = self.reports_dir / "explainability_notes.md"
        with open(notes_file, "w", encoding="utf-8") as f:
            f.write(notes)
        print(f"✓ Saved: {notes_file}")

        print("\n" + "=" * 80)
        print("STEP D COMPLETE")
        print("=" * 80)
        print("\nGenerated artifacts:")
        print(f"- {self.figures_dir / 'global_feature_importance.png'}")
        print(f"- {self.figures_dir / 'local_explanation_example.png'}")
        print(f"- {notes_file}")

        return importances, top_features


if __name__ == "__main__":
    explainer = ExplainabilityGenerator()
    importances, top_features = explainer.run()