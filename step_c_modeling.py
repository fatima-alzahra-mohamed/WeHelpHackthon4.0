import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

import warnings
warnings.filterwarnings("ignore")


class UnifiedCreditEngine:
    """
    Unified credit engine:
      - default_risk_proba: P(default=1)
      - eligibility: boolean decision derived from default risk + policy threshold
      - affordability_band: rule-based band (Low/Medium/High)
    """

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent

        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "outputs" / "datasets"
        self.models_dir = self.base_dir / "outputs" / "models"
        self.reports_dir = self.base_dir / "reports"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.model = None  # calibrated pipeline
        self.preprocessor = None
        self.feature_cols = None
        self.thresholds = {}
        self.evaluation_results = {}

    # -----------------------------
    # Data
    # -----------------------------
    def load_data(self):
        data_file = self.data_dir / "tunisia_loan_data_train.csv"
        return pd.read_csv(data_file)

    def split_features_target(self, df):
        target_col = "TARGET"  # 1 = default, 0 = repaid
        id_col = "applicant_id"

        self.feature_cols = [c for c in df.columns if c not in [id_col, target_col]]

        X = df[self.feature_cols].copy()
        y = df[target_col].astype(int).copy()

        return X, y

    # -----------------------------
    # Affordability (Rule-Based)
    # -----------------------------
    def compute_affordability_band(self, df):
        proxies = []
        weights = []

        def safe_norm(s):
            mx = s.max()
            return (s / mx) if mx and mx > 0 else pd.Series(np.zeros(len(s)), index=s.index)

        if "avg_monthly_airtime_tnd" in df.columns:
            proxies.append(safe_norm(df["avg_monthly_airtime_tnd"].fillna(0)))
            weights.append(0.30)

        if "avg_monthly_remittance_tnd" in df.columns:
            proxies.append(safe_norm(df["avg_monthly_remittance_tnd"].fillna(0)))
            weights.append(0.20)

        if "requested_amount_tnd" in df.columns:
            proxies.append(safe_norm(df["requested_amount_tnd"].fillna(0)))
            weights.append(0.20)

        if "term_months" in df.columns:
            term = df["term_months"].fillna(df["term_months"].median())
            mx = term.max()
            inv = 1 - (term / mx) if mx and mx > 0 else pd.Series(np.zeros(len(term)), index=term.index)
            proxies.append(inv)
            weights.append(0.15)

        if "education" in df.columns:
            edu_map = {
                "Primary": 0.2,
                "Secondary": 0.4,
                "Some University": 0.6,
                "University": 0.8,
                "Graduate": 1.0,
            }
            proxies.append(df["education"].map(edu_map).fillna(0.4))
            weights.append(0.15)

        if not proxies:
            return pd.Series(["Unknown"] * len(df), index=df.index)

        w = np.array(weights, dtype=float)
        w = w / w.sum()

        score = np.zeros(len(df), dtype=float)
        for p, wi in zip(proxies, w):
            score += p.values * wi

        band = pd.cut(
            score,
            bins=[0, 0.33, 0.66, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

        return band.astype(str)

    # -----------------------------
    # Preprocessing
    # -----------------------------
    def build_preprocessor(self, X):
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_cols),
                ("cat", categorical_pipe, categorical_cols),
            ],
            remainder="drop",
        )

    # -----------------------------
    # Model + Calibration (Random Forest)
    # -----------------------------
    def build_model(self, class_weight=None):
        base = RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            min_samples_leaf=10,
            min_samples_split=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
            class_weight=class_weight,  # <-- cost sensitivity
        )

        # Calibrate probabilities: sigmoid (Platt scaling)
        # Note: calibration already does internal CV (cv=3)
        return CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=3,
        )

    def build_pipeline(self, X, class_weight=None):
        self.preprocessor = self.build_preprocessor(X)
        model = self.build_model(class_weight=class_weight)
        return Pipeline(
            steps=[
                ("preprocess", self.preprocessor),
                ("model", model),
            ]
        )

    # -----------------------------
    # Evaluation helpers
    # -----------------------------
    def pr_auc_score(self, y_true, y_proba):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        return auc(recall, precision)

    def find_cost_optimal_threshold(self, y_true, y_proba, fn_cost=10, fp_cost=1):
        """
        Find threshold minimizing cost = FN*fn_cost + FP*fp_cost
        Use thresholds from precision_recall_curve for finer search.
        """
        _, _, thresholds = precision_recall_curve(y_true, y_proba)
        if thresholds is None or len(thresholds) == 0:
            return 0.5, None

        best_t = 0.5
        best_cost = float("inf")

        for t in thresholds:
            y_pred = (y_proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            cost = fn * fn_cost + fp * fp_cost
            if cost < best_cost:
                best_cost = cost
                best_t = float(t)

        return best_t, float(best_cost)

    def compute_class_weight_from_cost(self, fn_cost=10, fp_cost=1):
        """
        Convert business costs to class_weight for training:
          - default class (1) weight ~ FN cost
          - non-default class (0) weight ~ FP cost
        This biases the model to reduce FN (missed defaults).
        """
        # scale to keep numbers small and readable
        return {0: float(fp_cost), 1: float(fn_cost)}

    # -----------------------------
    # Training / CV
    # -----------------------------
    def cross_validate(self, X, y, n_splits=5, fn_cost=10, fp_cost=1):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        class_weight = self.compute_class_weight_from_cost(fn_cost=fn_cost, fp_cost=fp_cost)

        aucs = []
        pr_aucs = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

            pipe = self.build_pipeline(X_tr, class_weight=class_weight)
            pipe.fit(X_tr, y_tr)

            y_proba = pipe.predict_proba(X_va)[:, 1]
            aucs.append(roc_auc_score(y_va, y_proba))
            pr_aucs.append(self.pr_auc_score(y_va, y_proba))

        return {
            "cv_auc_roc_mean": float(np.mean(aucs)),
            "cv_auc_roc_std": float(np.std(aucs)),
            "cv_pr_auc_mean": float(np.mean(pr_aucs)),
            "cv_pr_auc_std": float(np.std(pr_aucs)),
            "class_weight_used": class_weight,
        }

    def train_final(self, X_train, y_train, fn_cost=10, fp_cost=1):
        class_weight = self.compute_class_weight_from_cost(fn_cost=fn_cost, fp_cost=fp_cost)
        self.model = self.build_pipeline(X_train, class_weight=class_weight)
        self.model.fit(X_train, y_train)
        return self.model

    # -----------------------------
    # Eligibility policy
    # -----------------------------
    def eligibility_decision(self, default_proba, threshold):
        """
        Eligible if risk below threshold.
        (Add hard rules later: age>=18, min tenure, etc.)
        """
        return (default_proba < threshold).astype(int)

    # -----------------------------
    # Saving
    # -----------------------------
    def save_artifacts(self):
        package = {
            "pipeline": self.model,
            "feature_cols": self.feature_cols,
            "thresholds": self.thresholds,
        }

        model_file = self.models_dir / "credit_engine.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(package, f)

        metrics_file = self.reports_dir / "model_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(self.evaluation_results, f, indent=2)

        print(f"✓ Saved model package: {model_file}")
        print(f"✓ Saved metrics report: {metrics_file}")

    # -----------------------------
    # Run
    # -----------------------------
    def run(self):

        df = self.load_data()
        print(f"\nLoaded {len(df)} records")

        # affordability (rule-based)
        affordability_band = self.compute_affordability_band(df)
        print("\nAffordability (rule-based) distribution:")
        print(affordability_band.value_counts(dropna=False))

        X, y = self.split_features_target(df)

        fn_cost = 10  # missing a default
        fp_cost = 1   # rejecting a good client

        # reliable metrics via CV
        print("\nRunning 5-fold Stratified CV (Default Risk - Cost-Sensitive RF)...")
        cv_metrics = self.cross_validate(X, y, n_splits=5, fn_cost=fn_cost, fp_cost=fp_cost)
        print(f"✓ CV AUC-ROC: {cv_metrics['cv_auc_roc_mean']:.4f} ± {cv_metrics['cv_auc_roc_std']:.4f}")
        print(f"✓ CV PR-AUC:  {cv_metrics['cv_pr_auc_mean']:.4f} ± {cv_metrics['cv_pr_auc_std']:.4f}")
        print(f"✓ class_weight: {cv_metrics['class_weight_used']}")

        # train/val split for threshold optimization and final report
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        print(f"\nTrain size: {len(X_train)} | Val size: {len(X_val)}")

        # train final pipeline
        print("\nTraining calibrated default-risk model (Cost-Sensitive RF)...")
        self.train_final(X_train, y_train, fn_cost=fn_cost, fp_cost=fp_cost)

        # validate
        y_val_proba = self.model.predict_proba(X_val)[:, 1]
        auc_roc = roc_auc_score(y_val, y_val_proba)
        pr_auc = self.pr_auc_score(y_val, y_val_proba)

        # threshold optimization (business costs)
        best_t, best_cost = self.find_cost_optimal_threshold(
            y_val, y_val_proba, fn_cost=fn_cost, fp_cost=fp_cost
        )

        self.thresholds["eligibility_default_risk_threshold"] = best_t

        # final decisions at threshold
        y_val_pred_default = (y_val_proba >= best_t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred_default).ravel()

        recall_default = tp / (tp + fn) if (tp + fn) else 0.0
        precision_default = tp / (tp + fp) if (tp + fp) else 0.0

        # eligibility decisions derived
        y_val_eligible = self.eligibility_decision(y_val_proba, threshold=best_t)

        self.evaluation_results = {
            "default_risk": {
                "model_used": "Cost-Sensitive Random Forest (calibrated)",
                "holdout_auc_roc": float(auc_roc),
                "holdout_pr_auc": float(pr_auc),
                "threshold_cost_policy": {
                    "fn_cost": fn_cost,
                    "fp_cost": fp_cost,
                    "best_threshold": float(best_t),
                    "expected_cost": best_cost,
                },
                "confusion_matrix_at_threshold": {
                    "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
                },
                "precision_default": float(precision_default),
                "recall_default": float(recall_default),
                "classification_report": classification_report(
                    y_val, y_val_pred_default, output_dict=True
                ),
            },
            "cv": cv_metrics,
            "eligibility": {
                "definition": "eligible = P(default) < threshold",
                "threshold": float(best_t),
                "eligible_rate_on_val": float(y_val_eligible.mean()),
            },
            "affordability": {
                "type": "rule_based_proxy",
                "distribution": affordability_band.value_counts(dropna=False).to_dict()
            }
        }

        print("\nValidation Summary (Default Risk):")
        print(f"✓ Holdout AUC-ROC: {auc_roc:.4f}")
        print(f"✓ Holdout PR-AUC:  {pr_auc:.4f}")
        print(f"✓ Cost-optimal threshold: {best_t:.4f}")
        print(f"✓ Recall(default): {recall_default:.4f}")
        print(f"✓ Precision(default): {precision_default:.4f}")

        # save
        print("\nSaving artifacts...")
        self.save_artifacts()

        print("\n" + "=" * 80)
        print("STEP C COMPLETE")
        print("=" * 80)

        return self.model, self.evaluation_results


if __name__ == "__main__":
    engine = UnifiedCreditEngine()
    model, results = engine.run()
