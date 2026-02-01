"""
Step C: Unified Credit Engine - Eligibility, Default Risk, and Affordability Models
"""
import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, confusion_matrix,
    classification_report, f1_score, balanced_accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class UnifiedCreditEngine:
    """Unified AI credit scoring engine with three outputs"""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'outputs' / 'datasets'
        self.models_dir = self.base_dir / 'outputs' / 'models'
        self.reports_dir = self.base_dir / 'reports'
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.encoders = {}
        self.scaler = None
        self.feature_names = None
        self.evaluation_results = {}
    
    def load_data(self):
        """Load safe training dataset"""
        data_file = self.data_dir / 'tunisia_loan_data_train.csv'
        df = pd.read_csv(data_file)
        return df
    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        
        # Separate features from target
        target_col = 'TARGET'
        id_col = 'applicant_id'
        
        # Exclude ID and TARGET
        feature_cols = [col for col in df.columns if col not in [id_col, target_col]]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        ids = df[id_col].copy()
        
        # Handle categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown')
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le
        
        # Handle missing values in numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            X[col] = X[col].fillna(X[col].median())
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y, ids
    
    def create_affordability_labels(self, df):
        """Create affordability bands from alternative indicators"""
        
        # Use multiple proxies to estimate income bands
        # Avoid direct leakage by combining multiple signals
        
        # Proxies for income estimation (weighted combination)
        proxies = []
        weights = []
        
        if 'avg_monthly_airtime_tnd' in df.columns:
            # Airtime spending proxy (normalize)
            airtime_norm = df['avg_monthly_airtime_tnd'] / df['avg_monthly_airtime_tnd'].max()
            proxies.append(airtime_norm)
            weights.append(0.3)
        
        if 'avg_monthly_remittance_tnd' in df.columns:
            # Remittance proxy
            remit_norm = df['avg_monthly_remittance_tnd'] / (df['avg_monthly_remittance_tnd'].max() + 1)
            proxies.append(remit_norm)
            weights.append(0.2)
        
        if 'requested_amount_tnd' in df.columns:
            # Loan amount requested (normalized)
            loan_norm = df['requested_amount_tnd'] / df['requested_amount_tnd'].max()
            proxies.append(loan_norm)
            weights.append(0.2)
        
        if 'term_months' in df.columns:
            # Longer terms might indicate lower income
            term_norm = 1 - (df['term_months'] / df['term_months'].max())
            proxies.append(term_norm)
            weights.append(0.15)
        
        if 'education' in df.columns:
            # Education level proxy
            edu_map = {'Primary': 0.2, 'Secondary': 0.4, 'Some University': 0.6, 
                      'University': 0.8, 'Graduate': 1.0}
            edu_norm = df['education'].map(edu_map).fillna(0.4)
            proxies.append(edu_norm)
            weights.append(0.15)
        
        # Weighted combination
        weights = np.array(weights) / np.sum(weights)
        income_proxy = np.zeros(len(df))
        
        for proxy, weight in zip(proxies, weights):
            income_proxy += proxy.values * weight
        
        # Create affordability bands
        # Low: 0-33%, Medium: 33-66%, High: 66-100%
        affordability_band = pd.cut(
            income_proxy,
            bins=[0, 0.33, 0.66, 1.0],
            labels=['Low', 'Medium', 'High'],
            include_lowest=True
        )
        
        return affordability_band
    
    def train_eligibility_model(self, X_train, y_train, X_val, y_val):
        """Train eligibility model (predicts good borrower vs default)"""
        
        print("\n[C1] Training Eligibility Model...")
        
        # Eligibility: predict if applicant should be considered eligible
        # Target: 0 = default (not eligible), 1 = repaid (eligible)
        # Invert TARGET so 1 = eligible
        y_train_eligibility = 1 - y_train
        y_val_eligibility = 1 - y_val
        
        # Handle class imbalance with class weights
        n_eligible = y_train_eligibility.sum()
        n_not_eligible = len(y_train_eligibility) - n_eligible
        scale_pos_weight = n_not_eligible / n_eligible if n_eligible > 0 else 1
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train_eligibility)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Evaluation
        auc_score = roc_auc_score(y_val_eligibility, y_pred_proba)
        
        # Find optimal threshold
        thresholds = np.arange(0.3, 0.8, 0.05)
        best_threshold = 0.5
        best_f1 = 0
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            f1 = f1_score(y_val_eligibility, y_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
        
        self.models['eligibility'] = model
        self.evaluation_results['eligibility'] = {
            'auc_roc': float(auc_score),
            'best_threshold': float(best_threshold),
            'best_f1': float(best_f1),
            'classification_report': classification_report(
                y_val_eligibility, y_pred_optimal, output_dict=True
            )
        }
        
        print(f"✓ Eligibility Model AUC-ROC: {auc_score:.4f}")
        print(f"✓ Optimal Threshold: {best_threshold:.2f}")
        print(f"✓ F1 Score at optimal threshold: {best_f1:.4f}")
        
        return model
    
    def train_default_risk_model(self, X_train, y_train, X_val, y_val):
        """Train default risk model with imbalance handling"""
        
        print("\n[C2] Training Default Risk Model...")
        
        # Handle class imbalance
        # 1. Class weights
        n_default = y_train.sum()
        n_no_default = len(y_train) - n_default
        scale_pos_weight = n_no_default / n_default if n_default > 0 else 1
        
        print(f"Class distribution - Default: {n_default}, No Default: {n_no_default}")
        print(f"Using class_weight='balanced' for imbalance handling")
        
        # Use class_weight='balanced' instead of SMOTE
        X_train_resampled = X_train
        y_train_resampled = y_train
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_resampled, y_train_resampled)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Evaluation
        auc_roc = roc_auc_score(y_val, y_pred_proba)
        
        # PR-AUC
        precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Threshold optimization based on business costs
        # FN (miss default) cost is 10x FP (reject good customer) cost
        fn_cost = 10
        fp_cost = 1
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_cost = float('inf')
        
        for thresh in thresholds:
            y_pred = (y_pred_proba >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
            total_cost = fn * fn_cost + fp * fp_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_threshold = thresh
        
        # Final predictions with optimal threshold
        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred_optimal).ravel()
        
        self.models['default_risk'] = model
        self.evaluation_results['default_risk'] = {
            'auc_roc': float(auc_roc),
            'pr_auc': float(pr_auc),
            'best_threshold': float(best_threshold),
            'confusion_matrix': {
                'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            },
            'recall_default_class': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
            'precision_default_class': float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
            'expected_cost': float(best_cost),
            'classification_report': classification_report(
                y_val, y_pred_optimal, output_dict=True
            )
        }
        
        print(f"✓ Default Risk Model AUC-ROC: {auc_roc:.4f}")
        print(f"✓ PR-AUC: {pr_auc:.4f}")
        print(f"✓ Optimal Threshold (cost-based): {best_threshold:.2f}")
        print(f"✓ Recall for default class: {self.evaluation_results['default_risk']['recall_default_class']:.4f}")
        print(f"✓ Expected cost: {best_cost:.0f}")
        
        return model
    
    def train_affordability_model(self, X_train, y_train_afford, X_val, y_val_afford):
        """Train affordability band classification model"""
        
        print("\n[C3] Training Affordability Model...")
        
        # Encode affordability labels
        le_afford = LabelEncoder()
        y_train_encoded = le_afford.fit_transform(y_train_afford)
        y_val_encoded = le_afford.transform(y_val_afford)
        self.encoders['affordability'] = le_afford
        
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train_encoded)
        
        # Predictions
        y_pred = model.predict(X_val)
        
        # Evaluation
        macro_f1 = f1_score(y_val_encoded, y_pred, average='macro')
        balanced_acc = balanced_accuracy_score(y_val_encoded, y_pred)
        
        self.models['affordability'] = model
        self.evaluation_results['affordability'] = {
            'macro_f1': float(macro_f1),
            'balanced_accuracy': float(balanced_acc),
            'classes': le_afford.classes_.tolist(),
            'classification_report': classification_report(
                y_val_encoded, y_pred, 
                target_names=le_afford.classes_.tolist(),
                output_dict=True
            )
        }
        
        print(f"✓ Affordability Model Macro F1: {macro_f1:.4f}")
        print(f"✓ Balanced Accuracy: {balanced_acc:.4f}")
        
        return model
    
    def save_models(self):
        """Save all models and encoders"""
        
        engine_package = {
            'models': self.models,
            'encoders': self.encoders,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        model_file = self.models_dir / 'credit_engine.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(engine_package, f)
        
        print(f"✓ Saved unified credit engine to {model_file}")
    
    def save_evaluation_report(self):
        """Save evaluation metrics"""
        
        # JSON report
        json_file = self.reports_dir / 'model_metrics.json'
        with open(json_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"✓ Saved evaluation metrics to {json_file}")
    
    def run(self):
        """Execute complete modeling pipeline"""
        
        print("=" * 80)
        print("STEP C: UNIFIED CREDIT ENGINE MODELING")
        print("=" * 80)
        
        # Load data
        print("\nLoading training data...")
        df = self.load_data()
        print(f"Loaded {len(df)} records")
        
        # Prepare features
        print("\nPreparing features...")
        X, y, ids = self.prepare_features(df)
        print(f"Feature matrix shape: {X.shape}")
        
        # Create affordability labels
        print("\nCreating affordability labels...")
        affordability_bands = self.create_affordability_labels(df)
        print(f"Affordability distribution:\n{affordability_bands.value_counts()}")
        
        # Train-validation split
        print("\nSplitting data...")
        X_train, X_val, y_train, y_val, afford_train, afford_val = train_test_split(
            X, y, affordability_bands,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Scale features
        print("\nScaling features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Convert back to DataFrame for XGBoost
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X.columns)
        
        # Train models
        self.train_eligibility_model(X_train_scaled, y_train, X_val_scaled, y_val)
        self.train_default_risk_model(X_train_scaled, y_train, X_val_scaled, y_val)
        self.train_affordability_model(X_train_scaled, afford_train, X_val_scaled, afford_val)
        
        # Save models
        print("\nSaving models...")
        self.save_models()
        self.save_evaluation_report()
        
        print("\n" + "=" * 80)
        print("STEP C COMPLETE")
        print("=" * 80)
        print("\nModel Performance Summary:")
        print(f"- Eligibility AUC-ROC: {self.evaluation_results['eligibility']['auc_roc']:.4f}")
        print(f"- Default Risk AUC-ROC: {self.evaluation_results['default_risk']['auc_roc']:.4f}")
        print(f"- Affordability Macro F1: {self.evaluation_results['affordability']['macro_f1']:.4f}")
        
        return self.models, self.evaluation_results


if __name__ == '__main__':
    engine = UnifiedCreditEngine()
    models, results = engine.run()
