# Tunisia AI Credit Scoring Demo
## Arab Tunisian Bank (ATB) - Hackathon Submission

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**AI-powered credit scoring using alternative financial indicators for the Tunisian market**

---

## 🎯 Project Overview

This demonstration builds a unified AI credit engine for Arab Tunisian Bank (ATB) that estimates **loan eligibility**, **default risk**, and **affordability** using alternative financial data sources available in Tunisia.

### Key Innovation

Traditional credit scoring relies on formal credit history, which many Tunisians lack. Our system leverages:
- 📱 **Telecom data**: Payment patterns, usage behavior, tenure
- 💰 **Mobile money**: D17/e-dinar transaction activity
- 🛒 **Marketplace activity**: E-commerce ratings and history
- 💸 **Remittances**: International transfers as income signals
- 📍 **Location stability**: Geographic consistency from mobile usage

### Business Value

- **Financial Inclusion**: Serve underbanked segments without traditional credit history
- **Risk Management**: Improved default prediction using alternative signals
- **Operational Efficiency**: Automated eligibility and affordability assessment
- **Regulatory Compliance**: Transparent, explainable AI decisions

---

## 📁 Repository Structure

```
atb_credit_scoring/
├── run_complete_pipeline.py          # Single-command end-to-end runner
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── .gitignore                        # Git ignore rules
│
├── data/
│   └── raw/                          # Place Home Credit CSV files here (optional)
│       ├── application_train.csv     # Main applications table
│       └── ...                       # Other Home Credit tables (optional)
│
├── scripts/                          # Core pipeline scripts
│   ├── step_a_data_generation.py    # Tunisia dataset generation
│   ├── step_b_data_quality.py       # Quality audit & leakage checks
│   ├── step_c_modeling.py           # Unified credit engine training
│   ├── step_d_explainability.py     # SHAP explanations
│   └── step_e_reporting.py          # Graphics & final report
│
├── outputs/                          # Generated outputs
│   ├── datasets/                     # Generated datasets
│   │   ├── tunisia_loan_data.csv
│   │   ├── tunisia_loan_data_train.csv
│   │   ├── tunisia_loan_data_dictionary.csv
│   │   └── feature_mapping_reference.csv
│   ├── models/
│   │   └── credit_engine.pkl        # Trained models
│   └── figures/                      # Visualizations
│       ├── system_schema.png
│       ├── data_snapshot.png
│       ├── target_distribution.png
│       ├── loan_amount_by_category.png
│       ├── correlation_heatmap.png
│       ├── shap_global_summary.png
│       └── shap_local_example.png
│
└── reports/                          # Analysis reports
    ├── FEATURE_MAPPING.md
    ├── data_generation_summary.json
    ├── data_quality_analysis.json
    ├── data_quality_report.md
    ├── model_metrics.json
    ├── explainability_notes.md
    └── ATB_Demo_Report.md            # Main demo report
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Home Credit dataset files

### Installation

```bash
# Clone or download the repository
cd atb_credit_scoring

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Complete Pipeline

**One command to generate everything:**

```bash
python3 run_complete_pipeline.py
```

This will:
1. ✓ Generate Tunisia-style loan dataset (or load Home Credit data if available)
2. ✓ Perform data quality audit and leakage checks
3. ✓ Train unified credit engine (3 models)
4. ✓ Generate SHAP explanations
5. ✓ Create visualizations and final report

**Estimated runtime**: 2-5 minutes depending on hardware

---

## 📊 Data Setup

### Option 1: Automatic Synthetic Data (Default)

The pipeline automatically generates a synthetic Tunisia dataset based on Home Credit structure. **No data files needed**.

### Option 2: Using Real Home Credit Data

1. Download Home Credit Default Risk dataset from [Kaggle](https://www.kaggle.com/c/home-credit-default-risk/data)
2. Place files in `data/raw/`:
   ```
   data/raw/
   ├── application_train.csv  (required)
   ├── bureau.csv            (optional)
   ├── previous_application.csv (optional)
   └── ...
   ```
3. Run pipeline normally - it will detect and use the real data

---

## 🎯 What Gets Generated

### Datasets
- **tunisia_loan_data.csv**: Complete Tunisia loan dataset with alternative indicators
- **tunisia_loan_data_train.csv**: Safe training dataset (leakage-free)
- **tunisia_loan_data_dictionary.csv**: Data dictionary with descriptions
- **feature_mapping_reference.csv**: Home Credit → Tunisia feature mapping

### Models
- **credit_engine.pkl**: Unified model package with:
  - Eligibility model (predicts good borrower vs default)
  - Default risk model (predicts probability of default)
  - Affordability model (classifies income bands: Low/Medium/High)

### Reports
- **ATB_Demo_Report.md**: Executive summary and technical details
- **model_metrics.json**: Detailed performance metrics
- **explainability_notes.md**: SHAP analysis and insights
- **data_quality_report.md**: Data quality audit results

### Visualizations
- System architecture diagram
- Data snapshot examples
- Target distribution (class imbalance)
- Loan amounts by category
- Feature correlation heatmap
- SHAP global importance
- SHAP local explanation example

---

## 🔍 Model Performance

The unified credit engine provides three outputs:

| Model | Metric | Performance | Purpose |
|-------|--------|-------------|---------|
| **Eligibility** | AUC-ROC | ~0.85-0.90 | Should ATB consider this applicant? |
| **Default Risk** | AUC-ROC | ~0.75-0.85 | What's the probability of default? |
| **Default Risk** | PR-AUC | ~0.40-0.60 | Precision-recall (imbalanced classes) |
| **Affordability** | Macro F1 | ~0.65-0.75 | Which income band does applicant belong to? |

*Note: Actual performance depends on generated data and may vary between runs*

### Class Imbalance Handling
- ✓ SMOTE oversampling on training set
- ✓ Class weights in models
- ✓ Cost-sensitive threshold optimization (FN cost = 10x FP cost)

---

## 📝 Tunisia Market Features

### ATB Loan Products

| Product | Max LTV | Max Term | Amount Cap | Appraisal Required |
|---------|---------|----------|------------|-------------------|
| Housing | 80% | 300 months | None | Yes |
| Vehicle | 85% | 84 months | 150,000 TND | Yes |
| Consumer | 100% | 60 months | 50,000 TND | No |
| Business | 70% | 120 months | None | Yes |
| Corporate | 75% | 180 months | None | Yes |

### Alternative Data Sources (Tunisia Context)

**Telecom Providers**
- Ooredoo Tunisia
- Orange Tunisia  
- Tunisie Telecom

**Mobile Money**
- D17 (ATB's mobile money platform)
- e-dinar

**E-commerce Platforms**
- Jumia Tunisia
- Tunisie Net
- Local marketplaces

**Remittances**
- International transfers from Europe (France, Italy, Germany)
- Gulf countries (Saudi Arabia, UAE, Qatar)

**Geography**
- Coverage across all 24 Tunisia governorates
- Urban/rural representation

---

## 🛡️ Data Quality & Leakage Prevention

### Automated Checks

✓ **No Target Leakage**: High-correlation features automatically identified and removed  
✓ **No Post-Outcome Features**: Only pre-decision data used  
✓ **No Affordability Leakage**: Income bands derived from multiple independent proxies  
✓ **Metadata Isolation**: Product constraints not used in training  
✓ **Duplicate Detection**: Automatic duplicate removal  
✓ **Missing Value Handling**: Median imputation for numeric, mode for categorical

### Leakage Audit Report

Every run generates `data_quality_report.md` with:
- Correlation analysis with TARGET
- Post-outcome feature detection
- Feature leakage flags
- Distribution sanity checks
- Safe training dataset creation

---

## 🔬 Explainable AI

### SHAP (SHapley Additive exPlanations)

**Global Explanations**
- Which features drive predictions across all applicants
- Feature importance ranking
- Interaction effects

**Local Explanations**  
- Why did a specific applicant get their score?
- Which features pushed risk up/down?
- Transparent communication with customers

### Key Drivers (Typical)

1. **Payment on-time rate**: Historical payment behavior
2. **Transaction regularity**: Financial stability
3. **Months with provider**: Account tenure/stability
4. **Location consistency**: Geographic stability
5. **Mobile money activity**: Financial engagement

See `reports/explainability_notes.md` for detailed analysis.

---

## 🧪 Running Individual Steps

You can run pipeline steps independently for development:

```bash
# Step A: Data generation only
python3 scripts/step_a_data_generation.py

# Step B: Data quality audit only
python3 scripts/step_b_data_quality.py

# Step C: Modeling only
python3 scripts/step_c_modeling.py

# Step D: Explainability only
python3 scripts/step_d_explainability.py

# Step E: Reporting only
python3 scripts/step_e_reporting.py
```

**Note**: Steps must be run in order as each depends on previous outputs.

---

## 📈 Demo Flow for Hackathon

### 1. Executive Pitch (2 minutes)
- Show **ATB_Demo_Report.md** executive summary
- Highlight alternative data innovation
- Present model performance metrics

### 2. Technical Deep-Dive (3 minutes)
- Show **system_schema.png** architecture
- Explain feature groups and 3 outputs
- Walk through **data_snapshot.png** example records

### 3. Model Performance (2 minutes)
- Show **target_distribution.png** (class imbalance)
- Present **model_metrics.json** results
- Explain threshold optimization

### 4. Explainability Demo (2 minutes)
- Show **shap_global_summary.png** (top drivers)
- Walk through **shap_local_example.png** (specific applicant)
- Demonstrate transparency for regulatory compliance

### 5. Q&A (1 minute)
- Reference **data_quality_report.md** for leakage prevention
- Show **FEATURE_MAPPING.md** for Tunisia market alignment

---

## 🔧 Customization

### Adjusting Sample Size

Edit `run_complete_pipeline.py`:
```python
tunisia_df, data_dict, summary = generator.run(n_samples=20000)  # Change from 10000
```

### Modifying Loan Categories

Edit `scripts/step_a_data_generation.py`:
```python
def assign_loan_category(credit, occ, org):
    # Add your custom logic here
```

### Tuning Model Hyperparameters

Edit `scripts/step_c_modeling.py`:
```python
model = XGBClassifier(
    n_estimators=300,      # Increase trees
    max_depth=7,           # Deeper trees
    learning_rate=0.03,    # Lower learning rate
    # ...
)
```

---

## 📚 Key Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML algorithms and preprocessing
- **xgboost**: Gradient boosting models
- **imbalanced-learn**: SMOTE for class imbalance
- **shap**: Model explainability
- **matplotlib/seaborn**: Visualizations

See `requirements.txt` for complete list.

---

## 🤝 Contributing

This is a hackathon demo project. For production deployment:

1. **Data Partnerships**: Establish MoUs with telecom providers, mobile money operators
2. **Real Data Integration**: Replace synthetic data with actual ATB customer data
3. **Model Monitoring**: Implement drift detection and retraining pipelines
4. **A/B Testing**: Shadow scoring before full deployment
5. **Regulatory Review**: Ensure compliance with BCT (Central Bank of Tunisia) requirements

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Team

[Add your team information here]

---

## 📞 Contact

For questions or collaboration opportunities:
- Email: [your-email]
- LinkedIn: [your-linkedin]
- GitHub: [your-github]

---

## 🙏 Acknowledgments

- **Home Credit** for providing the base dataset structure
- **Anthropic Claude** for AI assistance in development
- **ATB** for inspiration and Tunisia market context
- **Open source community** for excellent ML libraries

---

## 📊 Suggested Commit Messages

```bash
git commit -m "feat: initial project structure and requirements"
git commit -m "feat: add Tunisia data generation pipeline (Step A)"
git commit -m "feat: add data quality audit and leakage checks (Step B)"
git commit -m "feat: implement unified credit engine with 3 models (Step C)"
git commit -m "feat: add SHAP explainability (Step D)"
git commit -m "feat: add graphics and final reporting (Step E)"
git commit -m "docs: add comprehensive README and documentation"
git commit -m "chore: add .gitignore and requirements.txt"
```

---

**Built with ❤️ for Tunisia's financial inclusion**
