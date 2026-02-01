# Tunisia AI Credit Scoring Demo
## Arab Tunisian Bank (ATB) - Hackathon Submission

---

## Executive Summary

This demonstration showcases an AI-powered credit scoring system for Arab Tunisian Bank (ATB) that leverages **alternative financial indicators** to assess loan eligibility, default risk, and affordability for Tunisian borrowers.

### Key Innovation
Traditional credit scoring relies heavily on formal credit history, which many Tunisians lack. Our system uses:
- **Telecom data**: Payment history, usage patterns, tenure
- **Mobile money**: Transaction frequency and volumes (D17, e-dinar)
- **Marketplace activity**: E-commerce ratings and transaction history
- **Remittance flows**: International transfers as income stability signal
- **Location stability**: Geographic consistency from mobile usage

### Business Impact
- **Dataset**: 10,000 loan applications analyzed
- **Default Rate**: 7.9% (realistic Tunisia market conditions)
- **Average Loan**: 103,207 TND
- **Product Coverage**: Housing, Vehicle, Consumer, Business, Corporate

---

## Technical Performance

### Model 1: Eligibility Assessment
- **AUC-ROC**: 0.9803
- **Optimal Threshold**: 0.35
- **F1 Score**: 0.9822
- **Purpose**: Determine if applicant should be considered for loan product

### Model 2: Default Risk Prediction
- **AUC-ROC**: 0.9804
- **PR-AUC**: 0.8650
- **Recall (Default Class)**: 0.8917
- **Optimal Threshold**: 0.35 (cost-optimized)
- **Class Imbalance Handling**: SMOTE + class weights + threshold optimization
- **Business Cost Optimization**: FN cost = 10x FP cost

### Model 3: Affordability Classification
- **Macro F1 Score**: 0.8985
- **Balanced Accuracy**: 0.9480
- **Classes**: High, Low, Medium
- **Purpose**: Estimate appropriate loan amounts based on income proxies

---

## Dataset Characteristics

### Loan Product Distribution
- **Consumer**: 4,428 loans (44.3%)
- **Housing**: 3,705 loans (37.0%)
- **Business**: 1,045 loans (10.4%)
- **Vehicle**: 459 loans (4.6%)
- **Corporate**: 363 loans (3.6%)

### Alternative Data Coverage
- **Mobile Money Users**: 6,886 (68.9%)
- **Remittance Recipients**: 2,002 (20.0%)
- **Marketplace Sellers**: 1,970 (19.7%)

---

## Model Explainability

Our system provides transparent, explainable decisions using SHAP (SHapley Additive exPlanations):

- **Global Explanations**: Identify which features drive predictions across all applicants
- **Local Explanations**: Show exactly why a specific applicant received their score
- **Regulatory Compliance**: Supports transparency requirements for AI lending

See `explainability_notes.md` for detailed analysis.

---

## Tunisia Market Relevance

### ATB Product Alignment
The system handles all major ATB loan categories with product-specific constraints:

| Product | Max LTV | Max Term | Cap (TND) | Appraisal |
|---------|---------|----------|-----------|-----------|
| Housing | 80% | 300 mo | None | Yes |
| Vehicle | 85% | 84 mo | 150,000 | Yes |
| Consumer | 100% | 60 mo | 50,000 | No |
| Business | 70% | 120 mo | None | Yes |
| Corporate | 75% | 180 mo | None | Yes |

### Data Sources (Tunisia Context)
- **Telecom**: Ooredoo, Orange, Tunisie Telecom
- **Mobile Money**: D17 (ATB's mobile money), e-dinar
- **E-commerce**: Jumia, Tunisie Net, local marketplaces
- **Remittances**: International transfers (Europe, Gulf countries)
- **Governorates**: All 24 Tunisia governorates represented

---

## Leakage Prevention & Data Quality

Rigorous data quality checks ensure model integrity:

- ✓ **No Target Leakage**: All high-correlation features removed
- ✓ **No Post-Outcome Features**: Features only use pre-decision data
- ✓ **No Affordability Leakage**: Income bands derived from multiple proxies
- ✓ **Metadata Isolation**: Product constraints not used in training

See `data_quality_report.md` for complete audit results.

---

## Implementation Roadmap

### Phase 1: Pilot (3 months)
1. Deploy shadow scoring alongside existing underwriting
2. Collect feedback from loan officers
3. Validate predictions against actual outcomes

### Phase 2: Integration (6 months)
1. Establish data partnerships (telecom, mobile money)
2. Build real-time API integrations
3. Train loan officers on AI-assisted decisioning

### Phase 3: Scale (12 months)
1. Full production deployment
2. Expand to underbanked segments
3. Continuous model monitoring and retraining

---

## Risk Mitigation

### Model Risks
- **Data drift**: Monitor feature distributions quarterly
- **Concept drift**: Retrain models semi-annually
- **Bias**: Regular fairness audits across demographics

### Operational Risks
- **Data quality**: Automated validation pipelines
- **System downtime**: Fallback to manual underwriting
- **Regulatory changes**: Flexible architecture for compliance

---

## Conclusion

This demonstration proves the viability of AI-powered credit scoring using alternative data for the Tunisian market. The system:

✓ Achieves strong predictive performance without traditional credit bureaus
✓ Provides transparent, explainable decisions
✓ Aligns with ATB's product portfolio and business processes
✓ Leverages Tunisia-specific data sources
✓ Follows best practices for data quality and leakage prevention

**Next Steps**: Partner with ATB to pilot on real data and refine for production deployment.

---

## Appendices

### Generated Artifacts
- `outputs/datasets/tunisia_loan_data.csv` - Full dataset
- `outputs/datasets/tunisia_loan_data_train.csv` - Safe training data
- `outputs/models/credit_engine.pkl` - Trained models
- `reports/model_metrics.json` - Detailed metrics
- `reports/explainability_notes.md` - SHAP analysis
- `outputs/figures/` - All visualizations

### Contact
For questions or collaboration: [Your Contact Info]

---

**Report Generated**: 2026-02-01 01:32:28