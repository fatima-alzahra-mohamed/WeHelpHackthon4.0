# Data Quality Analysis Report

## Summary
- **Total Records**: 10,000
- **Total Features**: 28
- **Duplicate Rows**: 0
- **Duplicate Applicant IDs**: 0

## Missing Values
- **marketplace_seller_rating**: 8030 (80.3%)
- **product_cap_tnd**: 5113 (51.13%)

## Target Leakage Analysis

### High-Risk Features (Removed)
- No high-risk features detected

### Medium-Risk Features (Monitored)
- **payment_on_time_rate**: Moderate correlation with TARGET (>0.4) (r=-0.578)
- **transaction_regularity_score**: Moderate correlation with TARGET (>0.4) (r=-0.458)
- **marketplace_seller_rating**: Moderate correlation with TARGET (>0.4) (r=-0.548)

## Affordability Leakage Check
- **Note**: Affordability will be modeled as income bands.
- **Potential Issues**:
  - avg_monthly_airtime_tnd: avg_monthly_airtime_tnd is used as input feature, ensure it does not directly encode income bands
  - avg_monthly_remittance_tnd: avg_monthly_remittance_tnd is used as input feature, ensure it does not directly encode income bands

## Distribution Checks by Loan Category

### Business
- Count: 1,045
- Default Rate: 9.38%
- Avg Loan Amount: 46,113.72 TND
- Avg Term: 36 months

### Vehicle
- Count: 459
- Default Rate: 8.93%
- Avg Loan Amount: 40,932.10 TND
- Avg Term: 37 months

### Housing
- Count: 3,705
- Default Rate: 5.10%
- Avg Loan Amount: 216,408.18 TND
- Avg Term: 36 months

### Consumer
- Count: 4,428
- Default Rate: 9.69%
- Avg Loan Amount: 29,818.44 TND
- Avg Term: 36 months

### Corporate
- Count: 363
- Default Rate: 7.99%
- Avg Loan Amount: 86,126.31 TND
- Avg Term: 36 months

## Columns Removed for Safe Training
- product_cap_tnd
- max_financing_ratio
- requires_appraisal_flag
- max_term_months

## Conclusion
- Data quality checks completed
- Target leakage analysis completed
- Safe training dataset created (tunisia_loan_data_train.csv)
- Ready for modeling phase