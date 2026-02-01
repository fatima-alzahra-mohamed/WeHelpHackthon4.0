# Explainability Analysis - Default Risk Model

## Overview
This report explains how the default risk model makes predictions using **model-native feature importance**.

## Global Feature Importance
Top features driving model predictions (by global importance):

1. **payment_on_time_rate**: 0.4804
2. **transaction_regularity_score**: 0.2989
3. **marketplace_seller_rating**: 0.0327
4. **dispute_rate**: 0.0272
5. **requested_amount_tnd**: 0.0224
6. **business_hours_data_usage**: 0.0171
7. **months_with_provider**: 0.0148
8. **avg_monthly_airtime_tnd**: 0.0141
9. **term_months**: 0.0136
10. **peer_transfer_network_size**: 0.0116

## How to Interpret
- **Higher importance** means the model relies more on that feature overall.
- Importance does **not** indicate direction (increase/decrease risk); it indicates influence strength.

## Local Explanations (Proxy)
The local chart uses a simple proxy contribution:
- Contribution ≈ **standardized feature value (z-score)** × **global importance**
- This is not SHAP, but it gives an intuitive demo-friendly explanation of what is pushing a specific case.

## Recommendations
1. Add true SHAP later for stronger, direction-aware explanations (requires `shap` dependency).
2. Monitor importance drift over time (data distribution changes).
3. Validate key features with ATB domain experts for business sanity checks.