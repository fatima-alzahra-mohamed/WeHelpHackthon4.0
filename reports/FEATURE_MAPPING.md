# Tunisia Alternative Data Feature Mapping

**Purpose**: Map Home Credit dataset features to Tunisia-relevant alternative financial indicators

**LLM Role**: This mapping was designed with LLM assistance to ensure Tunisia market relevance


## Identity/Demographics

### applicant_id
- **Source**: SK_ID_CURR
- **Description**: Unique applicant identifier
- **Transformation**: direct_copy

### age
- **Source**: DAYS_BIRTH
- **Description**: Age in years derived from days
- **Transformation**: abs(DAYS_BIRTH) / 365.25

### gender
- **Source**: CODE_GENDER
- **Description**: Gender (M/F)
- **Transformation**: direct_copy

### governorate
- **Source**: REGION_RATING_CLIENT
- **Description**: Tunisia governorate (mapped from region rating)
- **Transformation**: map_to_tunisia_governorates

### education
- **Source**: NAME_EDUCATION_TYPE
- **Description**: Education level
- **Transformation**: map_to_tunisia_education


## Alternative Indicators - Payment Behavior

### payment_on_time_rate
- **Source**: EXT_SOURCE_2
- **Description**: Historical on-time payment rate from mobile money/telecom bills
- **Transformation**: EXT_SOURCE_2 with noise
- **Tunisia Context**: Aggregated from D17, Ooredoo, Orange bill payments

### transaction_regularity_score
- **Source**: EXT_SOURCE_3
- **Description**: Consistency of financial transactions over time
- **Transformation**: EXT_SOURCE_3 with scaling
- **Tunisia Context**: Monthly transaction pattern stability

### dispute_rate
- **Source**: AMT_REQ_CREDIT_BUREAU_QRT
- **Description**: Rate of payment disputes or chargebacks
- **Transformation**: normalized_inverse(AMT_REQ_CREDIT_BUREAU_QRT)
- **Tunisia Context**: Disputes from e-commerce, mobile services


## Alternative Indicators - Telecom

### months_with_provider
- **Source**: DAYS_EMPLOYED
- **Description**: Tenure with current telecom provider
- **Transformation**: abs(DAYS_EMPLOYED) / 30, capped at 120
- **Tunisia Context**: Ooredoo, Orange, Tunisie Telecom tenure

### avg_monthly_airtime_tnd
- **Source**: AMT_ANNUITY
- **Description**: Average monthly mobile airtime spending
- **Transformation**: AMT_ANNUITY * 0.02 with noise
- **Tunisia Context**: Proxy for spending capacity

### mobile_topup_frequency
- **Source**: derived
- **Description**: Number of mobile top-ups per month
- **Transformation**: random_4_to_20_based_on_income
- **Tunisia Context**: Higher frequency = regular income flow

### business_hours_data_usage
- **Source**: OCCUPATION_TYPE
- **Description**: Percentage of data used during business hours
- **Transformation**: occupation_based_pattern
- **Tunisia Context**: Employment stability indicator


## Alternative Indicators - Mobile Money

### mobile_money_account
- **Source**: derived
- **Description**: Has active mobile money account (D17, e-dinar)
- **Transformation**: probabilistic_based_on_age_education
- **Tunisia Context**: D17 (ATB mobile money) or e-dinar usage

### avg_monthly_transactions
- **Source**: derived
- **Description**: Average monthly mobile money transactions
- **Transformation**: income_and_education_based
- **Tunisia Context**: Financial activity level


## Alternative Indicators - Marketplace

### marketplace_seller_rating
- **Source**: EXT_SOURCE_2
- **Description**: Seller rating on e-commerce platforms (if applicable)
- **Transformation**: EXT_SOURCE_2 * 5, nullable for non-sellers
- **Tunisia Context**: Jumia, Tunisie Net, local marketplaces

### completed_transactions
- **Source**: derived
- **Description**: Number of completed marketplace transactions
- **Transformation**: random_for_sellers_only
- **Tunisia Context**: E-commerce trust indicator


## Alternative Indicators - Remittances/Social

### receives_remittances
- **Source**: derived
- **Description**: Receives international remittances
- **Transformation**: probabilistic_20_percent
- **Tunisia Context**: Important income source for ~20% Tunisians

### avg_monthly_remittance_tnd
- **Source**: AMT_INCOME_TOTAL
- **Description**: Average monthly remittance amount
- **Transformation**: AMT_INCOME_TOTAL * 0.15 for receivers only
- **Tunisia Context**: From family abroad (Europe, Gulf)

### peer_transfer_network_size
- **Source**: derived
- **Description**: Number of unique peer-to-peer transfer contacts
- **Transformation**: random_2_to_30
- **Tunisia Context**: Social financial network strength


## Alternative Indicators - Stability

### location_consistency_score
- **Source**: REGION_RATING_CLIENT
- **Description**: Geographic stability based on transaction locations
- **Transformation**: region_rating_to_stability
- **Tunisia Context**: GPS/cell tower location patterns


## Product Context

### loan_product_category
- **Source**: AMT_CREDIT + OCCUPATION_TYPE
- **Description**: ATB loan product category
- **Transformation**: rule_based_categorization
- **Categories**: Housing, Vehicle, Consumer, Business, Corporate

### requested_amount_tnd
- **Source**: AMT_CREDIT
- **Description**: Requested loan amount in Tunisian Dinars
- **Transformation**: AMT_CREDIT * 0.33 (rough USD to TND)
- **Tunisia Context**: TND amounts realistic for Tunisia market

### term_months
- **Source**: AMT_CREDIT / AMT_ANNUITY
- **Description**: Requested loan term in months
- **Transformation**: calculated_from_credit_annuity
- **Tunisia Context**: Standard ATB term ranges

