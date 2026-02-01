"""
Step A: Dataset Creation - Home Credit to Tunisia Alternative Data Mapping
Generates Tunisia-style credit scoring dataset with alternative financial indicators
"""
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

class TunisiaDataGenerator:
    """Generate Tunisia loan dataset from Home Credit base structure"""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data' / 'raw'
        self.output_dir = self.base_dir / 'outputs' / 'datasets'
        self.reports_dir = self.base_dir / 'reports'
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def load_or_generate_home_credit_base(self, n_samples=10000):
        """Load Home Credit data or generate synthetic base if not available"""
        app_train_path = self.data_dir / 'application_train.csv'
        
        if app_train_path.exists():
            print(f"Loading Home Credit data from {app_train_path}")
            df = pd.read_csv(app_train_path)
            if len(df) > n_samples:
                df = df.sample(n=n_samples, random_state=42)
            return df
        else:
            print(f"Home Credit data not found. Generating synthetic base dataset...")
            return self._generate_synthetic_home_credit_base(n_samples)
    
    def _generate_synthetic_home_credit_base(self, n_samples=10000):
        """Generate synthetic Home Credit-like dataset for demo purposes"""
        
        # TARGET: 1 = default, 0 = repaid (realistic imbalance ~8% default)
        default_rate = 0.08
        target = np.random.choice([0, 1], size=n_samples, p=[1-default_rate, default_rate])
        
        # Age: 21-70, skewed younger
        age_days = np.random.gamma(shape=2, scale=7000, size=n_samples)
        age_days = np.clip(age_days, 21*365, 70*365)
        
        # Income: correlated with non-default
        base_income = np.random.lognormal(mean=11.5, sigma=0.6, size=n_samples)
        income_adjustment = np.where(target == 0, 1.15, 0.85)  # Non-defaulters earn more
        amt_income_total = base_income * income_adjustment
        
        # Credit amount: reasonable loan-to-income ratios
        credit_ratio = np.random.uniform(0.5, 4.0, size=n_samples)
        amt_credit = amt_income_total * credit_ratio
        
        # Annuity: monthly payment
        amt_annuity = amt_credit / np.random.uniform(12, 60, size=n_samples)
        
        # Gender
        code_gender = np.random.choice(['M', 'F'], size=n_samples, p=[0.35, 0.65])
        
        # Education
        name_education_type = np.random.choice([
            'Secondary / secondary special',
            'Higher education',
            'Incomplete higher',
            'Lower secondary',
            'Academic degree'
        ], size=n_samples, p=[0.50, 0.25, 0.15, 0.08, 0.02])
        
        # Family status
        name_family_status = np.random.choice([
            'Married', 'Single / not married', 'Civil marriage', 
            'Separated', 'Widow'
        ], size=n_samples, p=[0.63, 0.21, 0.09, 0.05, 0.02])
        
        # Employment
        days_employed = -np.abs(np.random.gamma(shape=2, scale=1000, size=n_samples))
        
        # Previous queries (credit bureau)
        # Defaulters tend to have more recent queries
        amt_req_credit_bureau_qrt = np.random.poisson(
            lam=np.where(target == 0, 0.3, 0.8), 
            size=n_samples
        )
        
        # External scores (proxy for payment behavior)
        # Higher scores = lower default risk
        ext_source_2 = np.random.beta(
            a=np.where(target == 0, 5, 2),
            b=np.where(target == 0, 2, 5),
            size=n_samples
        )
        ext_source_3 = np.random.beta(
            a=np.where(target == 0, 4, 2),
            b=np.where(target == 0, 2, 4),
            size=n_samples
        )
        
        # Region rating
        region_rating_client = np.random.choice([1, 2, 3], size=n_samples, p=[0.15, 0.60, 0.25])
        
        # Occupation type
        occupation_type = np.random.choice([
            'Laborers', 'Sales staff', 'Core staff', 'Managers', 
            'Drivers', 'High skill tech staff', 'Accountants', 
            'Medicine staff', 'Security staff', 'Cooking staff',
            'Cleaning staff', 'Private service staff', 'Low-skill Laborers'
        ], size=n_samples)
        
        # Organization type
        organization_type = np.random.choice([
            'Business Entity Type 3', 'Self-employed', 'Other', 
            'Medicine', 'Government', 'Trade: type 7', 'Construction',
            'School', 'Industry: type 9', 'Military'
        ], size=n_samples)
        
        df = pd.DataFrame({
            'SK_ID_CURR': range(100000, 100000 + n_samples),
            'TARGET': target,
            'CODE_GENDER': code_gender,
            'DAYS_BIRTH': -age_days.astype(int),
            'DAYS_EMPLOYED': days_employed.astype(int),
            'NAME_EDUCATION_TYPE': name_education_type,
            'NAME_FAMILY_STATUS': name_family_status,
            'AMT_INCOME_TOTAL': amt_income_total,
            'AMT_CREDIT': amt_credit,
            'AMT_ANNUITY': amt_annuity,
            'REGION_RATING_CLIENT': region_rating_client,
            'EXT_SOURCE_2': ext_source_2,
            'EXT_SOURCE_3': ext_source_3,
            'AMT_REQ_CREDIT_BUREAU_QRT': amt_req_credit_bureau_qrt,
            'OCCUPATION_TYPE': occupation_type,
            'ORGANIZATION_TYPE': organization_type,
        })
        
        return df
    
    def create_feature_mapping(self):
        """Create mapping from Home Credit features to Tunisia alternative indicators"""
        
        mapping = {
            'Identity/Demographics': {
                'applicant_id': {
                    'source': 'SK_ID_CURR',
                    'description': 'Unique applicant identifier',
                    'transformation': 'direct_copy'
                },
                'age': {
                    'source': 'DAYS_BIRTH',
                    'description': 'Age in years derived from days',
                    'transformation': 'abs(DAYS_BIRTH) / 365.25'
                },
                'gender': {
                    'source': 'CODE_GENDER',
                    'description': 'Gender (M/F)',
                    'transformation': 'direct_copy'
                },
                'governorate': {
                    'source': 'REGION_RATING_CLIENT',
                    'description': 'Tunisia governorate (mapped from region rating)',
                    'transformation': 'map_to_tunisia_governorates'
                },
                'education': {
                    'source': 'NAME_EDUCATION_TYPE',
                    'description': 'Education level',
                    'transformation': 'map_to_tunisia_education'
                }
            },
            'Alternative Indicators - Payment Behavior': {
                'payment_on_time_rate': {
                    'source': 'EXT_SOURCE_2',
                    'description': 'Historical on-time payment rate from mobile money/telecom bills',
                    'transformation': 'EXT_SOURCE_2 with noise',
                    'tunisia_context': 'Aggregated from D17, Ooredoo, Orange bill payments'
                },
                'transaction_regularity_score': {
                    'source': 'EXT_SOURCE_3',
                    'description': 'Consistency of financial transactions over time',
                    'transformation': 'EXT_SOURCE_3 with scaling',
                    'tunisia_context': 'Monthly transaction pattern stability'
                },
                'dispute_rate': {
                    'source': 'AMT_REQ_CREDIT_BUREAU_QRT',
                    'description': 'Rate of payment disputes or chargebacks',
                    'transformation': 'normalized_inverse(AMT_REQ_CREDIT_BUREAU_QRT)',
                    'tunisia_context': 'Disputes from e-commerce, mobile services'
                }
            },
            'Alternative Indicators - Telecom': {
                'months_with_provider': {
                    'source': 'DAYS_EMPLOYED',
                    'description': 'Tenure with current telecom provider',
                    'transformation': 'abs(DAYS_EMPLOYED) / 30, capped at 120',
                    'tunisia_context': 'Ooredoo, Orange, Tunisie Telecom tenure'
                },
                'avg_monthly_airtime_tnd': {
                    'source': 'AMT_ANNUITY',
                    'description': 'Average monthly mobile airtime spending',
                    'transformation': 'AMT_ANNUITY * 0.02 with noise',
                    'tunisia_context': 'Proxy for spending capacity'
                },
                'mobile_topup_frequency': {
                    'source': 'derived',
                    'description': 'Number of mobile top-ups per month',
                    'transformation': 'random_4_to_20_based_on_income',
                    'tunisia_context': 'Higher frequency = regular income flow'
                },
                'business_hours_data_usage': {
                    'source': 'OCCUPATION_TYPE',
                    'description': 'Percentage of data used during business hours',
                    'transformation': 'occupation_based_pattern',
                    'tunisia_context': 'Employment stability indicator'
                }
            },
            'Alternative Indicators - Mobile Money': {
                'mobile_money_account': {
                    'source': 'derived',
                    'description': 'Has active mobile money account (D17, e-dinar)',
                    'transformation': 'probabilistic_based_on_age_education',
                    'tunisia_context': 'D17 (ATB mobile money) or e-dinar usage'
                },
                'avg_monthly_transactions': {
                    'source': 'derived',
                    'description': 'Average monthly mobile money transactions',
                    'transformation': 'income_and_education_based',
                    'tunisia_context': 'Financial activity level'
                }
            },
            'Alternative Indicators - Marketplace': {
                'marketplace_seller_rating': {
                    'source': 'EXT_SOURCE_2',
                    'description': 'Seller rating on e-commerce platforms (if applicable)',
                    'transformation': 'EXT_SOURCE_2 * 5, nullable for non-sellers',
                    'tunisia_context': 'Jumia, Tunisie Net, local marketplaces'
                },
                'completed_transactions': {
                    'source': 'derived',
                    'description': 'Number of completed marketplace transactions',
                    'transformation': 'random_for_sellers_only',
                    'tunisia_context': 'E-commerce trust indicator'
                }
            },
            'Alternative Indicators - Remittances/Social': {
                'receives_remittances': {
                    'source': 'derived',
                    'description': 'Receives international remittances',
                    'transformation': 'probabilistic_20_percent',
                    'tunisia_context': 'Important income source for ~20% Tunisians'
                },
                'avg_monthly_remittance_tnd': {
                    'source': 'AMT_INCOME_TOTAL',
                    'description': 'Average monthly remittance amount',
                    'transformation': 'AMT_INCOME_TOTAL * 0.15 for receivers only',
                    'tunisia_context': 'From family abroad (Europe, Gulf)'
                },
                'peer_transfer_network_size': {
                    'source': 'derived',
                    'description': 'Number of unique peer-to-peer transfer contacts',
                    'transformation': 'random_2_to_30',
                    'tunisia_context': 'Social financial network strength'
                }
            },
            'Alternative Indicators - Stability': {
                'location_consistency_score': {
                    'source': 'REGION_RATING_CLIENT',
                    'description': 'Geographic stability based on transaction locations',
                    'transformation': 'region_rating_to_stability',
                    'tunisia_context': 'GPS/cell tower location patterns'
                }
            },
            'Product Context': {
                'loan_product_category': {
                    'source': 'AMT_CREDIT + OCCUPATION_TYPE',
                    'description': 'ATB loan product category',
                    'transformation': 'rule_based_categorization',
                    'categories': ['Housing', 'Vehicle', 'Consumer', 'Business', 'Corporate']
                },
                'requested_amount_tnd': {
                    'source': 'AMT_CREDIT',
                    'description': 'Requested loan amount in Tunisian Dinars',
                    'transformation': 'AMT_CREDIT * 0.33 (rough USD to TND)',
                    'tunisia_context': 'TND amounts realistic for Tunisia market'
                },
                'term_months': {
                    'source': 'AMT_CREDIT / AMT_ANNUITY',
                    'description': 'Requested loan term in months',
                    'transformation': 'calculated_from_credit_annuity',
                    'tunisia_context': 'Standard ATB term ranges'
                }
            }
        }
        
        # Save mapping as reference
        mapping_file = self.reports_dir / 'FEATURE_MAPPING.md'
        with open(mapping_file, 'w') as f:
            f.write("# Tunisia Alternative Data Feature Mapping\n\n")
            f.write("**Purpose**: Map Home Credit dataset features to Tunisia-relevant alternative financial indicators\n\n")
            f.write("**LLM Role**: This mapping was designed with LLM assistance to ensure Tunisia market relevance\n\n")
            
            for category, features in mapping.items():
                f.write(f"\n## {category}\n\n")
                for feature_name, details in features.items():
                    f.write(f"### {feature_name}\n")
                    f.write(f"- **Source**: {details['source']}\n")
                    f.write(f"- **Description**: {details['description']}\n")
                    f.write(f"- **Transformation**: {details['transformation']}\n")
                    if 'tunisia_context' in details:
                        f.write(f"- **Tunisia Context**: {details['tunisia_context']}\n")
                    if 'categories' in details:
                        f.write(f"- **Categories**: {', '.join(details['categories'])}\n")
                    f.write("\n")
        
        print(f"Feature mapping saved to {mapping_file}")
        
        # Save as CSV reference
        mapping_rows = []
        for category, features in mapping.items():
            for feature_name, details in features.items():
                mapping_rows.append({
                    'category': category,
                    'feature_name': feature_name,
                    'source': details['source'],
                    'description': details['description'],
                    'transformation': details['transformation']
                })
        
        mapping_df = pd.DataFrame(mapping_rows)
        mapping_csv = self.output_dir / 'feature_mapping_reference.csv'
        mapping_df.to_csv(mapping_csv, index=False)
        print(f"Feature mapping CSV saved to {mapping_csv}")
        
        return mapping
    
    def generate_tunisia_dataset(self, home_credit_df):
        """Transform Home Credit data into Tunisia alternative data format"""
        
        n = len(home_credit_df)
        
        # Tunisia governorates mapping
        governorates = [
            'Tunis', 'Ariana', 'Ben Arous', 'Manouba', 'Nabeul', 'Zaghouan',
            'Bizerte', 'Béja', 'Jendouba', 'Kef', 'Siliana', 'Sousse', 'Monastir',
            'Mahdia', 'Sfax', 'Kairouan', 'Kasserine', 'Sidi Bouzid', 'Gabès',
            'Medenine', 'Tataouine', 'Gafsa', 'Tozeur', 'Kebili'
        ]
        
        # Education mapping
        education_map = {
            'Secondary / secondary special': 'Secondary',
            'Higher education': 'University',
            'Incomplete higher': 'Some University',
            'Lower secondary': 'Primary',
            'Academic degree': 'Graduate'
        }
        
        # Initialize Tunisia dataset
        tunisia_data = pd.DataFrame()
        
        # Identity/Demographics
        tunisia_data['applicant_id'] = home_credit_df['SK_ID_CURR'].values
        tunisia_data['age'] = np.abs(home_credit_df['DAYS_BIRTH'].values) / 365.25
        tunisia_data['age'] = tunisia_data['age'].round().astype(int)
        tunisia_data['gender'] = home_credit_df['CODE_GENDER'].values
        
        # Governorate (region-based with urban concentration)
        region_to_gov = {
            1: np.random.choice(['Tunis', 'Ariana', 'Ben Arous', 'Sfax'], n),
            2: np.random.choice(governorates[6:12], n),  # Northern/central
            3: np.random.choice(governorates[12:], n)    # Southern
        }
        tunisia_data['governorate'] = home_credit_df['REGION_RATING_CLIENT'].apply(
            lambda x: np.random.choice(region_to_gov.get(x, governorates))
        )
        
        tunisia_data['education'] = home_credit_df['NAME_EDUCATION_TYPE'].map(education_map)
        
        # Alternative Indicators - Payment Behavior
        tunisia_data['payment_on_time_rate'] = (
            home_credit_df['EXT_SOURCE_2'].fillna(0.5) * 
            np.random.uniform(0.95, 1.05, n)
        ).clip(0, 1).round(3)
        
        tunisia_data['transaction_regularity_score'] = (
            home_credit_df['EXT_SOURCE_3'].fillna(0.5) * 100
        ).clip(0, 100).round(1)
        
        # Lower dispute rate for non-defaulters
        base_disputes = home_credit_df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0)
        tunisia_data['dispute_rate'] = (
            base_disputes / (base_disputes.max() + 1)
        ).clip(0, 0.5).round(3)
        
        # Alternative Indicators - Telecom
        tunisia_data['months_with_provider'] = (
            np.abs(home_credit_df['DAYS_EMPLOYED'].fillna(-365)) / 30
        ).clip(1, 120).round().astype(int)
        
        # Airtime spending correlated with income capacity
        tunisia_data['avg_monthly_airtime_tnd'] = (
            home_credit_df['AMT_ANNUITY'].fillna(home_credit_df['AMT_ANNUITY'].median()) * 
            0.02 * np.random.uniform(0.8, 1.2, n)
        ).clip(5, 200).round(2)
        
        tunisia_data['mobile_topup_frequency'] = np.random.randint(4, 21, n)
        
        # Business hours data usage based on occupation
        business_occupations = ['Managers', 'Core staff', 'Accountants', 'High skill tech staff']
        tunisia_data['business_hours_data_usage'] = home_credit_df['OCCUPATION_TYPE'].apply(
            lambda x: np.random.uniform(60, 85) if x in business_occupations else np.random.uniform(30, 60)
        ).round(1)
        
        # Alternative Indicators - Mobile Money
        # Younger, more educated people more likely to have mobile money
        mobile_money_prob = (
            (tunisia_data['age'] < 45).astype(float) * 0.3 +
            (tunisia_data['education'].isin(['University', 'Graduate'])).astype(float) * 0.3 +
            0.4
        )
        tunisia_data['mobile_money_account'] = (
            np.random.random(n) < mobile_money_prob
        ).astype(int)
        
        tunisia_data['avg_monthly_transactions'] = (
            tunisia_data['mobile_money_account'] * 
            np.random.randint(5, 50, n)
        )
        
        # Alternative Indicators - Marketplace
        # 20% are sellers
        is_seller = np.random.random(n) < 0.2
        tunisia_data['marketplace_seller_rating'] = np.where(
            is_seller,
            (home_credit_df['EXT_SOURCE_2'].fillna(0.5) * 5).clip(0, 5).round(2),
            np.nan
        )
        tunisia_data['completed_transactions'] = np.where(
            is_seller,
            np.random.randint(10, 500, n),
            0
        )
        
        # Alternative Indicators - Remittances/Social
        tunisia_data['receives_remittances'] = (np.random.random(n) < 0.20).astype(int)
        tunisia_data['avg_monthly_remittance_tnd'] = np.where(
            tunisia_data['receives_remittances'] == 1,
            (home_credit_df['AMT_INCOME_TOTAL'].values * 0.33 * 0.15 * 
             np.random.uniform(0.8, 1.2, n)).clip(50, 2000).round(2),
            0
        )
        tunisia_data['peer_transfer_network_size'] = np.random.randint(2, 31, n)
        
        # Alternative Indicators - Stability
        tunisia_data['location_consistency_score'] = (
            (4 - home_credit_df['REGION_RATING_CLIENT'].values) / 3 * 100
        ).clip(0, 100).round(1)
        
        # Product Context - Loan Category
        # Rule-based categorization
        amt_credit = home_credit_df['AMT_CREDIT'].values
        occupation = home_credit_df['OCCUPATION_TYPE'].values
        organization = home_credit_df['ORGANIZATION_TYPE'].values
        
        def assign_loan_category(credit, occ, org):
            # Housing: large loans
            if credit > 300000:
                return 'Housing'
            # Corporate: business organizations + large loans
            elif org in ['Business Entity Type 3', 'Industry: type 9'] and credit > 200000:
                return 'Corporate'
            # Business: self-employed or business contexts
            elif org == 'Self-employed' or occ == 'Managers':
                return 'Business'
            # Vehicle: medium loans, certain occupations
            elif 100000 <= credit <= 250000 and occ in ['Drivers', 'Sales staff']:
                return 'Vehicle'
            # Consumer: everything else
            else:
                return 'Consumer'
        
        tunisia_data['loan_product_category'] = [
            assign_loan_category(c, o, org) 
            for c, o, org in zip(amt_credit, occupation, organization)
        ]
        
        # Adjust amounts to be realistic for Tunisia (convert and adjust)
        # USD to TND ~= 3:1, but adjust for purchasing power
        category_multipliers = {
            'Housing': 0.40,
            'Corporate': 0.35,
            'Business': 0.30,
            'Vehicle': 0.25,
            'Consumer': 0.20
        }
        
        tunisia_data['requested_amount_tnd'] = [
            amt_credit[i] * category_multipliers[cat] * np.random.uniform(0.9, 1.1)
            for i, cat in enumerate(tunisia_data['loan_product_category'])
        ]
        tunisia_data['requested_amount_tnd'] = tunisia_data['requested_amount_tnd'].round(2)
        
        # Term months
        tunisia_data['term_months'] = (
            home_credit_df['AMT_CREDIT'].values / 
            home_credit_df['AMT_ANNUITY'].fillna(home_credit_df['AMT_ANNUITY'].median()).values
        ).clip(6, 360).round().astype(int)
        
        # ATB Product Constraints (metadata, NOT for training)
        product_constraints = {
            'Housing': {'max_financing_ratio': 0.80, 'max_term_months': 300, 
                       'product_cap_tnd': None, 'requires_appraisal_flag': 1},
            'Vehicle': {'max_financing_ratio': 0.85, 'max_term_months': 84,
                       'product_cap_tnd': 150000, 'requires_appraisal_flag': 1},
            'Consumer': {'max_financing_ratio': 1.0, 'max_term_months': 60,
                        'product_cap_tnd': 50000, 'requires_appraisal_flag': 0},
            'Business': {'max_financing_ratio': 0.70, 'max_term_months': 120,
                        'product_cap_tnd': None, 'requires_appraisal_flag': 1},
            'Corporate': {'max_financing_ratio': 0.75, 'max_term_months': 180,
                         'product_cap_tnd': None, 'requires_appraisal_flag': 1}
        }
        
        for col in ['max_financing_ratio', 'max_term_months', 'product_cap_tnd', 'requires_appraisal_flag']:
            tunisia_data[col] = tunisia_data['loan_product_category'].map(
                {k: v[col] for k, v in product_constraints.items()}
            )
        
        # TARGET (keep from Home Credit)
        tunisia_data['TARGET'] = home_credit_df['TARGET'].values
        
        return tunisia_data
    
    def create_data_dictionary(self, tunisia_df):
        """Create data dictionary with leakage risk flags"""
        
        data_dict = []
        
        for col in tunisia_df.columns:
            dtype = str(tunisia_df[col].dtype)
            
            # Identify potential leakage
            leakage_risk = 'none'
            if col == 'TARGET':
                leakage_risk = 'target'
            elif col in ['max_financing_ratio', 'max_term_months', 'product_cap_tnd', 'requires_appraisal_flag']:
                leakage_risk = 'metadata_only'
            
            # Descriptions
            descriptions = {
                'applicant_id': 'Unique applicant identifier',
                'age': 'Applicant age in years',
                'gender': 'Gender (M/F)',
                'governorate': 'Tunisia governorate of residence',
                'education': 'Highest education level attained',
                'payment_on_time_rate': 'Historical on-time payment rate (0-1) from mobile/telecom bills',
                'transaction_regularity_score': 'Transaction consistency score (0-100)',
                'dispute_rate': 'Rate of payment disputes or chargebacks (0-1)',
                'months_with_provider': 'Months with current telecom provider',
                'avg_monthly_airtime_tnd': 'Average monthly mobile airtime spending in TND',
                'mobile_topup_frequency': 'Number of mobile top-ups per month',
                'business_hours_data_usage': 'Percentage of data used during business hours',
                'mobile_money_account': 'Has active mobile money account (1=yes, 0=no)',
                'avg_monthly_transactions': 'Average monthly mobile money transactions',
                'marketplace_seller_rating': 'E-commerce seller rating (0-5, null if not seller)',
                'completed_transactions': 'Number of completed marketplace transactions',
                'receives_remittances': 'Receives international remittances (1=yes, 0=no)',
                'avg_monthly_remittance_tnd': 'Average monthly remittance in TND',
                'peer_transfer_network_size': 'Number of unique P2P transfer contacts',
                'location_consistency_score': 'Geographic stability score (0-100)',
                'loan_product_category': 'ATB loan product category',
                'requested_amount_tnd': 'Requested loan amount in Tunisian Dinars',
                'term_months': 'Requested loan term in months',
                'max_financing_ratio': 'Maximum LTV ratio for product (metadata)',
                'max_term_months': 'Maximum term for product (metadata)',
                'product_cap_tnd': 'Product amount cap in TND (metadata)',
                'requires_appraisal_flag': 'Requires asset appraisal (metadata)',
                'TARGET': 'Loan outcome (1=default, 0=repaid)'
            }
            
            data_dict.append({
                'column': col,
                'dtype': dtype,
                'description': descriptions.get(col, 'Unknown'),
                'leakage_risk': leakage_risk
            })
        
        return pd.DataFrame(data_dict)
    
    def generate_summary_stats(self, tunisia_df):
        """Generate summary statistics for the dataset"""
        
        summary = {
            'total_records': len(tunisia_df),
            'target_distribution': tunisia_df['TARGET'].value_counts().to_dict(),
            'default_rate': tunisia_df['TARGET'].mean(),
            'missing_values': tunisia_df.isnull().sum().to_dict(),
            'loan_categories': tunisia_df['loan_product_category'].value_counts().to_dict(),
            'governorates': tunisia_df['governorate'].nunique(),
            'age_range': {
                'min': int(tunisia_df['age'].min()),
                'max': int(tunisia_df['age'].max()),
                'mean': float(tunisia_df['age'].mean())
            },
            'loan_amount_stats': {
                'min': float(tunisia_df['requested_amount_tnd'].min()),
                'max': float(tunisia_df['requested_amount_tnd'].max()),
                'mean': float(tunisia_df['requested_amount_tnd'].mean()),
                'median': float(tunisia_df['requested_amount_tnd'].median())
            }
        }
        
        return summary
    
    def run(self, n_samples=10000):
        """Execute complete data generation pipeline"""
        
        print("=" * 80)
        print("STEP A: TUNISIA DATASET GENERATION")
        print("=" * 80)
        
        # A1-A2: Load/generate Home Credit base
        print("\n[A1-A2] Loading Home Credit base data...")
        home_credit_df = self.load_or_generate_home_credit_base(n_samples)
        print(f"Loaded {len(home_credit_df)} records")
        print(f"Default rate: {home_credit_df['TARGET'].mean():.2%}")
        
        # A3: Create feature mapping
        print("\n[A3] Creating feature mapping...")
        self.create_feature_mapping()
        
        # A4-A5: Generate Tunisia dataset
        print("\n[A4-A5] Generating Tunisia alternative data dataset...")
        tunisia_df = self.generate_tunisia_dataset(home_credit_df)
        print(f"Generated {len(tunisia_df)} records with {len(tunisia_df.columns)} features")
        
        # A6: Save outputs
        print("\n[A6] Saving outputs...")
        
        # Main dataset
        output_file = self.output_dir / 'tunisia_loan_data.csv'
        tunisia_df.to_csv(output_file, index=False)
        print(f"✓ Saved: {output_file}")
        
        # Data dictionary
        data_dict_df = self.create_data_dictionary(tunisia_df)
        dict_file = self.output_dir / 'tunisia_loan_data_dictionary.csv'
        data_dict_df.to_csv(dict_file, index=False)
        print(f"✓ Saved: {dict_file}")
        
        # Summary stats
        summary = self.generate_summary_stats(tunisia_df)
        summary_file = self.reports_dir / 'data_generation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved: {summary_file}")
        
        print("\n" + "=" * 80)
        print("STEP A COMPLETE")
        print("=" * 80)
        print(f"\nDataset shape: {tunisia_df.shape}")
        print(f"Default rate: {tunisia_df['TARGET'].mean():.2%}")
        print(f"Loan categories: {tunisia_df['loan_product_category'].value_counts().to_dict()}")
        
        return tunisia_df, data_dict_df, summary


if __name__ == '__main__':
    generator = TunisiaDataGenerator()
    tunisia_df, data_dict, summary = generator.run(n_samples=10000)
