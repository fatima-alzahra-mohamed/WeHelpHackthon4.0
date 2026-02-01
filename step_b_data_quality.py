"""
Step B: Data Quality & Leakage Audit
Validates data quality and identifies/removes potential target leakage
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy.stats import pearsonr


class DataQualityAuditor:
    """Audit data quality and check for target leakage"""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'outputs' / 'datasets'
        self.reports_dir = self.base_dir / 'reports'
        
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load the generated Tunisia dataset"""
        data_file = self.data_dir / 'tunisia_loan_data.csv'
        dict_file = self.data_dir / 'tunisia_loan_data_dictionary.csv'
        
        df = pd.read_csv(data_file)
        data_dict = pd.read_csv(dict_file)
        
        return df, data_dict
    
    def check_basic_quality(self, df):
        """Check basic data quality issues"""
        
        quality_report = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'duplicates': {
                'duplicate_rows': int(df.duplicated().sum()),
                'duplicate_applicant_ids': int(df['applicant_id'].duplicated().sum())
            },
            'missing_values': {},
            'data_types': {}
        }
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        
        for col in df.columns:
            if missing[col] > 0:
                quality_report['missing_values'][col] = {
                    'count': int(missing[col]),
                    'percentage': float(missing_pct[col])
                }
        
        # Data types
        for col in df.columns:
            quality_report['data_types'][col] = str(df[col].dtype)
        
        return quality_report
    
    def check_target_leakage(self, df):
        """Check for features that may leak target information"""
        
        if 'TARGET' not in df.columns:
            return {'error': 'TARGET column not found'}
        
        target = df['TARGET']
        leakage_report = {
            'high_risk_features': [],
            'medium_risk_features': [],
            'correlations': {}
        }
        
        # Check correlations for numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'TARGET']
        
        for col in numeric_cols:
            # Skip metadata columns
            if col in ['max_financing_ratio', 'max_term_months', 'product_cap_tnd', 
                      'requires_appraisal_flag', 'applicant_id']:
                continue
            
            # Calculate correlation
            valid_mask = df[col].notna()
            if valid_mask.sum() > 10:
                corr, p_value = pearsonr(df.loc[valid_mask, col], target[valid_mask])
                
                leakage_report['correlations'][col] = {
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'abs_correlation': float(abs(corr))
                }
                
                # Flag high correlations (potential leakage)
                if abs(corr) > 0.7:
                    leakage_report['high_risk_features'].append({
                        'feature': col,
                        'correlation': float(corr),
                        'reason': 'Very high correlation with TARGET (>0.7)'
                    })
                elif abs(corr) > 0.4:
                    leakage_report['medium_risk_features'].append({
                        'feature': col,
                        'correlation': float(corr),
                        'reason': 'Moderate correlation with TARGET (>0.4)'
                    })
        
        # Check for post-outcome features (manual flags)
        post_outcome_patterns = [
            'approved', 'rejected', 'defaulted', 'paid', 'status',
            'outcome', 'result', 'final', 'actual'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in post_outcome_patterns):
                if col != 'TARGET':
                    leakage_report['high_risk_features'].append({
                        'feature': col,
                        'correlation': None,
                        'reason': 'Feature name suggests post-outcome information'
                    })
        
        return leakage_report
    
    def check_affordability_leakage(self, df):
        """Check for features that would leak affordability labels"""
        
        affordability_report = {
            'notes': 'Affordability will be modeled as income bands.',
            'potential_issues': []
        }
        
        # Check if income-like features are too directly derived
        income_proxies = ['avg_monthly_airtime_tnd', 'avg_monthly_remittance_tnd']
        
        for proxy in income_proxies:
            if proxy in df.columns:
                # These should be inputs, not targets
                affordability_report['potential_issues'].append({
                    'feature': proxy,
                    'note': f'{proxy} is used as input feature, ensure it does not directly encode income bands'
                })
        
        return affordability_report
    
    def check_distributions(self, df):
        """Check distribution sanity per loan category"""
        
        dist_report = {}
        
        if 'loan_product_category' in df.columns:
            categories = df['loan_product_category'].unique()
            
            for category in categories:
                cat_df = df[df['loan_product_category'] == category]
                
                dist_report[category] = {
                    'count': len(cat_df),
                    'default_rate': float(cat_df['TARGET'].mean()),
                    'avg_loan_amount': float(cat_df['requested_amount_tnd'].mean()),
                    'avg_term_months': float(cat_df['term_months'].mean())
                }
        
        return dist_report
    
    def create_safe_training_dataset(self, df, leakage_report):
        """Remove high-risk leakage columns and metadata-only columns"""
        
        # Columns to remove
        remove_cols = []
        
        # Remove metadata columns (not for training)
        metadata_cols = ['max_financing_ratio', 'max_term_months', 
                        'product_cap_tnd', 'requires_appraisal_flag']
        remove_cols.extend([col for col in metadata_cols if col in df.columns])
        
        # Remove high-risk leakage features
        high_risk_features = [item['feature'] for item in leakage_report['high_risk_features']]
        remove_cols.extend(high_risk_features)
        
        # Remove duplicates
        remove_cols = list(set(remove_cols))
        
        # Create safe dataset
        safe_df = df.drop(columns=remove_cols, errors='ignore')
        
        return safe_df, remove_cols
    
    def generate_markdown_report(self, quality_report, leakage_report, 
                                 affordability_report, dist_report, removed_cols):
        """Generate human-readable markdown report"""
        
        report_lines = [
            "# Data Quality Analysis Report",
            "",
            "## Summary",
            f"- **Total Records**: {quality_report['total_records']:,}",
            f"- **Total Features**: {quality_report['total_features']}",
            f"- **Duplicate Rows**: {quality_report['duplicates']['duplicate_rows']}",
            f"- **Duplicate Applicant IDs**: {quality_report['duplicates']['duplicate_applicant_ids']}",
            "",
            "## Missing Values",
        ]
        
        if quality_report['missing_values']:
            for col, stats in quality_report['missing_values'].items():
                report_lines.append(
                    f"- **{col}**: {stats['count']} ({stats['percentage']}%)"
                )
        else:
            report_lines.append("- No missing values detected")
        
        report_lines.extend([
            "",
            "## Target Leakage Analysis",
            "",
            "### High-Risk Features (Removed)",
        ])
        
        if leakage_report['high_risk_features']:
            for item in leakage_report['high_risk_features']:
                corr_str = f"r={item['correlation']:.3f}" if item['correlation'] else "N/A"
                report_lines.append(f"- **{item['feature']}**: {item['reason']} ({corr_str})")
        else:
            report_lines.append("- No high-risk features detected")
        
        report_lines.extend([
            "",
            "### Medium-Risk Features (Monitored)",
        ])
        
        if leakage_report['medium_risk_features']:
            for item in leakage_report['medium_risk_features']:
                report_lines.append(f"- **{item['feature']}**: {item['reason']} (r={item['correlation']:.3f})")
        else:
            report_lines.append("- No medium-risk features detected")
        
        report_lines.extend([
            "",
            "## Affordability Leakage Check",
            f"- **Note**: {affordability_report['notes']}",
        ])
        
        if affordability_report['potential_issues']:
            report_lines.append("- **Potential Issues**:")
            for issue in affordability_report['potential_issues']:
                report_lines.append(f"  - {issue['feature']}: {issue['note']}")
        
        report_lines.extend([
            "",
            "## Distribution Checks by Loan Category",
        ])
        
        for category, stats in dist_report.items():
            report_lines.extend([
                f"",
                f"### {category}",
                f"- Count: {stats['count']:,}",
                f"- Default Rate: {stats['default_rate']:.2%}",
                f"- Avg Loan Amount: {stats['avg_loan_amount']:,.2f} TND",
                f"- Avg Term: {stats['avg_term_months']:.0f} months"
            ])
        
        report_lines.extend([
            "",
            "## Columns Removed for Safe Training",
        ])
        
        if removed_cols:
            for col in removed_cols:
                report_lines.append(f"- {col}")
        else:
            report_lines.append("- None")
        
        report_lines.extend([
            "",
            "## Conclusion",
            "- Data quality checks completed",
            "- Target leakage analysis completed",
            "- Safe training dataset created (tunisia_loan_data_train.csv)",
            "- Ready for modeling phase"
        ])
        
        return "\n".join(report_lines)
    
    def run(self):
        """Execute complete data quality audit"""
        
        print("=" * 80)
        print("STEP B: DATA QUALITY & LEAKAGE AUDIT")
        print("=" * 80)
        
        # Load data
        print("\n[B1] Loading data...")
        df, data_dict = self.load_data()
        print(f"Loaded {len(df)} records, {len(df.columns)} columns")
        
        # Basic quality checks
        print("\n[B2] Running basic quality checks...")
        quality_report = self.check_basic_quality(df)
        print(f"✓ Found {quality_report['duplicates']['duplicate_rows']} duplicate rows")
        print(f"✓ Found {len(quality_report['missing_values'])} columns with missing values")
        
        # Target leakage checks
        print("\n[B3] Checking for target leakage...")
        leakage_report = self.check_target_leakage(df)
        print(f"✓ Found {len(leakage_report['high_risk_features'])} high-risk features")
        print(f"✓ Found {len(leakage_report['medium_risk_features'])} medium-risk features")
        
        # Affordability leakage
        print("\n[B4] Checking affordability feature leakage...")
        affordability_report = self.check_affordability_leakage(df)
        
        # Distribution checks
        print("\n[B5] Checking distributions by loan category...")
        dist_report = self.check_distributions(df)
        print(f"✓ Analyzed {len(dist_report)} loan categories")
        
        # Create safe training dataset
        print("\n[B6] Creating safe training dataset...")
        safe_df, removed_cols = self.create_safe_training_dataset(df, leakage_report)
        print(f"✓ Removed {len(removed_cols)} columns: {removed_cols}")
        print(f"✓ Safe dataset shape: {safe_df.shape}")
        
        # Save outputs
        print("\n[B7] Saving outputs...")
        
        # JSON report
        full_report = {
            'quality': quality_report,
            'leakage': leakage_report,
            'affordability': affordability_report,
            'distributions': dist_report,
            'removed_columns': removed_cols
        }
        
        json_file = self.reports_dir / 'data_quality_analysis.json'
        with open(json_file, 'w') as f:
            json.dump(full_report, f, indent=2)
        print(f"✓ Saved: {json_file}")
        
        # Markdown report
        md_content = self.generate_markdown_report(
            quality_report, leakage_report, affordability_report, 
            dist_report, removed_cols
        )
        md_file = self.reports_dir / 'data_quality_report.md'
        with open(md_file, 'w') as f:
            f.write(md_content)
        print(f"✓ Saved: {md_file}")
        
        # Safe training dataset
        safe_file = self.data_dir / 'tunisia_loan_data_train.csv'
        safe_df.to_csv(safe_file, index=False)
        print(f"✓ Saved: {safe_file}")
        
        print("\n" + "=" * 80)
        print("STEP B COMPLETE")
        print("=" * 80)
        print(f"\nSafe training dataset: {safe_df.shape}")
        print(f"Removed columns: {len(removed_cols)}")
        
        return safe_df, full_report


if __name__ == '__main__':
    auditor = DataQualityAuditor()
    safe_df, report = auditor.run()
