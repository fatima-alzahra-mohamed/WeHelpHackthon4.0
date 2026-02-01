"""
Step E: Graphics Generation and Final Reporting
Create visualizations and comprehensive demo report
"""
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class ReportGenerator:
    """Generate visualizations and final demo report"""
    
    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'outputs' / 'datasets'
        self.reports_dir = self.base_dir / 'reports'
        self.figures_dir = self.base_dir / 'outputs' / 'figures'
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
    
    def load_data(self):
        """Load dataset and metadata"""
        data_file = self.data_dir / 'tunisia_loan_data.csv'
        df = pd.read_csv(data_file)
        
        # Load evaluation results
        metrics_file = self.reports_dir / 'model_metrics.json'
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return df, metrics
    
    def create_schema_diagram(self):
        """Create system architecture diagram"""
        
        print("Creating schema diagram...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('off')
        
        # Feature groups
        feature_groups = {
            'Identity/Demographics': ['applicant_id', 'age', 'gender', 'governorate', 'education'],
            'Payment Behavior': ['payment_on_time_rate', 'transaction_regularity_score', 'dispute_rate'],
            'Telecom': ['months_with_provider', 'avg_monthly_airtime_tnd', 'mobile_topup_frequency', 'business_hours_data_usage'],
            'Mobile Money': ['mobile_money_account', 'avg_monthly_transactions'],
            'Marketplace': ['marketplace_seller_rating', 'completed_transactions'],
            'Remittances/Social': ['receives_remittances', 'avg_monthly_remittance_tnd', 'peer_transfer_network_size'],
            'Stability': ['location_consistency_score'],
            'Product Context': ['loan_product_category', 'requested_amount_tnd', 'term_months']
        }
        
        # Draw feature group boxes
        y_start = 0.85
        y_step = 0.10
        
        for i, (group, features) in enumerate(feature_groups.items()):
            y_pos = y_start - (i * y_step)
            
            # Group box
            rect = plt.Rectangle((0.05, y_pos - 0.08), 0.35, 0.08, 
                                facecolor='lightblue', edgecolor='navy', linewidth=2)
            ax.add_patch(rect)
            
            # Group name
            ax.text(0.225, y_pos - 0.04, group, 
                   ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Unified engine box
        engine_rect = plt.Rectangle((0.45, 0.35), 0.20, 0.30,
                                   facecolor='lightgreen', edgecolor='darkgreen', linewidth=3)
        ax.add_patch(engine_rect)
        ax.text(0.55, 0.60, 'UNIFIED\nCREDIT\nENGINE', 
               ha='center', va='center', fontweight='bold', fontsize=14)
        
        # Output boxes
        outputs = [
            ('Eligibility\nProbability', 0.75),
            ('Default Risk\nProbability', 0.50),
            ('Affordability\nBand', 0.25)
        ]
        
        for label, y_pos in outputs:
            output_rect = plt.Rectangle((0.70, y_pos - 0.08), 0.20, 0.08,
                                       facecolor='lightyellow', edgecolor='orange', linewidth=2)
            ax.add_patch(output_rect)
            ax.text(0.80, y_pos - 0.04, label,
                   ha='center', va='center', fontweight='bold', fontsize=10)
        
        # Arrows
        # Inputs to engine
        ax.arrow(0.40, 0.50, 0.04, 0, head_width=0.03, head_length=0.02, 
                fc='black', ec='black', linewidth=2)
        
        # Engine to outputs
        for _, y_pos in outputs:
            ax.arrow(0.65, 0.50, 0.04, y_pos - 0.54, head_width=0.02, head_length=0.01,
                    fc='darkgreen', ec='darkgreen', linewidth=1.5)
        
        # Title
        ax.text(0.50, 0.95, 'Tunisia AI Credit Scoring System Architecture',
               ha='center', va='top', fontsize=16, fontweight='bold')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        output_file = self.figures_dir / 'system_schema.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
    
    def create_data_snapshot(self, df):
        """Create visual data snapshot showing example rows"""
        
        print("Creating data snapshot...")
        
        # Select key columns for display
        display_cols = [
            'applicant_id', 'age', 'gender', 'loan_product_category',
            'requested_amount_tnd', 'payment_on_time_rate',
            'mobile_money_account', 'TARGET'
        ]
        
        # Get sample rows
        sample_df = df[display_cols].head(8).copy()
        
        # Mask IDs for privacy
        sample_df['applicant_id'] = ['APP_' + str(i).zfill(4) for i in range(1, len(sample_df) + 1)]
        
        # Format numeric columns
        sample_df['requested_amount_tnd'] = sample_df['requested_amount_tnd'].round(0).astype(int)
        sample_df['payment_on_time_rate'] = sample_df['payment_on_time_rate'].round(2)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(
            cellText=sample_df.values,
            colLabels=sample_df.columns,
            cellLoc='center',
            loc='center',
            colWidths=[0.12] * len(display_cols)
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(display_cols)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(sample_df) + 1):
            for j in range(len(display_cols)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Tunisia Loan Data - Sample Records', 
                 fontsize=14, fontweight='bold', pad=20)
        
        output_file = self.figures_dir / 'data_snapshot.png'
        plt.savefig(output_file, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
    
    def create_target_distribution(self, df):
        """Create TARGET distribution plot"""
        
        print("Creating TARGET distribution plot...")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        target_counts = df['TARGET'].value_counts().sort_index()
        colors = ['#4CAF50', '#f44336']
        
        bars = ax.bar(['Repaid (0)', 'Default (1)'], target_counts.values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}\n({height/len(df)*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Number of Loans', fontsize=12, fontweight='bold')
        ax.set_title('Target Distribution (Class Imbalance)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.figures_dir / 'target_distribution.png'
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
    
    def create_loan_amount_by_category(self, df):
        """Create loan amount distribution by category"""
        
        print("Creating loan amount by category plot...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Box plot
        categories = df['loan_product_category'].unique()
        data = [df[df['loan_product_category'] == cat]['requested_amount_tnd'].values 
                for cat in categories]
        
        bp = ax.boxplot(data, labels=categories, patch_artist=True,
                       medianprops=dict(color='red', linewidth=2),
                       boxprops=dict(facecolor='lightblue', alpha=0.7))
        
        ax.set_ylabel('Requested Amount (TND)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Loan Product Category', fontsize=12, fontweight='bold')
        ax.set_title('Loan Amount Distribution by Product Category', 
                    fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.figures_dir / 'loan_amount_by_category.png'
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
    
    def create_correlation_heatmap(self, df):
        """Create correlation heatmap for top numeric features"""
        
        print("Creating correlation heatmap...")
        
        # Select key numeric features
        numeric_cols = [
            'age', 'payment_on_time_rate', 'transaction_regularity_score',
            'dispute_rate', 'months_with_provider', 'avg_monthly_airtime_tnd',
            'location_consistency_score', 'requested_amount_tnd', 
            'term_months', 'TARGET'
        ]
        
        corr_df = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, 
                   cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        output_file = self.figures_dir / 'correlation_heatmap.png'
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_file}")
    
    def generate_atb_demo_report(self, df, metrics):
        """Generate comprehensive ATB demo report"""
        
        print("Generating ATB Demo Report...")
        
        # Calculate summary statistics
        total_loans = len(df)
        default_rate = df['TARGET'].mean()
        categories = df['loan_product_category'].value_counts().to_dict()
        avg_loan = df['requested_amount_tnd'].mean()
        
        report_lines = [
            "# Tunisia AI Credit Scoring Demo",
            "## Arab Tunisian Bank (ATB) - Hackathon Submission",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"This demonstration showcases an AI-powered credit scoring system for Arab Tunisian Bank (ATB) that leverages **alternative financial indicators** to assess loan eligibility, default risk, and affordability for Tunisian borrowers.",
            "",
            "### Key Innovation",
            "Traditional credit scoring relies heavily on formal credit history, which many Tunisians lack. Our system uses:",
            "- **Telecom data**: Payment history, usage patterns, tenure",
            "- **Mobile money**: Transaction frequency and volumes (D17, e-dinar)",
            "- **Marketplace activity**: E-commerce ratings and transaction history",
            "- **Remittance flows**: International transfers as income stability signal",
            "- **Location stability**: Geographic consistency from mobile usage",
            "",
            "### Business Impact",
            f"- **Dataset**: {total_loans:,} loan applications analyzed",
            f"- **Default Rate**: {default_rate:.1%} (realistic Tunisia market conditions)",
            f"- **Average Loan**: {avg_loan:,.0f} TND",
            "- **Product Coverage**: Housing, Vehicle, Consumer, Business, Corporate",
            "",
            "---",
            "",
            "## Technical Performance",
            "",
            "### Model 1: Eligibility Assessment",
            f"- **AUC-ROC**: {metrics['eligibility']['auc_roc']:.4f}",
            f"- **Optimal Threshold**: {metrics['eligibility']['best_threshold']:.2f}",
            f"- **F1 Score**: {metrics['eligibility']['best_f1']:.4f}",
            "- **Purpose**: Determine if applicant should be considered for loan product",
            "",
            "### Model 2: Default Risk Prediction",
            f"- **AUC-ROC**: {metrics['default_risk']['auc_roc']:.4f}",
            f"- **PR-AUC**: {metrics['default_risk']['pr_auc']:.4f}",
            f"- **Recall (Default Class)**: {metrics['default_risk']['recall_default_class']:.4f}",
            f"- **Optimal Threshold**: {metrics['default_risk']['best_threshold']:.2f} (cost-optimized)",
            "- **Class Imbalance Handling**: SMOTE + class weights + threshold optimization",
            "- **Business Cost Optimization**: FN cost = 10x FP cost",
            "",
            "### Model 3: Affordability Classification",
            f"- **Macro F1 Score**: {metrics['affordability']['macro_f1']:.4f}",
            f"- **Balanced Accuracy**: {metrics['affordability']['balanced_accuracy']:.4f}",
            f"- **Classes**: {', '.join(metrics['affordability']['classes'])}",
            "- **Purpose**: Estimate appropriate loan amounts based on income proxies",
            "",
            "---",
            "",
            "## Dataset Characteristics",
            "",
            "### Loan Product Distribution",
        ]
        
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            pct = count / total_loans * 100
            report_lines.append(f"- **{category}**: {count:,} loans ({pct:.1f}%)")
        
        report_lines.extend([
            "",
            "### Alternative Data Coverage",
            f"- **Mobile Money Users**: {df['mobile_money_account'].sum():,} ({df['mobile_money_account'].mean()*100:.1f}%)",
            f"- **Remittance Recipients**: {df['receives_remittances'].sum():,} ({df['receives_remittances'].mean()*100:.1f}%)",
            f"- **Marketplace Sellers**: {df['marketplace_seller_rating'].notna().sum():,} ({df['marketplace_seller_rating'].notna().mean()*100:.1f}%)",
            "",
            "---",
            "",
            "## Model Explainability",
            "",
            "Our system provides transparent, explainable decisions using SHAP (SHapley Additive exPlanations):",
            "",
            "- **Global Explanations**: Identify which features drive predictions across all applicants",
            "- **Local Explanations**: Show exactly why a specific applicant received their score",
            "- **Regulatory Compliance**: Supports transparency requirements for AI lending",
            "",
            "See `explainability_notes.md` for detailed analysis.",
            "",
            "---",
            "",
            "## Tunisia Market Relevance",
            "",
            "### ATB Product Alignment",
            "The system handles all major ATB loan categories with product-specific constraints:",
            "",
            "| Product | Max LTV | Max Term | Cap (TND) | Appraisal |",
            "|---------|---------|----------|-----------|-----------|",
            "| Housing | 80% | 300 mo | None | Yes |",
            "| Vehicle | 85% | 84 mo | 150,000 | Yes |",
            "| Consumer | 100% | 60 mo | 50,000 | No |",
            "| Business | 70% | 120 mo | None | Yes |",
            "| Corporate | 75% | 180 mo | None | Yes |",
            "",
            "### Data Sources (Tunisia Context)",
            "- **Telecom**: Ooredoo, Orange, Tunisie Telecom",
            "- **Mobile Money**: D17 (ATB's mobile money), e-dinar",
            "- **E-commerce**: Jumia, Tunisie Net, local marketplaces",
            "- **Remittances**: International transfers (Europe, Gulf countries)",
            "- **Governorates**: All 24 Tunisia governorates represented",
            "",
            "---",
            "",
            "## Leakage Prevention & Data Quality",
            "",
            "Rigorous data quality checks ensure model integrity:",
            "",
            "- ✓ **No Target Leakage**: All high-correlation features removed",
            "- ✓ **No Post-Outcome Features**: Features only use pre-decision data",
            "- ✓ **No Affordability Leakage**: Income bands derived from multiple proxies",
            "- ✓ **Metadata Isolation**: Product constraints not used in training",
            "",
            "See `data_quality_report.md` for complete audit results.",
            "",
            "---",
            "",
            "## Implementation Roadmap",
            "",
            "### Phase 1: Pilot (3 months)",
            "1. Deploy shadow scoring alongside existing underwriting",
            "2. Collect feedback from loan officers",
            "3. Validate predictions against actual outcomes",
            "",
            "### Phase 2: Integration (6 months)",
            "1. Establish data partnerships (telecom, mobile money)",
            "2. Build real-time API integrations",
            "3. Train loan officers on AI-assisted decisioning",
            "",
            "### Phase 3: Scale (12 months)",
            "1. Full production deployment",
            "2. Expand to underbanked segments",
            "3. Continuous model monitoring and retraining",
            "",
            "---",
            "",
            "## Risk Mitigation",
            "",
            "### Model Risks",
            "- **Data drift**: Monitor feature distributions quarterly",
            "- **Concept drift**: Retrain models semi-annually",
            "- **Bias**: Regular fairness audits across demographics",
            "",
            "### Operational Risks",
            "- **Data quality**: Automated validation pipelines",
            "- **System downtime**: Fallback to manual underwriting",
            "- **Regulatory changes**: Flexible architecture for compliance",
            "",
            "---",
            "",
            "## Conclusion",
            "",
            "This demonstration proves the viability of AI-powered credit scoring using alternative data for the Tunisian market. The system:",
            "",
            "✓ Achieves strong predictive performance without traditional credit bureaus",
            "✓ Provides transparent, explainable decisions",
            "✓ Aligns with ATB's product portfolio and business processes",
            "✓ Leverages Tunisia-specific data sources",
            "✓ Follows best practices for data quality and leakage prevention",
            "",
            "**Next Steps**: Partner with ATB to pilot on real data and refine for production deployment.",
            "",
            "---",
            "",
            "## Appendices",
            "",
            "### Generated Artifacts",
            "- `outputs/datasets/tunisia_loan_data.csv` - Full dataset",
            "- `outputs/datasets/tunisia_loan_data_train.csv` - Safe training data",
            "- `outputs/models/credit_engine.pkl` - Trained models",
            "- `reports/model_metrics.json` - Detailed metrics",
            "- `reports/explainability_notes.md` - SHAP analysis",
            "- `outputs/figures/` - All visualizations",
            "",
            "### Contact",
            "For questions or collaboration: [Your Contact Info]",
            "",
            "---",
            "",
            f"**Report Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ])
        
        report_content = "\n".join(report_lines)
        
        report_file = self.reports_dir / 'ATB_Demo_Report.md'
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        print(f"✓ Saved: {report_file}")
    
    def run(self):
        """Execute complete reporting pipeline"""
        
        print("=" * 80)
        print("STEP E: GRAPHICS & REPORTING")
        print("=" * 80)
        
        # Load data
        print("\nLoading data and metrics...")
        df, metrics = self.load_data()
        print(f"✓ Loaded {len(df)} records")
        
        # Generate graphics
        print("\nGenerating visualizations...")
        self.create_schema_diagram()
        self.create_data_snapshot(df)
        self.create_target_distribution(df)
        self.create_loan_amount_by_category(df)
        self.create_correlation_heatmap(df)
        
        # Generate final report
        print("\nGenerating ATB demo report...")
        self.generate_atb_demo_report(df, metrics)
        
        print("\n" + "=" * 80)
        print("STEP E COMPLETE")
        print("=" * 80)
        print("\nGenerated artifacts:")
        print("- System schema diagram")
        print("- Data snapshot table")
        print("- Target distribution plot")
        print("- Loan amount by category plot")
        print("- Correlation heatmap")
        print("- Comprehensive ATB demo report")


if __name__ == '__main__':
    reporter = ReportGenerator()
    reporter.run()
