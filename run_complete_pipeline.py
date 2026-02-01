"""
Tunisia AI Credit Scoring Pipeline - Complete End-to-End Runner
Executes all steps from data generation to final reporting
"""
import sys
from pathlib import Path
import time

# Add scripts directory to path
project_root = Path(__file__).parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

# Import all step modules
from step_a_data_generation import TunisiaDataGenerator
from step_b_data_quality import DataQualityAuditor
from step_c_modeling import UnifiedCreditEngine
from step_d_explainability import ExplainabilityGenerator
from step_e_reporting import ReportGenerator


def print_header(text):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def main():
    """Execute complete Tunisia AI credit scoring pipeline"""
    start_time = time.time()

    print_header("TUNISIA AI CREDIT SCORING PIPELINE")
    print("Arab Tunisian Bank (ATB) - Hackathon Demo")
    print(f"Project Root: {project_root}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        # STEP A: Data Generation
        print_header("STEP A: DATASET GENERATION")
        generator = TunisiaDataGenerator(base_dir=project_root)
        generator.run(n_samples=10000)
        print(f"\n✓ Step A completed successfully in {time.time() - start_time:.1f}s")

        # STEP B: Data Quality Audit
        step_b_start = time.time()
        print_header("STEP B: DATA QUALITY & LEAKAGE AUDIT")
        auditor = DataQualityAuditor(base_dir=project_root)
        auditor.run()
        print(f"\n✓ Step B completed successfully in {time.time() - step_b_start:.1f}s")

        # STEP C: Modeling
        step_c_start = time.time()
        print_header("STEP C: UNIFIED CREDIT ENGINE MODELING")
        engine = UnifiedCreditEngine(base_dir=project_root)
        engine.run()
        print(f"\n✓ Step C completed successfully in {time.time() - step_c_start:.1f}s")

        # STEP D: Explainability (NOT SHAP)
        step_d_start = time.time()
        print_header("STEP D: EXPLAINABILITY (GLOBAL + LOCAL)")
        explainer = ExplainabilityGenerator(base_dir=project_root)
        explainer.run()
        print(f"\n✓ Step D completed successfully in {time.time() - step_d_start:.1f}s")

        # STEP E: Reporting
        step_e_start = time.time()
        print_header("STEP E: GRAPHICS & FINAL REPORTING")
        reporter = ReportGenerator(base_dir=project_root)
        reporter.run()
        print(f"\n✓ Step E completed successfully in {time.time() - step_e_start:.1f}s")

        # Final summary
        total_time = time.time() - start_time
        print_header("PIPELINE COMPLETE")
        print("✓ All steps executed successfully!")
        print(f"\nTotal execution time: {total_time:.1f}s ({total_time/60:.1f} minutes)\n")

        print("Generated Artifacts:")
        print("-" * 80)
        print("\nDatasets:")
        print("  • outputs/datasets/tunisia_loan_data.csv")
        print("  • outputs/datasets/tunisia_loan_data_train.csv")
        print("  • outputs/datasets/tunisia_loan_data_dictionary.csv")
        print("  • outputs/datasets/feature_mapping_reference.csv")

        print("\nModels:")
        print("  • outputs/models/credit_engine.pkl")

        print("\nReports:")
        print("  • reports/FEATURE_MAPPING.md")
        print("  • reports/data_generation_summary.json")
        print("  • reports/data_quality_analysis.json")
        print("  • reports/data_quality_report.md")
        print("  • reports/model_metrics.json")
        print("  • reports/explainability_notes.md")
        print("  • reports/ATB_Demo_Report.md")

        print("\nFigures:")
        print("  • outputs/figures/system_schema.png")
        # Only include if your Step E still generates it:
        # print("  • outputs/figures/data_snapshot.png")
        print("  • outputs/figures/target_distribution.png")
        print("  • outputs/figures/loan_amount_by_category.png")
        print("  • outputs/figures/correlation_heatmap.png")
        print("  • outputs/figures/global_feature_importance.png")
        print("  • outputs/figures/local_explanation_example.png")

        print("\n" + "=" * 80)
        print("Next Steps:")
        print("-" * 80)
        print("1. Review reports/ATB_Demo_Report.md for executive summary")
        print("2. Examine visualizations in outputs/figures/")
        print("3. Check model metrics in reports/model_metrics.json")
        print("4. Review explainability in reports/explainability_notes.md")
        print("5. Validate data quality in reports/data_quality_report.md")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)