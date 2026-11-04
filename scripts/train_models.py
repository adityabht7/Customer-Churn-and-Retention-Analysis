"""
Script to train candidate models on the benchmark dataset, report comparative metrics,
and persist the champion model pipeline to disk.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data_loader import load_churn_data
from src.churn_model import train_and_benchmark_models, save_pipeline, MODEL_FILE

def main():
    print("Loading customer churn dataset...")
    df = load_churn_data()
    print(f"Loaded {len(df):,} customer records.")

    print("\nTraining and benchmarking classification models...")
    benchmark_out = train_and_benchmark_models(df)
    summary = benchmark_out["benchmark_summary"]

    print("\n" + "=" * 65)
    print("MODEL BENCHMARK PERFORMANCE RESULTS")
    print("=" * 65)
    print(f"{'Model Name':<22} | {'ROC-AUC':<8} | {'F1-Score':<8} | {'Recall':<8} | {'Precision':<8}")
    print("-" * 65)

    for name, m in summary["models"].items():
        print(f"{name:<22} | {m['roc_auc']:.4f}   | {m['f1_score']:.4f}   | {m['recall']:.4f}   | {m['precision']:.4f}")

    print("=" * 65)
    print(f"Champion Model: {summary['best_model_name']} (ROC-AUC: {summary['best_roc_auc']:.4f})")

    print("\nTop 8 Churn Predictive Features:")
    for i, item in enumerate(summary["feature_importances"][:8], 1):
        print(f"  {i}. {item['feature']:<30} (Weight: {item['importance']:.4f})")

    print(f"\nPersisting champion pipeline to '{MODEL_FILE}'...")
    save_pipeline(benchmark_out["best_pipeline"], MODEL_FILE)
    print("Model persisted successfully! Ready for real-time inference.")

if __name__ == "__main__":
    main()
