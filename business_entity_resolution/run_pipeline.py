"""
Run Pipeline: CLI entry point for the Business Entity Resolution System.
Reproduces end-to-end entity resolution workflow from raw TSVs to final submission ZIP.
"""

import sys
import os
import argparse

# Ensure parent directory is in python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from src.pipeline import EntityResolutionPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run Business Entity Resolution Pipeline (Challenge Edition)."
    )
    parser.add_argument(
        "--config",
        "-c",
        default="configs/config.yaml",
        help="Path to YAML configuration file (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Run only data profiling phase.",
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help="Run test inference directly using pre-trained model.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Limit number of test S1 records for fast validation run.",
    )
    args = parser.parse_args()

    pipeline = EntityResolutionPipeline(config_path=args.config)

    if args.profile_only:
        pipeline.run_profiling()
        print("Data profiling completed successfully.")
        return 0

    if args.inference_only:
        print("Running Inference-Only using pre-trained model...")
        zip_path = pipeline.run_inference(test_s1_limit=args.test_limit)
        print(f"\nFinal submission successfully packaged into: {zip_path}")
        return 0

    # 1. Profile data
    pipeline.run_profiling()

    # 2. Train & optimize threshold
    train_metrics = pipeline.train_and_validate()
    print("\n--- Training & Validation Summary ---")
    print(f"Optimal Threshold: {pipeline.optimal_threshold}")
    print(f"Validation F0.5:   {train_metrics['threshold_optimization']['best_macro_f05']:.4f}")

    # 3. Test Inference & Submission Packaging
    zip_path = pipeline.run_inference(test_s1_limit=args.test_limit)
    print(f"\nFinal submission successfully packaged into: {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
