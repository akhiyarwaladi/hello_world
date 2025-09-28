#!/usr/bin/env python3
"""
Test script untuk IoU Analysis yang tidak melakukan re-testing
Menggunakan pre-computed training results dari results.csv
"""

import os
import sys
from pathlib import Path

# Add the scripts/analysis directory to Python path
sys.path.append(str(Path(__file__).parent / "scripts" / "analysis"))

from compare_models_performance import MalariaPerformanceAnalyzer

def test_iou_analysis_fast():
    """Test IoU analysis using pre-computed results"""
    print("=== TEST: IoU Analysis from Pre-computed Results ===")

    # Find a sample results.csv file from recent experiments
    results_dir = Path("results")

    # Look for results.csv files in experiment directories
    sample_results_csv = None
    experiment_name = "Unknown"

    if results_dir.exists():
        for exp_dir in results_dir.glob("exp_*"):
            if exp_dir.is_dir():
                # Look for detection results
                detection_results = exp_dir / "detection" / "results.csv"
                if detection_results.exists():
                    sample_results_csv = str(detection_results)
                    experiment_name = exp_dir.name
                    break

                # Alternative path structure
                for subdir in exp_dir.iterdir():
                    if subdir.is_dir():
                        results_csv = subdir / "results.csv"
                        if results_csv.exists():
                            sample_results_csv = str(results_csv)
                            experiment_name = f"{exp_dir.name}_{subdir.name}"
                            break
                if sample_results_csv:
                    break

    if not sample_results_csv:
        print("[ERROR] No sample results.csv found for testing")
        print("[INFO] Run a detection training first to generate results.csv")
        return False

    print(f"[FOUND] Using sample results: {sample_results_csv}")
    print(f"[EXPERIMENT] {experiment_name}")

    # Create test output directory
    test_output = "test_iou_analysis_output"

    # Create analyzer and run the fast IoU analysis
    analyzer = MalariaPerformanceAnalyzer()

    print(f"\n[TEST] Running IoU analysis from pre-computed results...")
    results = analyzer.run_iou_analysis_from_results(
        results_csv_path=sample_results_csv,
        output_dir=test_output,
        experiment_name=experiment_name
    )

    if results:
        print(f"\n[SUCCESS] IoU analysis completed successfully!")
        print(f"[OUTPUT] Results saved to: {test_output}")

        # List generated files
        output_path = Path(test_output)
        if output_path.exists():
            print(f"\n[FILES] Generated files:")
            for file in output_path.iterdir():
                print(f"  - {file.name}")

        return True
    else:
        print(f"\n[ERROR] IoU analysis failed!")
        return False

if __name__ == "__main__":
    success = test_iou_analysis_fast()
    if success:
        print(f"\n[CONCLUSION] Fast IoU analysis works correctly!")
        print(f"[ADVANTAGE] No model loading or re-testing required")
        print(f"[PERFORMANCE] Analysis completes in seconds vs minutes")
    else:
        print(f"\n[CONCLUSION] Test failed - check error messages above")