#!/usr/bin/env python3
"""
Test script untuk validasi pipeline
"""

from pipeline import DataValidator

def main():
    validator = DataValidator()

    print("Testing dataset validation...")

    # Test raw dataset validation
    print("\n" + "="*50)
    print("Testing Raw Dataset Validation")
    print("="*50)

    is_valid, result = validator.validate_raw_dataset("data/raw/mp_idb")
    print(f"Valid: {is_valid}")
    print(f"Stats: {result['stats']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")

    # Test detection dataset validation
    print("\n" + "="*50)
    print("Testing Detection Dataset Validation")
    print("="*50)

    is_valid, result = validator.validate_detection_dataset("data/detection_fixed")
    print(f"Valid: {is_valid}")
    print(f"Stats: {result['stats']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")

    # Test cropped dataset validation
    print("\n" + "="*50)
    print("Testing Cropped Dataset Validation")
    print("="*50)

    is_valid, result = validator.validate_cropped_dataset("data/classification_crops")
    print(f"Valid: {is_valid}")
    print(f"Stats: {result['stats']}")
    if result['errors']:
        print(f"Errors: {result['errors']}")
    if result['warnings']:
        print(f"Warnings: {result['warnings']}")

if __name__ == "__main__":
    main()