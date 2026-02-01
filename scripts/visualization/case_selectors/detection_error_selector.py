#!/usr/bin/env python3
"""
Detection Error Selector - Select Interesting Detection Errors

Selects images with detection errors for publication/analysis:
- False Positives (FP): Model detects parasites that don't exist
- False Negatives (FN): Model misses real parasites
- Mixed errors: Both FP and FN in same image
- Perfect detections: For comparison

Strategy: Select top N cases per error type per dataset
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from .base_selector import BaseSelector


class DetectionErrorSelector(BaseSelector):
    """
    Select interesting detection error cases.

    Error Categories:
    1. False Positives Only (FP) - score=3
    2. False Negatives Only (FN) - score=4
    3. Mixed (FP + FN) - score=5
    4. Perfect detections - score=9-10 (for comparison)

    Example:
        selector = DetectionErrorSelector(top_n=5)
        selected = selector.select_from_csv('detection_metadata.csv')
    """

    def __init__(self, top_n: int = 5, include_perfect: bool = True):
        """
        Initialize detection error selector.

        Args:
            top_n: Number of top cases per category
            include_perfect: Whether to include perfect detections for comparison
        """
        super().__init__(top_n=top_n)
        self.include_perfect = include_perfect

    def select(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select interesting detection error cases.

        Args:
            metadata_df: DataFrame with columns:
                - image_name
                - n_false_positives
                - n_false_negatives
                - status
                - paper_score
                - avg_confidence

        Returns:
            DataFrame with selected cases
        """
        selected = []

        # Category 1: False Positives Only (FP)
        fp_only = metadata_df[
            (metadata_df['n_false_positives'] > 0) &
            (metadata_df['n_false_negatives'] == 0)
        ].copy()

        if not fp_only.empty:
            # Sort by number of FP (descending), then by confidence
            fp_only = fp_only.sort_values(
                ['n_false_positives', 'avg_confidence'],
                ascending=[False, False]
            ).head(self.top_n)
            fp_only['error_category'] = 'False Positives (FP)'
            selected.append(fp_only)

        # Category 2: False Negatives Only (FN)
        fn_only = metadata_df[
            (metadata_df['n_false_negatives'] > 0) &
            (metadata_df['n_false_positives'] == 0)
        ].copy()

        if not fn_only.empty:
            # Sort by number of FN (descending)
            fn_only = fn_only.sort_values(
                'n_false_negatives',
                ascending=False
            ).head(self.top_n)
            fn_only['error_category'] = 'False Negatives (FN)'
            selected.append(fn_only)

        # Category 3: Mixed Errors (FP + FN)
        mixed = metadata_df[
            (metadata_df['n_false_positives'] > 0) &
            (metadata_df['n_false_negatives'] > 0)
        ].copy()

        if not mixed.empty:
            # Sort by total errors (FP + FN)
            mixed['total_errors'] = mixed['n_false_positives'] + mixed['n_false_negatives']
            mixed = mixed.sort_values('total_errors', ascending=False).head(self.top_n)
            mixed['error_category'] = 'Mixed (FP + FN)'
            selected.append(mixed)

        # Category 4: Perfect Detections (for comparison)
        if self.include_perfect:
            perfect = metadata_df[
                (metadata_df['status'] == 'Perfect')
            ].copy()

            if not perfect.empty:
                # Sort by confidence (descending)
                perfect = perfect.sort_values('avg_confidence', ascending=False).head(self.top_n)
                perfect['error_category'] = 'Perfect Detection'
                selected.append(perfect)

        # Combine all categories
        if selected:
            result = pd.concat(selected, ignore_index=True)

            # Add selection metadata
            result['selector_type'] = 'detection_error'
            result['top_n'] = self.top_n

            return result
        else:
            return pd.DataFrame()

    def get_selection_criteria(self) -> Dict[str, Any]:
        """Return selection criteria description."""
        return {
            'type': 'detection_errors',
            'categories': [
                'False Positives (FP)',
                'False Negatives (FN)',
                'Mixed (FP + FN)',
                'Perfect Detection' if self.include_perfect else None
            ],
            'top_n_per_category': self.top_n,
            'include_perfect': self.include_perfect
        }

    def get_category_stats(self, metadata_df: pd.DataFrame) -> Dict[str, int]:
        """
        Get statistics per error category.

        Args:
            metadata_df: Full metadata DataFrame

        Returns:
            Dict with counts per category
        """
        return {
            'fp_only': len(metadata_df[
                (metadata_df['n_false_positives'] > 0) &
                (metadata_df['n_false_negatives'] == 0)
            ]),
            'fn_only': len(metadata_df[
                (metadata_df['n_false_negatives'] > 0) &
                (metadata_df['n_false_positives'] == 0)
            ]),
            'mixed': len(metadata_df[
                (metadata_df['n_false_positives'] > 0) &
                (metadata_df['n_false_negatives'] > 0)
            ]),
            'perfect': len(metadata_df[metadata_df['status'] == 'Perfect']),
            'total': len(metadata_df)
        }
