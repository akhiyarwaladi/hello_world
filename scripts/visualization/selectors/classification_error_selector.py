#!/usr/bin/env python3
"""
Classification Error Selector - Select Interesting Classification Errors

Selects images with classification errors for publication/analysis:
- All wrong: All boxes misclassified
- High confidence errors: Wrong but confident (interesting failure)
- Mixed: Some correct, some wrong (partial success)
- Perfect: All correct (for comparison)

Strategy: Select top N cases per error type per dataset
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd

from .base_selector import BaseSelector


class ClassificationErrorSelector(BaseSelector):
    """
    Select interesting classification error cases.

    Error Categories:
    1. All Wrong (score=3) - Complete failure
    2. High Confidence Errors (score=5, conf>0.75) - Confident but wrong
    3. Mixed Results (score=6) - Some correct, some wrong
    4. Perfect (score=9-10) - All correct (for comparison)

    Example:
        selector = ClassificationErrorSelector(top_n=5, conf_threshold=0.75)
        selected = selector.select_from_csv('classification_metadata_images.csv')
    """

    def __init__(self, top_n: int = 5, conf_threshold: float = 0.75, include_perfect: bool = True):
        """
        Initialize classification error selector.

        Args:
            top_n: Number of top cases per category
            conf_threshold: Confidence threshold for "high confidence errors"
            include_perfect: Whether to include perfect classifications for comparison
        """
        super().__init__(top_n=top_n)
        self.conf_threshold = conf_threshold
        self.include_perfect = include_perfect

    def select(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select interesting classification error cases.

        Args:
            metadata_df: DataFrame with columns:
                - image_name
                - n_boxes
                - n_correct
                - n_incorrect
                - accuracy
                - avg_confidence
                - status
                - paper_score

        Returns:
            DataFrame with selected cases
        """
        selected = []

        # Convert string accuracy to float if needed
        if metadata_df['accuracy'].dtype == 'object':
            metadata_df['accuracy'] = pd.to_numeric(metadata_df['accuracy'], errors='coerce')
        if metadata_df['avg_confidence'].dtype == 'object':
            metadata_df['avg_confidence'] = pd.to_numeric(metadata_df['avg_confidence'], errors='coerce')

        # Category 1: All Wrong (complete failure)
        all_wrong = metadata_df[
            (metadata_df['status'] == 'All wrong')
        ].copy()

        if not all_wrong.empty:
            # Sort by number of incorrect boxes (descending), then by confidence
            all_wrong = all_wrong.sort_values(
                ['n_incorrect', 'avg_confidence'],
                ascending=[False, False]
            ).head(self.top_n)
            all_wrong['error_category'] = 'All Wrong'
            selected.append(all_wrong)

        # Category 2: High Confidence Errors (wrong but confident - interesting!)
        high_conf_errors = metadata_df[
            (metadata_df['n_incorrect'] > 0) &
            (metadata_df['avg_confidence'] > self.conf_threshold)
        ].copy()

        if not high_conf_errors.empty:
            # Sort by confidence (descending) - most confident errors first
            high_conf_errors = high_conf_errors.sort_values(
                'avg_confidence',
                ascending=False
            ).head(self.top_n)
            high_conf_errors['error_category'] = f'High Confidence Errors (>{self.conf_threshold})'
            selected.append(high_conf_errors)

        # Category 3: Mixed Results (some correct, some wrong)
        mixed = metadata_df[
            (metadata_df['status'] == 'Mixed (some correct, some wrong)')
        ].copy()

        if not mixed.empty:
            # Sort by number of errors, then by confidence
            mixed = mixed.sort_values(
                ['n_incorrect', 'avg_confidence'],
                ascending=[False, False]
            ).head(self.top_n)
            mixed['error_category'] = 'Mixed Results'
            selected.append(mixed)

        # Category 4: Low Confidence Correct (uncertain but right - interesting edge case)
        low_conf_correct = metadata_df[
            (metadata_df['n_correct'] == metadata_df['n_boxes']) &
            (metadata_df['avg_confidence'] < 0.6) &
            (metadata_df['n_boxes'] > 0)
        ].copy()

        if not low_conf_correct.empty:
            # Sort by confidence (ascending) - most uncertain first
            low_conf_correct = low_conf_correct.sort_values(
                'avg_confidence',
                ascending=True
            ).head(self.top_n)
            low_conf_correct['error_category'] = 'Low Confidence Correct (<0.6)'
            selected.append(low_conf_correct)

        # Category 5: Perfect Classifications (for comparison)
        if self.include_perfect:
            perfect = metadata_df[
                (metadata_df['status'] == 'All correct')
            ].copy()

            if not perfect.empty:
                # Sort by confidence (descending)
                perfect = perfect.sort_values('avg_confidence', ascending=False).head(self.top_n)
                perfect['error_category'] = 'Perfect Classification'
                selected.append(perfect)

        # Combine all categories
        if selected:
            result = pd.concat(selected, ignore_index=True)

            # Add selection metadata
            result['selector_type'] = 'classification_error'
            result['top_n'] = self.top_n
            result['conf_threshold'] = self.conf_threshold

            return result
        else:
            return pd.DataFrame()

    def get_selection_criteria(self) -> Dict[str, Any]:
        """Return selection criteria description."""
        return {
            'type': 'classification_errors',
            'categories': [
                'All Wrong',
                f'High Confidence Errors (>{self.conf_threshold})',
                'Mixed Results',
                'Low Confidence Correct (<0.6)',
                'Perfect Classification' if self.include_perfect else None
            ],
            'top_n_per_category': self.top_n,
            'conf_threshold': self.conf_threshold,
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
        # Convert strings to numeric if needed
        if metadata_df['avg_confidence'].dtype == 'object':
            metadata_df['avg_confidence'] = pd.to_numeric(metadata_df['avg_confidence'], errors='coerce')

        return {
            'all_wrong': len(metadata_df[metadata_df['status'] == 'All wrong']),
            'high_conf_errors': len(metadata_df[
                (metadata_df['n_incorrect'] > 0) &
                (metadata_df['avg_confidence'] > self.conf_threshold)
            ]),
            'mixed': len(metadata_df[metadata_df['status'] == 'Mixed (some correct, some wrong)']),
            'low_conf_correct': len(metadata_df[
                (metadata_df['n_correct'] == metadata_df['n_boxes']) &
                (metadata_df['avg_confidence'] < 0.6) &
                (metadata_df['n_boxes'] > 0)
            ]),
            'perfect': len(metadata_df[metadata_df['status'] == 'All correct']),
            'total': len(metadata_df)
        }

    def get_confusion_patterns(self, boxes_metadata_csv: Path) -> pd.DataFrame:
        """
        Analyze confusion patterns from box-level metadata.

        Args:
            boxes_metadata_csv: Path to classification_metadata_boxes.csv

        Returns:
            DataFrame with confusion matrix data
        """
        if not boxes_metadata_csv.exists():
            return pd.DataFrame()

        boxes_df = pd.read_csv(boxes_metadata_csv)

        # Create confusion matrix
        confusion = boxes_df[boxes_df['correct'] == 'No'].groupby(
            ['gt_class', 'pred_class']
        ).size().reset_index(name='count')

        return confusion.sort_values('count', ascending=False)
