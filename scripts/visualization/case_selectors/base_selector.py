#!/usr/bin/env python3
"""
Base Selector - Abstract Interface for Error Case Selection

Defines the interface that all concrete selectors must implement.
Makes it easy to add new selection strategies without modifying existing code.

Design Pattern: Strategy Pattern + Template Method
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd


class BaseSelector(ABC):
    """
    Abstract base class for all error case selectors.

    Architecture:
    - Subclasses implement select() method with custom logic
    - Common functionality (loading, saving) in base class
    - Consistent interface for all selectors

    Example:
        class MyCustomSelector(BaseSelector):
            def select(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
                # Your custom selection logic
                return filtered_df
    """

    def __init__(self, top_n: int = 5, min_score: Optional[int] = None):
        """
        Initialize selector with common parameters.

        Args:
            top_n: Number of top cases to select per category
            min_score: Minimum paper_score to consider (None = no filter)
        """
        self.top_n = top_n
        self.min_score = min_score
        self.results = []  # Store selected cases

    @abstractmethod
    def select(self, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Select interesting cases from metadata DataFrame.

        This is the main method that subclasses must implement.

        Args:
            metadata_df: DataFrame with metadata columns
                (image_name, status, paper_score, etc.)

        Returns:
            DataFrame with selected cases (subset of input)
        """
        pass

    @abstractmethod
    def get_selection_criteria(self) -> Dict[str, Any]:
        """
        Return selection criteria as dict for documentation.

        Returns:
            Dict with criteria description
            Example: {'type': 'detection_errors', 'categories': ['FP', 'FN']}
        """
        pass

    def select_from_csv(self, csv_path: Path) -> pd.DataFrame:
        """
        Load metadata CSV and run selection.

        Args:
            csv_path: Path to metadata CSV file

        Returns:
            DataFrame with selected cases
        """
        metadata_df = pd.read_csv(csv_path)

        # Apply min_score filter if specified
        if self.min_score is not None:
            if 'paper_score' in metadata_df.columns:
                metadata_df = metadata_df[metadata_df['paper_score'] >= self.min_score]

        # Run custom selection logic
        selected = self.select(metadata_df)

        # Store results
        self.results.append({
            'source': str(csv_path),
            'total_images': len(metadata_df),
            'selected': len(selected),
            'criteria': self.get_selection_criteria()
        })

        return selected

    def select_from_multiple_csvs(self, csv_paths: List[Path]) -> pd.DataFrame:
        """
        Load multiple metadata CSVs and run selection on combined data.

        Args:
            csv_paths: List of paths to metadata CSV files

        Returns:
            DataFrame with selected cases from all CSVs
        """
        all_selected = []

        for csv_path in csv_paths:
            try:
                selected = self.select_from_csv(csv_path)

                # Add source information
                if not selected.empty:
                    selected = selected.copy()
                    selected['source_csv'] = str(csv_path.name)
                    all_selected.append(selected)

            except Exception as e:
                print(f"[WARNING] Failed to process {csv_path}: {e}")

        if all_selected:
            return pd.concat(all_selected, ignore_index=True)
        else:
            return pd.DataFrame()

    def save_results(self, output_path: Path, selected_df: pd.DataFrame):
        """
        Save selected cases to CSV.

        Args:
            output_path: Path for output CSV
            selected_df: DataFrame with selected cases
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        selected_df.to_csv(output_path, index=False)

        print(f"\n[SAVED] {len(selected_df)} selected cases → {output_path}")

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of selection results.

        Returns:
            Dict with summary statistics
        """
        if not self.results:
            return {'message': 'No selections performed yet'}

        total_processed = sum(r['total_images'] for r in self.results)
        total_selected = sum(r['selected'] for r in self.results)

        return {
            'selector_type': self.__class__.__name__,
            'criteria': self.get_selection_criteria(),
            'files_processed': len(self.results),
            'total_images_processed': total_processed,
            'total_selected': total_selected,
            'selection_rate': f"{total_selected/total_processed*100:.1f}%" if total_processed > 0 else "0%",
            'top_n': self.top_n,
            'min_score': self.min_score
        }

    def print_summary(self):
        """Print human-readable summary."""
        summary = self.get_summary()

        print("\n" + "="*80)
        print(f"SELECTOR SUMMARY: {summary['selector_type']}")
        print("="*80)
        print(f"Files processed:     {summary.get('files_processed', 0)}")
        print(f"Images processed:    {summary.get('total_images_processed', 0)}")
        print(f"Cases selected:      {summary.get('total_selected', 0)}")
        print(f"Selection rate:      {summary.get('selection_rate', '0%')}")
        print(f"Top N per category:  {summary.get('top_n', 'N/A')}")
        print(f"Min score filter:    {summary.get('min_score', 'None')}")
        print(f"\nCriteria: {summary.get('criteria', {})}")
        print("="*80)
