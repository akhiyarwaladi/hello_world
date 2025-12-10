#!/usr/bin/env python3
"""
CSV Reporter - Export selected cases to CSV format

Generates CSV files compatible with Excel and pandas.
Perfect for:
- Machine-readable data export
- Further analysis in spreadsheet software
- Integration with data pipelines
"""

from pathlib import Path
from typing import Any
import pandas as pd

from .base_reporter import BaseReporter


class CSVReporter(BaseReporter):
    """
    Generate CSV reports from selected cases.

    Output format: Standard CSV with headers
    Compatible with: Excel, Google Sheets, pandas

    Example:
        reporter = CSVReporter(title="Selected Error Cases")
        reporter.generate(selected_df, 'selected_errors.csv')
    """

    def __init__(self, title: str = "Selected Cases", index: bool = False):
        """
        Initialize CSV reporter.

        Args:
            title: Report title (used in comments if supported)
            index: Whether to include DataFrame index in CSV
        """
        super().__init__(title)
        self.include_index = index

    def generate(self, data: Any, output_path: Path) -> bool:
        """
        Generate CSV report.

        Args:
            data: pandas DataFrame or dict to export
            output_path: Path for output CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            self.ensure_output_dir(output_path)

            # Convert to DataFrame if dict
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, pd.DataFrame):
                df = data
            else:
                print(f"[ERROR] Unsupported data type: {type(data)}")
                return False

            # Export to CSV
            df.to_csv(output_path, index=self.include_index)

            print(f"[SAVED] CSV report → {output_path} ({len(df)} rows)")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to generate CSV: {e}")
            return False

    def generate_multiple(self, data_dict: dict, output_dir: Path) -> int:
        """
        Generate multiple CSV files from dict of DataFrames.

        Args:
            data_dict: Dict mapping filenames to DataFrames
            output_dir: Output directory for CSV files

        Returns:
            Number of successfully generated files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        for filename, data in data_dict.items():
            output_path = output_dir / f"{filename}.csv"
            if self.generate(data, output_path):
                success_count += 1

        return success_count
