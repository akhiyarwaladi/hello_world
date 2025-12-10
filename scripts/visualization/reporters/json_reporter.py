#!/usr/bin/env python3
"""
JSON Reporter - Machine-Readable Metadata Export

Generates JSON files for:
- API integration
- Programmatic analysis
- Data pipeline integration
- Archival storage
"""

from pathlib import Path
from typing import Any, Dict
from datetime import datetime
import json
import pandas as pd
import numpy as np

from .base_reporter import BaseReporter


class JSONReporter(BaseReporter):
    """
    Generate JSON reports for machine-readable output.

    Output format: Pretty-printed JSON
    Compatible with: JavaScript, Python, REST APIs, NoSQL databases

    Example:
        reporter = JSONReporter(title="Visualization Metadata")
        reporter.generate(metadata_dict, 'metadata.json')
    """

    def __init__(self, title: str = "Metadata Report", indent: int = 2):
        """
        Initialize JSON reporter.

        Args:
            title: Report title
            indent: JSON indentation level (for pretty printing)
        """
        super().__init__(title)
        self.indent = indent

    def generate(self, data: Any, output_path: Path) -> bool:
        """
        Generate JSON report.

        Args:
            data: Data to export (dict, DataFrame, or JSON-serializable object)
            output_path: Path for output JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            self.ensure_output_dir(output_path)

            # Convert data to JSON-serializable format
            json_data = self._prepare_data(data)

            # Add metadata
            output_data = {
                'title': self.title,
                'generated_at': datetime.now().isoformat(),
                'version': '1.0',
                'data': json_data
            }

            # Write JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=self.indent, default=str)

            print(f"[SAVED] JSON report → {output_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to generate JSON: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _prepare_data(self, data: Any) -> Any:
        """
        Convert data to JSON-serializable format.

        Args:
            data: Input data (DataFrame, dict, list, etc.)

        Returns:
            JSON-serializable object
        """
        # DataFrame → dict
        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient='records')

        # Dict with DataFrames → dict with lists
        elif isinstance(data, dict):
            result = {}
            for key, value in data.items():
                result[key] = self._prepare_data(value)
            return result

        # List with mixed types
        elif isinstance(data, list):
            return [self._prepare_data(item) for item in data]

        # NumPy types → Python native types
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()

        elif isinstance(data, np.ndarray):
            return data.tolist()

        # Path objects → strings
        elif isinstance(data, Path):
            return str(data)

        # Already JSON-serializable
        else:
            return data

    def generate_compact(self, data: Any, output_path: Path) -> bool:
        """
        Generate compact JSON (no indentation) for smaller file size.

        Args:
            data: Data to export
            output_path: Path for output JSON file

        Returns:
            True if successful, False otherwise
        """
        original_indent = self.indent
        self.indent = None

        try:
            result = self.generate(data, output_path)
            return result
        finally:
            self.indent = original_indent
