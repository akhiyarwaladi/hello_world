#!/usr/bin/env python3
"""
Base Reporter - Abstract Interface for Report Generation

Defines the interface that all concrete reporters must implement.
Makes it easy to add new report formats without modifying existing code.

Design Pattern: Strategy Pattern
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class BaseReporter(ABC):
    """
    Abstract base class for all report generators.

    Architecture:
    - Subclasses implement generate() method for specific format
    - Common functionality in base class
    - Consistent interface for all reporters

    Example:
        class MyCustomReporter(BaseReporter):
            def generate(self, data: Any, output_path: Path):
                # Your custom format generation
                pass
    """

    def __init__(self, title: str = "Visualization Report"):
        """
        Initialize reporter.

        Args:
            title: Report title
        """
        self.title = title

    @abstractmethod
    def generate(self, data: Any, output_path: Path) -> bool:
        """
        Generate report in specific format.

        This is the main method that subclasses must implement.

        Args:
            data: Data to be reported (format depends on reporter type)
            output_path: Path for output file

        Returns:
            True if successful, False otherwise
        """
        pass

    def ensure_output_dir(self, output_path: Path):
        """Ensure output directory exists."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

    def get_format(self) -> str:
        """
        Get report format name.

        Returns:
            Format name (e.g., 'CSV', 'Markdown', 'JSON')
        """
        return self.__class__.__name__.replace('Reporter', '')
