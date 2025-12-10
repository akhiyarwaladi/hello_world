"""
Reporter Module - Extensible Output Formatters

This module provides a plugin architecture for generating reports
in different formats (CSV, Markdown, JSON, LaTeX, etc.).

Architecture:
- BaseReporter: Abstract base class defining the interface
- Concrete reporters: CSVReporter, MarkdownReporter, JSONReporter, etc.
- Easy to extend: Just inherit BaseReporter and implement generate() method

Usage:
    from reporters import CSVReporter, MarkdownReporter

    csv_reporter = CSVReporter()
    csv_reporter.generate(selected_data, 'output.csv')

    md_reporter = MarkdownReporter()
    md_reporter.generate(analysis_results, 'report.md')
"""

from .base_reporter import BaseReporter
from .csv_reporter import CSVReporter
from .markdown_reporter import MarkdownReporter
from .json_reporter import JSONReporter

__all__ = [
    'BaseReporter',
    'CSVReporter',
    'MarkdownReporter',
    'JSONReporter',
]
