"""
Selector Module - Extensible Error Case Selection

This module provides a plugin architecture for selecting interesting images
from metadata CSVs for publication, reports, or further analysis.

Architecture:
- BaseSelector: Abstract base class defining the interface
- Concrete selectors: DetectionErrorSelector, ClassificationErrorSelector, etc.
- Easy to extend: Just inherit BaseSelector and implement select() method

Usage:
    from selectors import DetectionErrorSelector

    selector = DetectionErrorSelector(top_n=5)
    selected = selector.select_from_csv('detection_metadata.csv')
    selector.save_results('output.csv')
"""

from .base_selector import BaseSelector
from .detection_error_selector import DetectionErrorSelector
from .classification_error_selector import ClassificationErrorSelector

__all__ = [
    'BaseSelector',
    'DetectionErrorSelector',
    'ClassificationErrorSelector',
]
