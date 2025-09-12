# Code Style and Conventions

## Python Style
- **PEP 8 compliant**: Standard Python style guide
- **Type hints**: Used extensively with `typing` module
- **Docstrings**: Google/NumPy style docstrings for classes and methods
- **Import organization**: Standard library, third-party, local imports
- **Class naming**: PascalCase (e.g., `MalariaDatasetDownloader`)
- **Function naming**: snake_case (e.g., `download_with_progress`)
- **Variable naming**: snake_case with descriptive names
- **Constants**: UPPER_CASE

## File Organization
- **Numbered scripts**: Sequential pipeline (01-06)
- **Class-based architecture**: Main functionality in classes
- **Utility functions**: Helper methods within classes
- **Configuration driven**: YAML files for parameters
- **Path handling**: Using `pathlib.Path` consistently
- **Error handling**: Try-except blocks for external operations

## Documentation
- **README.md**: Comprehensive project documentation
- **Inline comments**: Minimal, self-documenting code preferred
- **Configuration examples**: Sample YAML files provided
- **Usage examples**: Command line usage in README

## Dependencies
- **Requirements.txt**: Pinned versions with >= constraints
- **Virtual environment**: Isolated Python environment
- **Setup script**: Automated environment setup