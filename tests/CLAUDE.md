# Tests Directory - Context for Claude

## Purpose
Unit tests and integration tests for the malaria detection pipeline.

## Test Structure
```
tests/
├── unit/                 # Unit tests for individual modules
├── integration/          # Integration tests for pipeline stages  
├── fixtures/            # Test data and mock objects
└── conftest.py          # Pytest configuration
```

## Test Coverage Areas
- **Data Processing** - Preprocessing, integration, augmentation functions
- **Utility Functions** - Download, image processing, annotation utilities
- **Pipeline Integration** - End-to-end pipeline testing
- **Model Loading** - Model initialization and inference testing

## Testing Framework
- **pytest** - Primary testing framework
- **fixtures** - Reusable test data and mock objects
- **CI/CD** - Automated testing on commits

## Status
Ready for test development after pipeline stabilization.