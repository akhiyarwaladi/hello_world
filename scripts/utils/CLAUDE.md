# Scripts Utils Directory - Context for Claude

## Purpose
Helper utility modules for the malaria detection pipeline.

## Modules Created & Fixed
- `download_utils.py` - **Complete** - File download with progress, retry logic, integrity checks
- `image_utils.py` - **Complete** - Image processing, quality assessment, transformations  
- `annotation_utils.py` - **Complete** - Annotation parsing, YOLO format conversion, validation
- `__init__.py` - Package initialization with all imports

## Key Functions
### download_utils.py
- `download_with_progress()` - Download with retry and progress bar
- `download_google_drive()` - Google Drive specific downloads  
- `verify_file_integrity()` - MD5 and size verification
- `extract_archive()` - Multi-format archive extraction

### image_utils.py  
- `assess_image_quality()` - Quality metrics (sharpness, contrast, brightness)
- `resize_with_padding()` - Maintain aspect ratio resizing
- `normalize_image()` - CLAHE and standard normalization
- `augment_image()` - Data augmentation using Albumentations

### annotation_utils.py
- `parse_annotation()` - Multi-format annotation parsing
- `convert_to_yolo()` - Convert bounding boxes to YOLO format
- `validate_annotation()` - Annotation quality checks
- `create_label_map()` - Species to class ID mapping

## Status
✅ All utilities complete and functional
✅ Used by main pipeline scripts
✅ Comprehensive error handling and logging included