#!/bin/bash
# Cleanup Obsolete Visualization Scripts
#
# This script moves obsolete/superseded scripts to archive/

ARCHIVE_DIR="archive/scripts_visualization_old_$(date +%Y%m%d)"
mkdir -p "$ARCHIVE_DIR"

echo "==================================================================="
echo "CLEANING UP OBSOLETE VISUALIZATION SCRIPTS"
echo "==================================================================="
echo "Archive location: $ARCHIVE_DIR"
echo ""

# OBSOLETE SCRIPTS - Superseded by generate_all_centralized.py
echo "[1] Obsolete main scripts (replaced by generate_all_centralized.py):"
mv -v scripts/visualization/consolidate_all_outputs.py "$ARCHIVE_DIR/"
mv -v scripts/visualization/generate_all_test_visualizations.py "$ARCHIVE_DIR/"
mv -v scripts/visualization/generate_all_detection_classification_figures.py "$ARCHIVE_DIR/"

# OLD VERSIONS - No metadata export
echo ""
echo "[2] Old versions without metadata:"
mv -v scripts/visualization/generate_detection_only.py "$ARCHIVE_DIR/"
mv -v scripts/visualization/generate_classification_only.py "$ARCHIVE_DIR/"
mv -v scripts/visualization/generate_detection_classification_figures.py "$ARCHIVE_DIR/"

# SUPERSEDED - Old training curves version
echo ""
echo "[3] Superseded by modular version:"
mv -v scripts/visualization/generate_professional_training_curves_final.py "$ARCHIVE_DIR/"

# ONE-OFF RUNNERS - Not part of main pipeline
echo ""
echo "[4] One-off runner scripts:"
mv -v scripts/visualization/run_detection_classification_on_experiment.py "$ARCHIVE_DIR/" 2>/dev/null || true
mv -v scripts/visualization/run_improved_gradcam_on_experiments.py "$ARCHIVE_DIR/" 2>/dev/null || true

# GT-ONLY GENERATORS - May not be needed (used by separate model viz)
echo ""
echo "[5] GT-only generators (check if still needed):"
# Keeping these for now as they might be used by other scripts
# mv -v scripts/visualization/generate_gt_detection.py "$ARCHIVE_DIR/"
# mv -v scripts/visualization/generate_gt_classification.py "$ARCHIVE_DIR/"
echo "   [SKIP] Keeping GT generators (may be used by pipeline)"

# AUGMENTATION - Separate feature, keep in main scripts
echo ""
echo "[6] Augmentation scripts:"
echo "   [KEEP] generate_combined_4datasets_augmentation.py"
echo "   [KEEP] generate_compact_augmentation_figures.py"

# GRADCAM - Separate feature, keep in main scripts
echo ""
echo "[7] GradCAM scripts:"
echo "   [KEEP] generate_improved_gradcam.py"

# ARCHITECTURE DIAGRAM - Keep for paper
echo ""
echo "[8] Architecture diagram:"
echo "   [KEEP] generate_pipeline_architecture_diagram.py"

echo ""
echo "==================================================================="
echo "✅ CLEANUP COMPLETE"
echo "==================================================================="
echo "Archived: $ARCHIVE_DIR"
echo ""
echo "REMAINING ACTIVE SCRIPTS:"
ls -1 scripts/visualization/*.py

echo ""
echo "MAIN ENTRY POINT:"
echo "  ⭐ scripts/visualization/generate_all_centralized.py"
echo ""
echo "CORE MODULES:"
echo "  → selectors/ (detection/classification error selection)"
echo "  → reporters/ (CSV/Markdown/JSON output)"
echo "  → generate_training_curves.py"
echo "  → generate_consolidated_confusion_matrices.py"
echo "  → generate_*_with_metadata.py (test visualization generation)"
