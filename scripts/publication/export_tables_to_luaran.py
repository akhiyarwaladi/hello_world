#!/usr/bin/env python3
"""
Export Tables from Pipeline Results to luaran/auto_generated/tables/

This script extracts and formats tables from pipeline experiment results,
copying them to the luaran directory for publication use.

Usage:
    # Auto-detect latest experiment
    python scripts/publication/export_tables_to_luaran.py

    # Specific experiment
    python scripts/publication/export_tables_to_luaran.py --experiment results/optA_20251012_143000

    # Dry-run (show what would be copied without copying)
    python scripts/publication/export_tables_to_luaran.py --dry-run

    # Verbose output
    python scripts/publication/export_tables_to_luaran.py --verbose

Author: Malaria Detection Project
Date: 2025-10-12
"""

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class TableExporter:
    """Export tables from pipeline results to luaran directory."""

    def __init__(self, experiment_dir: Path, output_base: Path, dry_run: bool = False):
        """
        Initialize table exporter.

        Args:
            experiment_dir: Path to experiment directory (e.g., results/optA_20251012_143000)
            output_base: Base output directory (e.g., luaran/auto_generated/tables)
            dry_run: If True, show actions without executing them
        """
        self.experiment_dir = experiment_dir
        self.output_base = output_base
        self.dry_run = dry_run
        self.exported_files: List[Dict[str, str]] = []

        # Define output subdirectories
        self.detection_dir = output_base / "detection"
        self.classification_dir = output_base / "classification"
        self.statistics_dir = output_base / "statistics"

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def setup_directories(self) -> None:
        """Create output directory structure."""
        directories = [
            self.detection_dir,
            self.classification_dir,
            self.statistics_dir,
        ]

        for directory in directories:
            if self.dry_run:
                self.logger.info(f"[DRY-RUN] Would create directory: {directory}")
            else:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {directory}")

    def find_consolidated_analysis_dir(self) -> Optional[Path]:
        """
        Find consolidated_analysis directory.

        Returns:
            Path to consolidated_analysis/cross_dataset_comparison or None
        """
        # Look for consolidated_analysis in experiment directory
        consolidated = self.experiment_dir / "consolidated_analysis" / "cross_dataset_comparison"
        if consolidated.exists():
            return consolidated

        self.logger.warning(f"Consolidated analysis directory not found: {consolidated}")
        return None

    def find_experiment_folders(self) -> List[Path]:
        """
        Find all experiment_* folders (dataset-specific results).

        Returns:
            List of paths to experiment folders
        """
        experiments_dir = self.experiment_dir / "experiments"
        if not experiments_dir.exists():
            self.logger.warning(f"Experiments directory not found: {experiments_dir}")
            return []

        experiment_folders = sorted(experiments_dir.glob("experiment_*"))
        self.logger.info(f"Found {len(experiment_folders)} experiment folders")
        return experiment_folders

    def export_detection_tables(self) -> int:
        """
        Export detection performance tables.

        Returns:
            Number of files exported
        """
        self.logger.info("=" * 80)
        self.logger.info("EXPORTING DETECTION TABLES")
        self.logger.info("=" * 80)

        consolidated_dir = self.find_consolidated_analysis_dir()
        if not consolidated_dir:
            self.logger.warning("Skipping detection tables (no consolidated analysis)")
            return 0

        exported_count = 0

        # Export detection_performance_all_datasets.csv
        source_csv = consolidated_dir / "detection_performance_all_datasets.csv"
        if source_csv.exists():
            dest_csv = self.detection_dir / "Table1_Detection_Performance_All_Datasets.csv"
            exported_count += self._copy_and_format_csv(
                source_csv, dest_csv, "Detection Performance (All Datasets)"
            )

            # Also create Excel version
            dest_xlsx = dest_csv.with_suffix('.xlsx')
            exported_count += self._create_excel_version(dest_csv, dest_xlsx)
        else:
            self.logger.warning(f"Source file not found: {source_csv}")

        # Also export detection_performance_all_datasets.xlsx if exists
        source_xlsx = consolidated_dir / "detection_performance_all_datasets.xlsx"
        if source_xlsx.exists():
            dest_xlsx = self.detection_dir / "Table1_Detection_Performance_All_Datasets_Original.xlsx"
            exported_count += self._copy_file(source_xlsx, dest_xlsx, "Detection Performance Excel (Original)")

        return exported_count

    def export_classification_tables(self) -> int:
        """
        Export classification performance tables including Table 9 variants.

        Returns:
            Number of files exported
        """
        self.logger.info("=" * 80)
        self.logger.info("EXPORTING CLASSIFICATION TABLES")
        self.logger.info("=" * 80)

        exported_count = 0

        # Export from consolidated analysis
        consolidated_dir = self.find_consolidated_analysis_dir()
        if consolidated_dir:
            # Classification performance summary
            for suffix in ['csv', 'xlsx']:
                source = consolidated_dir / f"classification_performance_all_datasets.{suffix}"
                if source.exists():
                    dest = self.classification_dir / f"Table2_Classification_Performance_Summary.{suffix}"
                    exported_count += self._copy_file(source, dest, f"Classification Summary (.{suffix})")

            # Focal Loss results
            source = consolidated_dir / "classification_focal_loss_all_datasets.csv"
            if source.exists():
                dest = self.classification_dir / "Classification_Focal_Loss_All_Datasets.csv"
                exported_count += self._copy_and_format_csv(source, dest, "Focal Loss (All Datasets)")

            # Class-Balanced results
            source = consolidated_dir / "classification_class_balanced_all_datasets.csv"
            if source.exists():
                dest = self.classification_dir / "Classification_Class_Balanced_All_Datasets.csv"
                exported_count += self._copy_and_format_csv(source, dest, "Class-Balanced (All Datasets)")

        # Export Table 9 variants from individual experiment folders
        experiment_folders = self.find_experiment_folders()
        for exp_folder in experiment_folders:
            dataset_name = exp_folder.name.replace("experiment_", "")
            exported_count += self._export_table9_variants(exp_folder, dataset_name)

        return exported_count

    def _export_table9_variants(self, exp_folder: Path, dataset_name: str) -> int:
        """
        Export Table 9 variants for a specific dataset.

        Args:
            exp_folder: Path to experiment folder
            dataset_name: Name of dataset (e.g., 'iml_lifecycle')

        Returns:
            Number of files exported
        """
        exported_count = 0

        # Normalize dataset name for output filename
        dataset_clean = dataset_name.replace("_", "-").upper()

        # Table 9 Focal Loss
        source = exp_folder / "table9_focal_loss.csv"
        if source.exists():
            dest = self.classification_dir / f"Table9_{dataset_clean}_Focal_Loss.csv"
            exported_count += self._copy_and_format_csv(
                source, dest, f"Table 9 Focal Loss ({dataset_name})"
            )

        # Table 9 Class-Balanced
        source = exp_folder / "table9_class_balanced.csv"
        if source.exists():
            dest = self.classification_dir / f"Table9_{dataset_clean}_Class_Balanced.csv"
            exported_count += self._copy_and_format_csv(
                source, dest, f"Table 9 Class-Balanced ({dataset_name})"
            )

        # Table 9 Full (Excel with both sheets)
        source = exp_folder / "table9_classification_pivot.xlsx"
        if source.exists():
            dest = self.classification_dir / f"Table9_{dataset_clean}_Full.xlsx"
            exported_count += self._copy_file(
                source, dest, f"Table 9 Full ({dataset_name})"
            )

        return exported_count

    def export_statistics_tables(self) -> int:
        """
        Export dataset statistics tables.

        Returns:
            Number of files exported
        """
        self.logger.info("=" * 80)
        self.logger.info("EXPORTING STATISTICS TABLES")
        self.logger.info("=" * 80)

        consolidated_dir = self.find_consolidated_analysis_dir()
        if not consolidated_dir:
            self.logger.warning("Skipping statistics tables (no consolidated analysis)")
            return 0

        exported_count = 0

        # Dataset statistics
        source = consolidated_dir / "dataset_statistics_all.csv"
        if source.exists():
            dest = self.statistics_dir / "Dataset_Statistics_All.csv"
            exported_count += self._copy_and_format_csv(
                source, dest, "Dataset Statistics (All Datasets)"
            )

            # Create Excel version
            dest_xlsx = dest.with_suffix('.xlsx')
            exported_count += self._create_excel_version(dest, dest_xlsx)
        else:
            self.logger.warning(f"Source file not found: {source}")

        # Per-class statistics (if exists)
        source = consolidated_dir / "per_class_statistics.csv"
        if source.exists():
            dest = self.statistics_dir / "Per_Class_Statistics.csv"
            exported_count += self._copy_and_format_csv(
                source, dest, "Per-Class Statistics"
            )

            # Create Excel version
            dest_xlsx = dest.with_suffix('.xlsx')
            exported_count += self._create_excel_version(dest, dest_xlsx)

        return exported_count

    def _copy_file(self, source: Path, dest: Path, description: str) -> int:
        """
        Copy a file from source to destination.

        Args:
            source: Source file path
            dest: Destination file path
            description: Description for logging

        Returns:
            1 if successful, 0 otherwise
        """
        if not source.exists():
            self.logger.warning(f"Source file not found: {source}")
            return 0

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] Would copy {description}:")
            self.logger.info(f"  FROM: {source}")
            self.logger.info(f"  TO:   {dest}")
            return 1

        try:
            shutil.copy2(source, dest)
            self.logger.info(f"✓ Copied {description}:")
            self.logger.info(f"  FROM: {source}")
            self.logger.info(f"  TO:   {dest}")

            # Track exported file
            self.exported_files.append({
                "source": str(source),
                "destination": str(dest),
                "description": description,
                "type": "copy",
                "timestamp": datetime.now().isoformat()
            })

            return 1
        except Exception as e:
            self.logger.error(f"✗ Failed to copy {description}: {e}")
            return 0

    def _copy_and_format_csv(self, source: Path, dest: Path, description: str) -> int:
        """
        Copy and format a CSV file (clean column names, proper formatting).

        Args:
            source: Source CSV path
            dest: Destination CSV path
            description: Description for logging

        Returns:
            1 if successful, 0 otherwise
        """
        if not source.exists():
            self.logger.warning(f"Source file not found: {source}")
            return 0

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] Would format and copy {description}:")
            self.logger.info(f"  FROM: {source}")
            self.logger.info(f"  TO:   {dest}")
            return 1

        try:
            # Read CSV
            df = pd.read_csv(source)

            # Clean column names (remove extra spaces, standardize capitalization)
            df.columns = df.columns.str.strip()

            # Save formatted CSV
            df.to_csv(dest, index=False, float_format='%.4f')

            self.logger.info(f"✓ Formatted and copied {description}:")
            self.logger.info(f"  FROM: {source}")
            self.logger.info(f"  TO:   {dest}")
            self.logger.info(f"  ROWS: {len(df)}, COLUMNS: {len(df.columns)}")

            # Track exported file
            self.exported_files.append({
                "source": str(source),
                "destination": str(dest),
                "description": description,
                "type": "formatted_csv",
                "rows": len(df),
                "columns": len(df.columns),
                "timestamp": datetime.now().isoformat()
            })

            return 1
        except Exception as e:
            self.logger.error(f"✗ Failed to format {description}: {e}")
            return 0

    def _create_excel_version(self, csv_path: Path, xlsx_path: Path) -> int:
        """
        Create Excel version from CSV file.

        Args:
            csv_path: Source CSV path
            xlsx_path: Destination Excel path

        Returns:
            1 if successful, 0 otherwise
        """
        if not csv_path.exists():
            self.logger.warning(f"CSV file not found: {csv_path}")
            return 0

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] Would create Excel version:")
            self.logger.info(f"  FROM: {csv_path}")
            self.logger.info(f"  TO:   {xlsx_path}")
            return 1

        try:
            # Read CSV
            df = pd.read_csv(csv_path)

            # Save as Excel with auto-adjusted column widths
            with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Data')

                # Auto-adjust column widths
                worksheet = writer.sheets['Data']
                for idx, col in enumerate(df.columns, 1):
                    max_length = max(
                        df[col].astype(str).map(len).max(),
                        len(str(col))
                    )
                    adjusted_width = min(max_length + 2, 50)  # Cap at 50
                    worksheet.column_dimensions[chr(64 + idx)].width = adjusted_width

            self.logger.info(f"✓ Created Excel version: {xlsx_path.name}")

            # Track exported file
            self.exported_files.append({
                "source": str(csv_path),
                "destination": str(xlsx_path),
                "description": f"Excel version of {csv_path.name}",
                "type": "excel_conversion",
                "timestamp": datetime.now().isoformat()
            })

            return 1
        except Exception as e:
            self.logger.error(f"✗ Failed to create Excel version: {e}")
            return 0

    def generate_metadata(self) -> None:
        """Generate metadata.json tracking exported files."""
        metadata = {
            "experiment_directory": str(self.experiment_dir),
            "export_timestamp": datetime.now().isoformat(),
            "total_files_exported": len(self.exported_files),
            "exported_files": self.exported_files,
            "output_directories": {
                "detection": str(self.detection_dir),
                "classification": str(self.classification_dir),
                "statistics": str(self.statistics_dir),
            }
        }

        metadata_path = self.output_base / "export_metadata.json"

        if self.dry_run:
            self.logger.info(f"[DRY-RUN] Would save metadata to: {metadata_path}")
            self.logger.info(f"  Total files: {len(self.exported_files)}")
            return

        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            self.logger.info(f"✓ Saved metadata to: {metadata_path}")
        except Exception as e:
            self.logger.error(f"✗ Failed to save metadata: {e}")

    def run(self) -> Tuple[int, int, int]:
        """
        Run the complete export process.

        Returns:
            Tuple of (detection_count, classification_count, statistics_count)
        """
        self.logger.info("=" * 80)
        self.logger.info("TABLE EXPORT PROCESS STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        self.logger.info(f"Output directory: {self.output_base}")
        self.logger.info(f"Dry-run mode: {self.dry_run}")
        self.logger.info("")

        # Setup directories
        self.setup_directories()

        # Export tables
        detection_count = self.export_detection_tables()
        classification_count = self.export_classification_tables()
        statistics_count = self.export_statistics_tables()

        # Generate metadata
        self.logger.info("=" * 80)
        self.logger.info("GENERATING METADATA")
        self.logger.info("=" * 80)
        self.generate_metadata()

        # Summary
        total_count = detection_count + classification_count + statistics_count
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("EXPORT SUMMARY")
        self.logger.info("=" * 80)
        self.logger.info(f"Detection tables:      {detection_count}")
        self.logger.info(f"Classification tables: {classification_count}")
        self.logger.info(f"Statistics tables:     {statistics_count}")
        self.logger.info(f"TOTAL FILES EXPORTED:  {total_count}")
        self.logger.info("=" * 80)

        return detection_count, classification_count, statistics_count


def find_latest_experiment(results_dir: Path = Path("results")) -> Optional[Path]:
    """
    Find the latest optA_* experiment directory.

    Args:
        results_dir: Directory containing experiment results

    Returns:
        Path to latest experiment or None
    """
    if not results_dir.exists():
        logging.error(f"Results directory not found: {results_dir}")
        return None

    # Find all optA_* directories
    experiments = sorted(results_dir.glob("optA_*"), reverse=True)

    if not experiments:
        logging.error(f"No optA_* experiments found in {results_dir}")
        return None

    latest = experiments[0]
    logging.info(f"Auto-detected latest experiment: {latest.name}")
    return latest


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure logging format
    log_format = '%(message)s'

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export tables from pipeline results to luaran directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Auto-detect latest experiment
    python scripts/publication/export_tables_to_luaran.py

    # Specific experiment
    python scripts/publication/export_tables_to_luaran.py --experiment results/optA_20251012_143000

    # Dry-run (show what would be copied)
    python scripts/publication/export_tables_to_luaran.py --dry-run

    # Verbose output
    python scripts/publication/export_tables_to_luaran.py --verbose
        """
    )

    parser.add_argument(
        '--experiment',
        type=Path,
        help='Path to experiment directory (auto-detects latest if not specified)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('luaran/auto_generated/tables'),
        help='Output directory (default: luaran/auto_generated/tables)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be copied without actually copying'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    # Determine experiment directory
    if args.experiment:
        experiment_dir = args.experiment
        if not experiment_dir.exists():
            logger.error(f"Experiment directory not found: {experiment_dir}")
            return 1
    else:
        experiment_dir = find_latest_experiment()
        if not experiment_dir:
            logger.error("Could not find experiment directory")
            return 1

    # Verify experiment directory is valid
    if not (experiment_dir / "experiments").exists():
        logger.error(f"Invalid experiment directory (no 'experiments' folder): {experiment_dir}")
        logger.error("Expected structure: optA_[timestamp]/experiments/")
        return 1

    # Create exporter and run
    try:
        exporter = TableExporter(
            experiment_dir=experiment_dir,
            output_base=args.output,
            dry_run=args.dry_run
        )

        detection_count, classification_count, statistics_count = exporter.run()

        total_count = detection_count + classification_count + statistics_count

        if total_count == 0:
            logger.warning("No files were exported!")
            return 1

        if args.dry_run:
            logger.info("\n✓ Dry-run completed successfully")
            logger.info("  Run without --dry-run to actually export files")
        else:
            logger.info(f"\n✓ Successfully exported {total_count} files to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
