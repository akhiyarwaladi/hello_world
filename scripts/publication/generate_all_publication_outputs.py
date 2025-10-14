#!/usr/bin/env python3
"""
Generate ALL Publication Outputs - Master Orchestrator

This script generates everything needed for publication submission from a pipeline experiment.

Usage:
    # Auto-detect latest experiment
    python scripts/publication/generate_all_publication_outputs.py

    # Specific experiment
    python scripts/publication/generate_all_publication_outputs.py --experiment results/optA_20251012_143000

    # Dry-run (show what would be generated)
    python scripts/publication/generate_all_publication_outputs.py --dry-run

    # Skip verification (faster, but not recommended)
    python scripts/publication/generate_all_publication_outputs.py --skip-verify

Outputs:
    luaran/auto_generated/
    ├── figures/ (32 files: diagrams, augmentation, performance)
    ├── tables/ (17 files: detection, classification, statistics)
    └── _metadata.json (tracking info for all files)
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple


class PublicationOutputGenerator:
    """Master orchestrator for generating all publication outputs."""

    def __init__(self, experiment_dir: Path, dry_run: bool = False, skip_verify: bool = False):
        self.exp_dir = experiment_dir
        self.dry_run = dry_run
        self.skip_verify = skip_verify
        self.metadata: Dict = {
            "generated_at": datetime.now().isoformat(),
            "source_experiment": str(self.exp_dir),
            "generator_version": "1.0.0",
            "files": {}
        }
        self.step_results: List[Tuple[str, bool, str]] = []

    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run command with logging and error handling."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"\n[{timestamp}] {description}")

        if self.dry_run:
            print(f"  DRY-RUN: Would run: {' '.join(cmd)}")
            return True

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                print(f"  ❌ ERROR: {error_msg}")
                self.step_results.append((description, False, error_msg))
                return False

            # Show output if verbose
            if result.stdout and result.stdout.strip():
                for line in result.stdout.strip().split('\n')[-5:]:  # Last 5 lines
                    print(f"    {line}")

            print(f"  ✅ SUCCESS")
            self.step_results.append((description, True, ""))
            return True

        except Exception as e:
            error_msg = str(e)
            print(f"  ❌ EXCEPTION: {error_msg}")
            self.step_results.append((description, False, error_msg))
            return False

    def step1_export_tables(self) -> bool:
        """Export tables from experiment to luaran/auto_generated/tables/."""
        script_path = Path("scripts/publication/export_tables_to_luaran.py")

        if not script_path.exists():
            print(f"  ⚠️ WARNING: {script_path} not found, skipping table export")
            self.step_results.append(("Export Tables", False, "Script not found"))
            return False

        return self.run_command(
            [sys.executable, str(script_path), "--experiment", str(self.exp_dir)],
            "📊 Exporting tables to luaran/auto_generated/tables/"
        )

    def step2_generate_performance_figures(self) -> bool:
        """Generate performance analysis figures."""
        # Check if figures already exist in luaran/auto_generated/figures/performance/
        figures_dir = Path("luaran/auto_generated/figures/performance")

        if figures_dir.exists():
            figure_count = len(list(figures_dir.glob("*.png")))
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📈 Performance figures")
            print(f"  ℹ️ Found {figure_count} existing figures in {figures_dir}")
            print(f"  ✅ Using existing performance figures")
            self.step_results.append(("Performance Figures", True, f"{figure_count} files found"))
            return True

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📈 Performance figures")
        print(f"  ⚠️ No figures found in {figures_dir}")
        self.step_results.append(("Performance Figures", False, "No figures directory"))
        return False

    def step3_generate_pipeline_diagrams(self) -> bool:
        """Generate pipeline architecture diagrams."""
        script_path = Path("scripts/visualization/generate_pipeline_architecture_diagram.py")

        if not script_path.exists():
            print(f"  ⚠️ WARNING: {script_path} not found")
            self.step_results.append(("Pipeline Diagrams", False, "Script not found"))
            return False

        return self.run_command(
            [sys.executable, str(script_path)],
            "🔄 Generating pipeline architecture diagrams"
        )

    def step4_generate_augmentation_figures(self) -> bool:
        """Generate augmentation visualization figures."""
        # Check if augmentation figures already exist in auto_generated/
        aug_dir = Path("luaran/auto_generated/figures/augmentation")

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🎨 Augmentation figures")

        if aug_dir.exists():
            aug_figures = list(aug_dir.glob("augmentation_*.png"))
            if aug_figures:
                print(f"  ℹ️ Found {len(aug_figures)} existing augmentation figures")
                print(f"  ✅ Using existing augmentation figures")
                self.step_results.append(("Augmentation Figures", True, f"{len(aug_figures)} files found"))
                return True

        print(f"  ⚠️ No augmentation figures found in {aug_dir}")
        print(f"  ℹ️ To generate, run: scripts/visualization/generate_compact_augmentation_figures.py")
        self.step_results.append(("Augmentation Figures", False, "No figures found"))
        return False

    def step5_create_metadata(self) -> bool:
        """Create metadata.json tracking all outputs."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📝 Creating metadata")

        auto_gen_dir = Path("luaran/auto_generated")
        metadata_path = auto_gen_dir / "_metadata.json"

        if not auto_gen_dir.exists():
            print(f"  ⚠️ {auto_gen_dir} does not exist")
            self.step_results.append(("Create Metadata", False, "auto_generated directory missing"))
            return False

        # Scan auto_generated/ for all files
        file_count = 0
        for file_path in auto_gen_dir.rglob("*"):
            if file_path.is_file() and file_path.name not in ["_metadata.json", "_generation_summary.md"]:
                rel_path = file_path.relative_to(auto_gen_dir)
                self.metadata["files"][str(rel_path)] = {
                    "last_updated": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    "size_bytes": file_path.stat().st_size,
                    "category": self._categorize_file(file_path)
                }
                file_count += 1

        # Add summary statistics
        self.metadata["summary"] = {
            "total_files": file_count,
            "tables_count": len([f for f in self.metadata["files"] if "tables/" in f]),
            "figures_count": len([f for f in self.metadata["files"] if "figures/" in f]),
            "total_size_mb": sum(f["size_bytes"] for f in self.metadata["files"].values()) / (1024 * 1024)
        }

        if not self.dry_run:
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)

        print(f"  ✅ Created metadata ({file_count} files tracked)")
        print(f"     - Tables: {self.metadata['summary']['tables_count']}")
        print(f"     - Figures: {self.metadata['summary']['figures_count']}")
        print(f"     - Total size: {self.metadata['summary']['total_size_mb']:.2f} MB")
        self.step_results.append(("Create Metadata", True, f"{file_count} files"))
        return True

    def _categorize_file(self, file_path: Path) -> str:
        """Categorize file based on path and extension."""
        path_str = str(file_path)

        if "tables/" in path_str:
            return "table"
        elif "figures/" in path_str:
            if file_path.suffix == ".png":
                return "figure"
            elif file_path.suffix == ".pdf":
                return "vector_figure"

        return "other"

    def step6_verify_data(self) -> bool:
        """Verify data integrity."""
        if self.skip_verify:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 🔍 Data verification")
            print(f"  ⏭️ Skipping verification (--skip-verify)")
            self.step_results.append(("Verify Data", True, "Skipped by user"))
            return True

        script_path = Path("scripts/publication/verify_publication_data.py")

        if not script_path.exists():
            print(f"  ⚠️ WARNING: {script_path} not found")
            self.step_results.append(("Verify Data", False, "Script not found"))
            return False

        return self.run_command(
            [sys.executable, str(script_path), "--experiment", str(self.exp_dir)],
            "🔍 Verifying data integrity"
        )

    def step7_generate_summary(self) -> bool:
        """Generate summary report."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 📋 Generating summary report")

        summary_path = Path("luaran/auto_generated/_generation_summary.md")

        # Count files
        auto_gen = Path("luaran/auto_generated")
        table_files = list(auto_gen.glob("tables/*.csv")) if (auto_gen / "tables").exists() else []
        figure_files = list(auto_gen.glob("figures/*.png")) if (auto_gen / "figures").exists() else []

        # Build step results table
        step_table = "| Step | Status | Details |\n|------|--------|----------|\n"
        for step_name, success, details in self.step_results:
            status = "✅ SUCCESS" if success else "❌ FAILED"
            details_str = details if details else "-"
            step_table += f"| {step_name} | {status} | {details_str} |\n"

        summary = f"""# Publication Outputs Generation Summary

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Source Experiment**: `{self.exp_dir}`
**Mode**: {'🔍 DRY-RUN' if self.dry_run else '✅ PRODUCTION'}

## Generation Steps

{step_table}

## Files Generated

- **Tables**: {len(table_files)} CSV files
- **Figures**: {len(figure_files)} PNG files
- **Metadata**: `_metadata.json`

### Tables Directory
```
luaran/auto_generated/tables/
├── Dataset statistics (3 files)
├── Detection performance (3 files)
├── Classification performance (6 files)
└── Consolidated analysis (5 files)
```

### Figures Directory
```
luaran/auto_generated/figures/
├── Pipeline diagrams (2-4 files)
├── Augmentation visualizations (3 files)
└── Performance charts (varies)
```

## Next Steps

1. **Review generated files** in `luaran/auto_generated/`
2. **Check verification report** (if not skipped)
3. **Copy tables to papers** - Excel format available for easy import
4. **Use figures in submissions** - High-resolution PNG files ready

## File Locations

- **Auto-generated outputs**: `luaran/auto_generated/`
- **Manual figures**: `luaran/figures/`
- **Papers**: `luaran/papers/`
- **Reports**: `luaran/Laporan_Kemajuan.md`

## Regeneration

To regenerate all outputs:
```bash
# Auto-detect latest experiment
python scripts/publication/generate_all_publication_outputs.py

# Specific experiment
python scripts/publication/generate_all_publication_outputs.py --experiment {self.exp_dir}

# Dry-run (preview without changes)
python scripts/publication/generate_all_publication_outputs.py --dry-run
```

## Troubleshooting

- **Missing tables**: Ensure experiment has `consolidated_analysis/` folder
- **Missing figures**: Check if `luaran/figures/` exists with expected files
- **Verification failed**: Review experiment structure and data files

---
*Generated by `scripts/publication/generate_all_publication_outputs.py` v{self.metadata['generator_version']}*
"""

        if not self.dry_run:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"  ✅ Summary saved to {summary_path}")
        else:
            print(f"  DRY-RUN: Would save to {summary_path}")

        self.step_results.append(("Generate Summary", True, ""))
        return True

    def run_all(self) -> bool:
        """Execute all steps in sequence."""
        print(f"\n{'='*70}")
        print(f"  PUBLICATION OUTPUTS GENERATION")
        print(f"{'='*70}")
        print(f"  Experiment: {self.exp_dir}")
        print(f"  Mode: {'DRY-RUN (no changes)' if self.dry_run else 'PRODUCTION (live execution)'}")
        print(f"  Verification: {'Skipped' if self.skip_verify else 'Enabled'}")
        print(f"{'='*70}")

        steps = [
            ("Export Tables", self.step1_export_tables),
            ("Performance Figures", self.step2_generate_performance_figures),
            ("Pipeline Diagrams", self.step3_generate_pipeline_diagrams),
            ("Augmentation Figures", self.step4_generate_augmentation_figures),
            ("Create Metadata", self.step5_create_metadata),
            ("Verify Data", self.step6_verify_data),
            ("Generate Summary", self.step7_generate_summary),
        ]

        all_success = True
        for step_name, step_func in steps:
            success = step_func()
            if not success:
                print(f"\n⚠️ WARNING: Step '{step_name}' had issues, continuing...")
                all_success = False

        # Print final summary
        print(f"\n{'='*70}")
        success_count = sum(1 for _, success, _ in self.step_results if success)
        total_count = len(self.step_results)

        if all_success:
            print(f"  ✅ ALL STEPS COMPLETED SUCCESSFULLY!")
            print(f"  {success_count}/{total_count} steps succeeded")
        else:
            print(f"  ⚠️ COMPLETED WITH WARNINGS")
            print(f"  {success_count}/{total_count} steps succeeded")
            print(f"\n  Failed steps:")
            for step_name, success, details in self.step_results:
                if not success:
                    print(f"    - {step_name}: {details}")

        print(f"{'='*70}")
        print(f"\n📁 Output location: luaran/auto_generated/")
        print(f"📋 Summary report: luaran/auto_generated/_generation_summary.md")

        return all_success


def find_latest_experiment() -> Optional[Path]:
    """Find latest results/optA_* folder."""
    results_dir = Path("results")

    if not results_dir.exists():
        return None

    experiments = list(results_dir.glob("optA_*"))
    if not experiments:
        return None

    # Sort by timestamp in folder name (reverse for latest first)
    experiments.sort(reverse=True)
    return experiments[0]


def validate_experiment_dir(exp_dir: Path) -> bool:
    """Validate experiment directory structure."""
    if not exp_dir.exists():
        print(f"❌ ERROR: Experiment directory not found: {exp_dir}")
        return False

    # Check for experiments/ subdirectory (new structure)
    experiments_subdir = exp_dir / "experiments"
    if not experiments_subdir.exists():
        print(f"⚠️ WARNING: {experiments_subdir} not found")
        print(f"   This might be an old experiment structure")

    # Check for consolidated_analysis/ (required for table export)
    consolidated = exp_dir / "consolidated_analysis"
    if not consolidated.exists():
        print(f"⚠️ WARNING: {consolidated} not found")
        print(f"   Table export may be limited")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate all publication outputs from pipeline experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect latest experiment
  python scripts/publication/generate_all_publication_outputs.py

  # Specific experiment
  python scripts/publication/generate_all_publication_outputs.py --experiment results/optA_20251012_143000

  # Dry-run (preview only)
  python scripts/publication/generate_all_publication_outputs.py --dry-run

  # Skip verification (faster)
  python scripts/publication/generate_all_publication_outputs.py --skip-verify
        """
    )

    parser.add_argument(
        '--experiment',
        type=str,
        help='Experiment directory (auto-detect latest if not specified)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip data integrity verification'
    )

    args = parser.parse_args()

    # Find experiment directory
    if args.experiment:
        exp_dir = Path(args.experiment)
        print(f"📂 Using specified experiment: {exp_dir}")
    else:
        exp_dir = find_latest_experiment()
        if not exp_dir:
            print("❌ ERROR: No experiment found in results/ directory")
            print("   Specify with: --experiment results/optA_YYYYMMDD_HHMMSS")
            sys.exit(1)
        print(f"📂 Auto-detected latest experiment: {exp_dir}")

    # Validate experiment directory
    if not validate_experiment_dir(exp_dir):
        sys.exit(1)

    # Run generator
    generator = PublicationOutputGenerator(exp_dir, args.dry_run, args.skip_verify)
    success = generator.run_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
