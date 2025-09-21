#!/usr/bin/env python3
"""
Pipeline Manager: Utility for managing pipeline continuation
Provides commands to list, inspect, and continue experiments
"""

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.pipeline_continue import (
    list_available_experiments,
    print_experiment_status,
    get_available_experiments,
    check_completed_stages,
    determine_next_stage
)

def main():
    parser = argparse.ArgumentParser(description="Pipeline Manager: Manage and continue experiments")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List command
    list_parser = subparsers.add_parser('list', help='List available experiments')
    list_parser.add_argument('--all', action='store_true', help='Show all experiments (not just latest 10)')

    # Status command
    status_parser = subparsers.add_parser('status', help='Show detailed status of an experiment')
    status_parser.add_argument('experiment', help='Experiment name (e.g., exp_multi_pipeline_20250921_144544)')

    # Clean command
    clean_parser = subparsers.add_parser('clean', help='Clean up incomplete experiments')
    clean_parser.add_argument('--dry-run', action='store_true', help='Show what would be cleaned without actually doing it')

    args = parser.parse_args()

    if args.command == 'list':
        handle_list_command(args)
    elif args.command == 'status':
        handle_status_command(args)
    elif args.command == 'clean':
        handle_clean_command(args)
    else:
        parser.print_help()

def handle_list_command(args):
    """Handle the list command"""
    experiments = get_available_experiments()

    if not experiments:
        print("No experiments found in results/ directory")
        return

    print("📁 Available Experiments for Continuation:")
    print("=" * 60)

    limit = None if args.all else 10
    shown_experiments = experiments[:limit] if limit else experiments

    for exp in shown_experiments:
        exp_path = Path("results") / exp
        stages = check_completed_stages(str(exp_path))
        completed_count = sum(stages.values())
        next_stage = determine_next_stage(stages)

        # Color coding based on completion
        if completed_count == 0:
            status_icon = "🔴"  # No progress
        elif completed_count == 4:
            status_icon = "✅"  # Complete
        else:
            status_icon = "🟡"  # In progress

        print(f"  {status_icon} {exp}")
        print(f"     Progress: {completed_count}/4 stages")
        print(f"     Next: {next_stage}")

        # Show completed stages
        completed_stages = [stage for stage, done in stages.items() if done]
        if completed_stages:
            print(f"     Completed: {', '.join(completed_stages)}")

        print()

    if limit and len(experiments) > limit:
        print(f"... and {len(experiments) - limit} more experiments")
        print("Use --all to show all experiments")

    print("\nTo continue an experiment:")
    print("python run_multiple_models_pipeline.py --continue-from EXPERIMENT_NAME")
    print("\nTo see detailed status:")
    print("python pipeline_manager.py status EXPERIMENT_NAME")

def handle_status_command(args):
    """Handle the status command"""
    experiment_name = args.experiment

    # Handle both with and without "results/" prefix
    if experiment_name.startswith("results/"):
        experiment_path = experiment_name
    else:
        experiment_path = f"results/{experiment_name}"

    if not Path(experiment_path).exists():
        print(f"❌ Experiment not found: {experiment_path}")

        # Suggest similar experiments
        available = get_available_experiments()
        similar = [exp for exp in available if experiment_name.lower() in exp.lower()]

        if similar:
            print("\nDid you mean one of these?")
            for exp in similar[:5]:
                print(f"  - {exp}")
        else:
            print("\nUse 'python pipeline_manager.py list' to see available experiments")
        return

    print_experiment_status(experiment_path)

    # Show continue command
    stages = check_completed_stages(experiment_path)
    next_stage = determine_next_stage(stages)

    if next_stage != 'analysis' or not all(stages.values()):
        print("💡 To continue this experiment:")
        if next_stage == 'analysis':
            print(f"python run_multiple_models_pipeline.py --continue-from {experiment_name}")
        else:
            print(f"python run_multiple_models_pipeline.py --continue-from {experiment_name}")
            print(f"                                         # Will auto-start from {next_stage}")
            print(f"# Or force specific stage:")
            print(f"python run_multiple_models_pipeline.py --continue-from {experiment_name} --start-stage {next_stage}")

def handle_clean_command(args):
    """Handle the clean command"""
    experiments = get_available_experiments()

    if not experiments:
        print("No experiments found")
        return

    # Find experiments with no progress (no completed stages)
    empty_experiments = []
    incomplete_experiments = []

    for exp in experiments:
        exp_path = Path("results") / exp
        stages = check_completed_stages(str(exp_path))
        completed_count = sum(stages.values())

        if completed_count == 0:
            empty_experiments.append(exp)
        elif completed_count < 4 and completed_count > 0:
            incomplete_experiments.append(exp)

    if not empty_experiments and not incomplete_experiments:
        print("✅ No experiments need cleaning")
        return

    print("🧹 Cleanup Analysis:")
    print("=" * 50)

    if empty_experiments:
        print(f"📁 Empty experiments (no progress): {len(empty_experiments)}")
        for exp in empty_experiments[:5]:  # Show first 5
            print(f"  - {exp}")
        if len(empty_experiments) > 5:
            print(f"  ... and {len(empty_experiments) - 5} more")

    if incomplete_experiments:
        print(f"📁 Incomplete experiments (partial progress): {len(incomplete_experiments)}")
        for exp in incomplete_experiments[:5]:  # Show first 5
            exp_path = Path("results") / exp
            stages = check_completed_stages(str(exp_path))
            completed_count = sum(stages.values())
            print(f"  - {exp} ({completed_count}/4 stages)")
        if len(incomplete_experiments) > 5:
            print(f"  ... and {len(incomplete_experiments) - 5} more")

    if args.dry_run:
        print("\n🔍 This is a dry run. No files were deleted.")
        print("Remove --dry-run to actually clean up empty experiments.")
    else:
        print("\n⚠️  Cleanup not yet implemented for safety.")
        print("Manual cleanup recommended:")
        if empty_experiments:
            print("For empty experiments:")
            for exp in empty_experiments:
                print(f"  rm -rf results/{exp}")

if __name__ == "__main__":
    main()