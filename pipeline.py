#!/usr/bin/env python3
"""Unified CLI for the malaria detection + classification pipeline."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

CONFIG_PATH = Path("config/models.yaml")


def load_models_config(config_path: Path = CONFIG_PATH) -> Dict[str, Dict]:
    """Load model configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Model configuration not found at {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    models_cfg = config.get("models", {})
    if not models_cfg:
        raise ValueError("No 'models' section found in configuration file")
    return models_cfg


def build_parser(model_choices: List[str]) -> argparse.ArgumentParser:
    """Create CLI parser with subcommands that leverage model config."""
    parser = argparse.ArgumentParser(description="Malaria Detection Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available models")
    list_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show the underlying training script and default arguments",
    )

    train_parser = subparsers.add_parser("train", help="Train a configured model")
    train_parser.add_argument("model", choices=model_choices, help="Model key from config")
    train_parser.add_argument("--name", required=True, help="Experiment name")
    train_parser.add_argument("--data", help="Override dataset path")
    train_parser.add_argument("--epochs", type=int, help="Override training epochs")
    train_parser.add_argument("--batch", type=int, help="Override batch size")
    train_parser.add_argument("--device", help="Override training device")
    train_parser.add_argument("--imgsz", type=int, help="Override image size")
    train_parser.add_argument("--model-weights", dest="model_weights", help="Override pretrained weights")
    train_parser.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Set additional script-specific arguments (repeatable)",
    )
    train_parser.add_argument(
        "--background", action="store_true", help="Run the training command in background"
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the constructed command without executing it",
    )

    return parser


def parse_set_overrides(values: List[str]) -> Dict[str, str]:
    """Parse repeated KEY=VALUE overrides from CLI."""
    parsed: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected format KEY=VALUE")
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def normalize_value(value):
    """Convert values to string while keeping booleans consistent with CLI expectations."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_command(model_cfg: Dict, overrides: Dict[str, object]) -> Tuple[List[str], bool]:
    """Construct the command to execute along with metadata about its style."""

    base_cmd = shlex.split(model_cfg["script"])
    command_style_python = base_cmd and Path(base_cmd[0]).name.startswith("python")

    args = dict(model_cfg.get("args", {}))
    for key, value in overrides.items():
        if value is not None:
            args[key] = value

    command = base_cmd.copy()

    for key, value in args.items():
        if isinstance(value, bool) and not value and command_style_python:
            # False booleans do not need to be forwarded for python-style commands
            continue

        if command_style_python:
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                command.append(flag)
            else:
                command.extend([flag, normalize_value(value)])
        else:
            command.append(f"{key}={normalize_value(value)}")

    return command, command_style_python


def list_models(models_cfg: Dict[str, Dict], detailed: bool = False) -> None:
    """Display available models from configuration."""
    print("Available models:\n")
    for name, cfg in models_cfg.items():
        description = cfg.get("description", "")
        print(f"- {name}: {description}")
        if detailed:
            print(f"  Script : {cfg.get('script')}")
            default_args = cfg.get("args", {})
            if default_args:
                arg_string = ", ".join(f"{k}={v}" for k, v in default_args.items())
                print(f"  Defaults: {arg_string}")
        print()


def run_command(command: List[str], background: bool = False) -> int:
    """Execute the final command, optionally in background."""
    if background:
        process = subprocess.Popen(command)
        print(f"Started background process PID={process.pid}")
        return 0

    completed = subprocess.run(command)
    return completed.returncode


def main() -> int:
    try:
        models_cfg = load_models_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 1

    parser = build_parser(list(models_cfg.keys()))

    try:
        args = parser.parse_args()
    except ValueError as exc:
        parser.error(str(exc))

    if args.command == "list":
        list_models(models_cfg, detailed=args.detailed)
        return 0

    model_cfg = models_cfg[args.model]

    overrides = {
        "name": args.name,
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "device": args.device,
        "imgsz": args.imgsz,
        "model": args.model_weights,
    }

    try:
        overrides.update(parse_set_overrides(args.set))
    except ValueError as exc:
        parser.error(str(exc))

    command, is_python = build_command(model_cfg, overrides)
    display_cmd = " ".join(shlex.quote(part) for part in command)

    print("=" * 80)
    print(f"Executing {args.model} using {'python script' if is_python else 'YOLO CLI'}")
    print(display_cmd)
    print("=" * 80)

    if args.dry_run:
        print("Dry-run requested; command not executed.")
        return 0

    returncode = run_command(command, background=args.background)
    if returncode != 0:
        print(f"❌ Command exited with status {returncode}")
    return returncode


if __name__ == "__main__":
    sys.exit(main())
