import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Professional journal style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 11,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'figure.dpi': 400,
    'savefig.dpi': 400,
    'savefig.bbox': 'tight'
})

# Configuration
EXPERIMENT_DIR = Path("results/optA_20251016_200330/experiments")
OUTPUT_DIR = Path("luaran/auto_generated/figures/training_curves")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Professional color scheme (color-blind friendly)
COLORS = {
    'best_train': '#0173B2',      # Blue
    'best_val': '#DE8F05',        # Orange
    'worst_train': '#029E73',     # Teal/Green
    'worst_val': '#CC78BC',       # Purple
}

# Dataset configurations
DATASETS = {
    'iml_lifecycle': {
        'name': 'IML Lifecycle',
        'best': ('efficientnet_b1', 'EfficientNet-B1'),
        'worst': ('resnet101', 'ResNet101')
    },
    'mp_idb_species': {
        'name': 'MP-IDB Species',
        'best': ('efficientnet_b1', 'EfficientNet-B1'),
        'worst': ('efficientnet_b0', 'EfficientNet-B0')
    },
    'mp_idb_stages': {
        'name': 'MP-IDB Stages',
        'best': ('resnet50', 'ResNet50'),
        'worst': ('densenet121', 'DenseNet121')
    },
    'md_2019_stages': {
        'name': 'MD_2019 Stages',
        'best': ('efficientnet_b0', 'EfficientNet-B0'),
        'worst': ('resnet101', 'ResNet101')
    }
}

def load_training_data(dataset_key, model_name):
    """Load training history CSV"""
    csv_path = EXPERIMENT_DIR / f"experiment_{dataset_key}" / f"cls_{model_name}_focal" / "results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found!")
        return None
    return pd.read_csv(csv_path)

def create_accuracy_figure(dataset_key, config):
    """Create accuracy comparison figure (best vs worst)"""
    print(f"Creating accuracy figure for {dataset_key}...")

    # Load data
    best_data = load_training_data(dataset_key, config['best'][0])
    worst_data = load_training_data(dataset_key, config['worst'][0])

    if best_data is None or worst_data is None:
        print(f"Skipping {dataset_key} accuracy - missing data")
        return False

    # Create square figure (4x4) with no padding
    fig, ax = plt.subplots(figsize=(4, 4))

    # Plot best model
    ax.plot(best_data['epoch'], best_data['train_acc'],
            color=COLORS['best_train'], linewidth=2.5, linestyle='-',
            label=f'{config["best"][1]} (Train)', alpha=0.9)
    ax.plot(best_data['epoch'], best_data['val_acc'],
            color=COLORS['best_val'], linewidth=2.5, linestyle='--',
            label=f'{config["best"][1]} (Val)', alpha=0.9)

    # Plot worst model
    ax.plot(worst_data['epoch'], worst_data['train_acc'],
            color=COLORS['worst_train'], linewidth=2.5, linestyle='-',
            label=f'{config["worst"][1]} (Train)', alpha=0.9)
    ax.plot(worst_data['epoch'], worst_data['val_acc'],
            color=COLORS['worst_val'], linewidth=2.5, linestyle='--',
            label=f'{config["worst"][1]} (Val)', alpha=0.9)

    # Styling (no axis labels for space saving, zoomed Y-axis)
    ax.set_ylim([20, 100])
    ax.set_xlim([0, max(best_data['epoch'].max(), worst_data['epoch'].max())])
    ax.margins(0)  # Remove margins
    ax.grid(True, alpha=0.25, linewidth=0.8, linestyle=':')
    ax.legend(loc='lower right', frameon=True, framealpha=0.95,
              edgecolor='black', fontsize=10, ncol=1)

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, length=6)

    # Remove padding completely - aggressive margins
    plt.subplots_adjust(left=0.08, right=0.995, top=0.995, bottom=0.06)

    # Save with minimal padding
    output_path = OUTPUT_DIR / f"accuracy_{dataset_key}.png"
    plt.savefig(output_path, dpi=400, bbox_inches='tight', pad_inches=0.02, facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()
    return True

def create_loss_figure(dataset_key, config):
    """Create loss comparison figure (best vs worst)"""
    print(f"Creating loss figure for {dataset_key}...")

    # Load data
    best_data = load_training_data(dataset_key, config['best'][0])
    worst_data = load_training_data(dataset_key, config['worst'][0])

    if best_data is None or worst_data is None:
        print(f"Skipping {dataset_key} loss - missing data")
        return False

    # Create figure (5 width, 4.3 height - ratio 1.16:1)
    fig, ax = plt.subplots(figsize=(5, 4.3))

    # Plot best model
    ax.plot(best_data['epoch'], best_data['train_loss'],
            color=COLORS['best_train'], linewidth=2.5, linestyle='-',
            label=f'{config["best"][1]} (Train)', alpha=0.9)
    ax.plot(best_data['epoch'], best_data['val_loss'],
            color=COLORS['best_val'], linewidth=2.5, linestyle='--',
            label=f'{config["best"][1]} (Val)', alpha=0.9)

    # Plot worst model
    ax.plot(worst_data['epoch'], worst_data['train_loss'],
            color=COLORS['worst_train'], linewidth=2.5, linestyle='-',
            label=f'{config["worst"][1]} (Train)', alpha=0.9)
    ax.plot(worst_data['epoch'], worst_data['val_loss'],
            color=COLORS['worst_val'], linewidth=2.5, linestyle='--',
            label=f'{config["worst"][1]} (Val)', alpha=0.9)

    # Styling
    ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
    ax.set_ylim(bottom=0)

    # Set reasonable y-axis upper limit based on data
    max_loss = max(best_data['val_loss'].max(), worst_data['val_loss'].max())
    ax.set_ylim([0, min(max_loss * 1.1, 2.0)])  # Cap at 2.0 for readability

    ax.grid(True, alpha=0.25, linewidth=0.8, linestyle=':')
    ax.legend(loc='upper right', frameon=True, framealpha=0.95,
              edgecolor='black', fontsize=10, ncol=1)

    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=11, length=6)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / f"loss_{dataset_key}.png"
    plt.savefig(output_path, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")
    plt.close()
    return True

def main():
    """Generate all professional training curve figures"""

    print("=" * 80)
    print("GENERATING PROFESSIONAL TRAINING CURVES (Q1/Q2 Journal Quality)")
    print("=" * 80)
    print()
    print("Style: Color-blind friendly, minimal design, no axis labels")
    print("Output: 4 accuracy figures (no loss figures - space saving)")
    print()

    success_count = 0

    for dataset_key, config in DATASETS.items():
        print(f"\n[{config['name']}]")
        print(f"  Best:  {config['best'][1]}")
        print(f"  Worst: {config['worst'][1]}")

        # Generate accuracy figure only (no loss for space saving)
        if create_accuracy_figure(dataset_key, config):
            success_count += 1

    print("\n" + "=" * 80)
    print(f"✓ COMPLETE: {success_count}/4 figures generated successfully")
    print(f"✓ Location: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == '__main__':
    main()
