#!/usr/bin/env python3
"""
Consolidate All Malaria Detection Results
Mengumpulkan semua hasil yang terpencar dan membuat ringkasan lengkap
"""

import os
import sys
import json
import shutil
from pathlib import Path
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class ResultsConsolidator:
    """Mengkonsolidasi semua hasil eksperimen malaria detection"""

    def __init__(self, base_dir: str = "/home/gli/Documents/malaria_detection"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.consolidated_dir = self.base_dir / "consolidated_results"
        self.consolidated_dir.mkdir(exist_ok=True)

        # Struktur hasil yang ditemukan
        self.found_results = {
            'detection_models': [],
            'classification_models': [],
            'combination_models': [],
            'completed_models': []
        }

    def scan_all_results(self):
        """Scan semua folder hasil"""
        print("🔍 Scanning semua hasil eksperimen...")

        # Scan berbagai direktori hasil
        search_dirs = [
            self.results_dir / "current_experiments",
            self.results_dir / "completed_models",
            self.results_dir / "validation",
            self.results_dir,
            self.base_dir / "runs"  # Jika ada
        ]

        for search_dir in search_dirs:
            if search_dir.exists():
                self._scan_directory(search_dir)

        # Scan hasil PyTorch classification yang tersimpan terpisah
        pytorch_results_path = self.results_dir / "current_experiments" / "training" / "classification" / "pytorch_classification"
        if pytorch_results_path.exists():
            for model_dir in pytorch_results_path.iterdir():
                if model_dir.is_dir():
                    self._analyze_pytorch_model(model_dir)

    def _scan_directory(self, directory: Path, level: int = 0):
        """Recursively scan directory untuk model weights dan results"""
        if level > 4:  # Prevent infinite recursion
            return

        try:
            for item in directory.iterdir():
                if item.is_dir():
                    # Check if this is a model directory
                    if (item / "weights").exists() or (item / "best.pt").exists():
                        self._analyze_model_directory(item)
                    # Check if has results.txt
                    elif (item / "results.txt").exists():
                        self._analyze_results_file(item)
                    else:
                        # Continue recursing
                        self._scan_directory(item, level + 1)
        except PermissionError:
            pass

    def _analyze_model_directory(self, model_dir: Path):
        """Analyze individual model directory"""
        try:
            model_info = {
                'name': model_dir.name,
                'path': str(model_dir),
                'type': self._classify_model_type(model_dir.name),
                'has_weights': False,
                'has_results': False,
                'results': {}
            }

            # Check for weights
            weights_dir = model_dir / "weights"
            if weights_dir.exists():
                model_info['has_weights'] = True
                weights_files = list(weights_dir.glob("*.pt"))
                model_info['weights_files'] = [str(w) for w in weights_files]

            # Check for results
            results_file = model_dir / "results.txt"
            if results_file.exists():
                model_info['has_results'] = True
                model_info['results'] = self._parse_results_file(results_file)

            # Classify and store
            model_type = model_info['type']
            self.found_results[f'{model_type}_models'].append(model_info)

        except Exception as e:
            print(f"⚠️ Error analyzing {model_dir}: {e}")

    def _analyze_pytorch_model(self, model_dir: Path):
        """Analyze PyTorch model results"""
        try:
            model_info = {
                'name': model_dir.name,
                'path': str(model_dir),
                'type': 'combination',  # PyTorch models are usually combination
                'framework': 'pytorch',
                'has_weights': False,
                'has_results': False,
                'results': {}
            }

            # Check for PyTorch weights (.pt files)
            pt_files = list(model_dir.glob("*.pt"))
            if pt_files:
                model_info['has_weights'] = True
                model_info['weights_files'] = [str(w) for w in pt_files]

            # Check for results
            results_file = model_dir / "results.txt"
            if results_file.exists():
                model_info['has_results'] = True
                model_info['results'] = self._parse_results_file(results_file)

            self.found_results['combination_models'].append(model_info)

        except Exception as e:
            print(f"⚠️ Error analyzing PyTorch model {model_dir}: {e}")

    def _analyze_results_file(self, results_dir: Path):
        """Analyze directory with results.txt"""
        results_file = results_dir / "results.txt"
        try:
            results = self._parse_results_file(results_file)
            model_info = {
                'name': results_dir.name,
                'path': str(results_dir),
                'type': self._classify_model_type(results_dir.name),
                'has_weights': False,
                'has_results': True,
                'results': results
            }

            model_type = model_info['type']
            self.found_results[f'{model_type}_models'].append(model_info)

        except Exception as e:
            print(f"⚠️ Error analyzing results in {results_dir}: {e}")

    def _classify_model_type(self, name: str) -> str:
        """Classify model type based on name"""
        name_lower = name.lower()

        if any(word in name_lower for word in ['det_to', 'combo', 'species_aware', 'ground_truth_to']):
            return 'combination'
        elif any(word in name_lower for word in ['detection', 'detect', 'yolo8', 'yolo11', 'rtdetr']):
            return 'detection'
        elif any(word in name_lower for word in ['classification', 'classify', 'cls', 'resnet', 'efficientnet', 'densenet', 'mobilenet']):
            return 'classification'
        else:
            return 'completed'  # Default category

    def _parse_results_file(self, results_file: Path) -> dict:
        """Parse results.txt file"""
        results = {}
        try:
            with open(results_file, 'r') as f:
                content = f.read()

            # Parse different metrics
            lines = content.split('\n')
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()

                        # Try to extract numeric values
                        if any(metric in key.lower() for metric in ['acc', 'accuracy', 'loss', 'precision', 'recall', 'f1']):
                            try:
                                # Extract number (handling percentages)
                                import re
                                numbers = re.findall(r'-?\d+\.?\d*', value)
                                if numbers:
                                    results[key] = float(numbers[0])
                            except:
                                results[key] = value
                        else:
                            results[key] = value

        except Exception as e:
            print(f"⚠️ Error parsing {results_file}: {e}")

        return results

    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("📊 Generating comprehensive summary report...")

        # Create summary data
        summary_data = {
            'scan_date': datetime.now().isoformat(),
            'total_models_found': sum(len(models) for models in self.found_results.values()),
            'categories': {}
        }

        report_lines = [
            "# MALARIA DETECTION - RINGKASAN LENGKAP HASIL EKSPERIMEN",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## RINGKASAN TOTAL",
        ]

        # Count by category
        for category, models in self.found_results.items():
            count = len(models)
            summary_data['categories'][category] = count
            category_name = category.replace('_', ' ').title()
            report_lines.append(f"- {category_name}: {count} model")

        report_lines.extend([
            f"- **TOTAL**: {summary_data['total_models_found']} model",
            "",
            "## HASIL TERBAIK PER KATEGORI",
            ""
        ])

        # Find best models per category
        best_models = {}
        for category, models in self.found_results.items():
            if models:
                # Find model with highest accuracy
                best_model = None
                best_accuracy = -1

                for model in models:
                    if model['has_results'] and model['results']:
                        accuracy = self._extract_best_accuracy(model['results'])
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_model = model

                if best_model:
                    best_models[category] = {
                        'model': best_model,
                        'accuracy': best_accuracy
                    }

        # Add best models to report
        for category, best_info in best_models.items():
            category_name = category.replace('_', ' ').title()
            model = best_info['model']
            accuracy = best_info['accuracy']

            report_lines.extend([
                f"### {category_name}",
                f"- **Model Terbaik**: {model['name']}",
                f"- **Akurasi**: {accuracy:.2f}%",
                f"- **Path**: {model['path']}",
                ""
            ])

        # Detailed breakdown
        report_lines.extend([
            "## DETAIL SEMUA MODEL",
            ""
        ])

        for category, models in self.found_results.items():
            if models:
                category_name = category.replace('_', ' ').title()
                report_lines.extend([
                    f"### {category_name} ({len(models)} model)",
                    ""
                ])

                # Sort by accuracy if available
                models_with_acc = []
                models_without_acc = []

                for model in models:
                    if model['has_results'] and model['results']:
                        accuracy = self._extract_best_accuracy(model['results'])
                        if accuracy > 0:
                            models_with_acc.append((model, accuracy))
                        else:
                            models_without_acc.append(model)
                    else:
                        models_without_acc.append(model)

                # Sort by accuracy descending
                models_with_acc.sort(key=lambda x: x[1], reverse=True)

                # Add sorted models to report
                for i, (model, accuracy) in enumerate(models_with_acc):
                    report_lines.extend([
                        f"{i+1}. **{model['name']}**",
                        f"   - Akurasi: {accuracy:.2f}%",
                        f"   - Weights: {'✅' if model['has_weights'] else '❌'}",
                        f"   - Path: {model['path']}",
                        ""
                    ])

                # Add models without accuracy
                for model in models_without_acc:
                    report_lines.extend([
                        f"- **{model['name']}**",
                        f"  - Weights: {'✅' if model['has_weights'] else '❌'}",
                        f"  - Results: {'✅' if model['has_results'] else '❌'}",
                        f"  - Path: {model['path']}",
                        ""
                    ])

        # Save report
        report_file = self.consolidated_dir / "RINGKASAN_LENGKAP.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        # Save JSON data
        json_file = self.consolidated_dir / "summary_data.json"
        with open(json_file, 'w') as f:
            json.dump({
                'summary': summary_data,
                'best_models': best_models,
                'all_results': self.found_results
            }, f, indent=2, default=str)

        print(f"✅ Report saved: {report_file}")
        print(f"💾 JSON data saved: {json_file}")

        return best_models

    def _extract_best_accuracy(self, results: dict) -> float:
        """Extract best accuracy from results"""
        accuracy = 0.0

        # Look for various accuracy keys
        accuracy_keys = [
            'Test Acc', 'test_acc', 'Test Accuracy', 'test_accuracy',
            'Val Acc', 'val_acc', 'Validation Accuracy', 'validation_accuracy',
            'Best Val Acc', 'best_val_acc', 'Best Acc', 'best_acc',
            'Accuracy', 'accuracy'
        ]

        for key in accuracy_keys:
            if key in results:
                try:
                    acc_value = results[key]
                    if isinstance(acc_value, (int, float)):
                        accuracy = max(accuracy, float(acc_value))
                    elif isinstance(acc_value, str):
                        # Try to extract number from string
                        import re
                        numbers = re.findall(r'-?\d+\.?\d*', acc_value)
                        if numbers:
                            accuracy = max(accuracy, float(numbers[0]))
                except:
                    continue

        return accuracy

    def create_visualization(self, best_models):
        """Create visualization of results"""
        print("📈 Creating visualizations...")

        # Prepare data for plotting
        categories = []
        accuracies = []
        model_names = []

        for category, best_info in best_models.items():
            categories.append(category.replace('_', ' ').title())
            accuracies.append(best_info['accuracy'])
            model_names.append(best_info['model']['name'][:20] + '...' if len(best_info['model']['name']) > 20 else best_info['model']['name'])

        if categories:
            # Create bar plot
            plt.figure(figsize=(14, 8))

            bars = plt.bar(categories, accuracies, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])

            # Add value labels on bars
            for bar, accuracy, name in zip(bars, accuracies, model_names):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{accuracy:.1f}%',
                        ha='center', va='bottom', fontweight='bold')
                plt.text(bar.get_x() + bar.get_width()/2., height/2,
                        name,
                        ha='center', va='center', rotation=90, fontsize=9)

            plt.title('Model Terbaik Per Kategori - Akurasi Test', fontsize=16, fontweight='bold')
            plt.ylabel('Test Accuracy (%)', fontsize=12)
            plt.xlabel('Kategori Model', fontsize=12)
            plt.ylim(0, 105)
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()

            # Save plot
            plot_file = self.consolidated_dir / "best_models_comparison.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📊 Visualization saved: {plot_file}")

    def copy_best_models(self, best_models):
        """Copy best model weights to consolidated directory"""
        print("💾 Copying model weights terbaik...")

        best_models_dir = self.consolidated_dir / "best_models"
        best_models_dir.mkdir(exist_ok=True)

        for category, best_info in best_models.items():
            model = best_info['model']

            if model['has_weights']:
                # Create category directory
                category_dir = best_models_dir / category
                category_dir.mkdir(exist_ok=True)

                model_dir = Path(model['path'])

                try:
                    # Copy weights directory
                    weights_src = model_dir / "weights"
                    if weights_src.exists():
                        weights_dst = category_dir / "weights"
                        if weights_dst.exists():
                            shutil.rmtree(weights_dst)
                        shutil.copytree(weights_src, weights_dst)

                    # Copy results file
                    results_src = model_dir / "results.txt"
                    if results_src.exists():
                        results_dst = category_dir / "results.txt"
                        shutil.copy2(results_src, results_dst)

                    # Create info file
                    info_file = category_dir / "model_info.json"
                    with open(info_file, 'w') as f:
                        json.dump({
                            'model_name': model['name'],
                            'accuracy': best_info['accuracy'],
                            'original_path': model['path'],
                            'category': category,
                            'copied_at': datetime.now().isoformat()
                        }, f, indent=2)

                    print(f"✅ Copied {category}: {model['name']}")

                except Exception as e:
                    print(f"⚠️ Error copying {model['name']}: {e}")

def main():
    print("=" * 60)
    print("MALARIA DETECTION - CONSOLIDATION HASIL")
    print("=" * 60)

    consolidator = ResultsConsolidator()

    # Scan all results
    consolidator.scan_all_results()

    # Generate summary report
    best_models = consolidator.generate_summary_report()

    # Create visualizations
    if best_models:
        consolidator.create_visualization(best_models)
        consolidator.copy_best_models(best_models)

    # Print summary
    total_found = sum(len(models) for models in consolidator.found_results.values())

    print("\n" + "=" * 60)
    print("🎉 CONSOLIDATION SELESAI!")
    print("=" * 60)
    print(f"📊 Total model ditemukan: {total_found}")
    print(f"🏆 Model terbaik per kategori: {len(best_models)}")
    print(f"📁 Hasil tersimpan di: {consolidator.consolidated_dir}")
    print(f"📄 Baca ringkasan: {consolidator.consolidated_dir}/RINGKASAN_LENGKAP.md")

    # Show best models summary
    if best_models:
        print("\n🏆 MODEL TERBAIK:")
        for category, best_info in best_models.items():
            category_name = category.replace('_', ' ').title()
            print(f"   {category_name}: {best_info['model']['name']} ({best_info['accuracy']:.2f}%)")

if __name__ == "__main__":
    main()