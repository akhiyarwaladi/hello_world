#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import json
from collections import defaultdict
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ensemble_predictions.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class EnsembleMalariaPredictor:
    def __init__(self, model_results_dir, output_dir):
        self.model_results_dir = Path(model_results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging()

        self.class_names = ["P_falciparum", "P_vivax", "P_malariae", "P_ovale", "Mixed_infection", "Uninfected"]

    def load_model_predictions(self):
        """Load predictions from all trained models"""
        predictions = {}

        # Scan through all result directories
        for model_type in ["detection", "classification"]:
            model_dir = self.model_results_dir / model_type
            if not model_dir.exists():
                continue

            for experiment_dir in model_dir.iterdir():
                if not experiment_dir.is_dir():
                    continue

                # Look for prediction files
                pred_files = list(experiment_dir.glob("**/predictions.json"))
                if not pred_files:
                    pred_files = list(experiment_dir.glob("**/results.csv"))

                for pred_file in pred_files:
                    model_name = f"{model_type}_{experiment_dir.name}"
                    try:
                        predictions[model_name] = self._load_prediction_file(pred_file)
                        self.logger.info(f"Loaded predictions from {model_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load {pred_file}: {e}")

        return predictions

    def _load_prediction_file(self, file_path):
        """Load predictions from file"""
        if file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                return json.load(f)
        elif file_path.suffix == '.csv':
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def voting_ensemble(self, predictions, method='hard'):
        """Create voting ensemble predictions"""
        ensemble_results = {}

        if method == 'hard':
            # Hard voting - majority vote
            for image_id in self._get_common_images(predictions):
                votes = []
                for model_name, model_preds in predictions.items():
                    if image_id in model_preds:
                        votes.append(model_preds[image_id]['predicted_class'])

                # Get majority vote
                vote_counts = defaultdict(int)
                for vote in votes:
                    vote_counts[vote] += 1

                ensemble_results[image_id] = {
                    'predicted_class': max(vote_counts, key=vote_counts.get),
                    'confidence': vote_counts[max(vote_counts, key=vote_counts.get)] / len(votes),
                    'individual_votes': votes
                }

        elif method == 'soft':
            # Soft voting - average probabilities
            for image_id in self._get_common_images(predictions):
                prob_sum = np.zeros(len(self.class_names))
                model_count = 0

                for model_name, model_preds in predictions.items():
                    if image_id in model_preds and 'probabilities' in model_preds[image_id]:
                        prob_sum += np.array(model_preds[image_id]['probabilities'])
                        model_count += 1

                if model_count > 0:
                    avg_probs = prob_sum / model_count
                    ensemble_results[image_id] = {
                        'predicted_class': self.class_names[np.argmax(avg_probs)],
                        'confidence': float(np.max(avg_probs)),
                        'probabilities': avg_probs.tolist(),
                        'model_count': model_count
                    }

        return ensemble_results

    def weighted_ensemble(self, predictions, model_weights=None):
        """Create weighted ensemble based on model performance"""
        if model_weights is None:
            # Use equal weights if not provided
            model_weights = {name: 1.0 for name in predictions.keys()}

        ensemble_results = {}

        for image_id in self._get_common_images(predictions):
            weighted_prob_sum = np.zeros(len(self.class_names))
            total_weight = 0

            for model_name, model_preds in predictions.items():
                if image_id in model_preds and 'probabilities' in model_preds[image_id]:
                    weight = model_weights.get(model_name, 1.0)
                    weighted_prob_sum += weight * np.array(model_preds[image_id]['probabilities'])
                    total_weight += weight

            if total_weight > 0:
                final_probs = weighted_prob_sum / total_weight
                ensemble_results[image_id] = {
                    'predicted_class': self.class_names[np.argmax(final_probs)],
                    'confidence': float(np.max(final_probs)),
                    'probabilities': final_probs.tolist(),
                    'total_weight': total_weight
                }

        return ensemble_results

    def stacking_ensemble(self, predictions, meta_learner='random_forest'):
        """Create stacking ensemble with meta-learner"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict
            from sklearn.preprocessing import LabelEncoder
        except ImportError:
            self.logger.error("scikit-learn required for stacking ensemble")
            return {}

        # Prepare features and labels
        X, y, image_ids = self._prepare_stacking_data(predictions)

        if len(X) == 0:
            self.logger.warning("No data available for stacking ensemble")
            return {}

        # Choose meta-learner
        if meta_learner == 'random_forest':
            meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            meta_model = LogisticRegression(random_state=42)

        # Train meta-learner
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Use cross-validation to get meta-features
        meta_predictions = cross_val_predict(meta_model, X, y_encoded, cv=5, method='predict_proba')

        # Fit final model
        meta_model.fit(X, y_encoded)

        # Generate ensemble results
        ensemble_results = {}
        for i, image_id in enumerate(image_ids):
            pred_class_idx = np.argmax(meta_predictions[i])
            ensemble_results[image_id] = {
                'predicted_class': le.inverse_transform([pred_class_idx])[0],
                'confidence': float(np.max(meta_predictions[i])),
                'probabilities': meta_predictions[i].tolist()
            }

        return ensemble_results

    def _get_common_images(self, predictions):
        """Get images that appear in all models"""
        if not predictions:
            return []

        image_sets = [set(pred.keys()) for pred in predictions.values()]
        return list(set.intersection(*image_sets))

    def _prepare_stacking_data(self, predictions):
        """Prepare data for stacking ensemble"""
        common_images = self._get_common_images(predictions)
        X, y, image_ids = [], [], []

        for image_id in common_images:
            features = []
            true_label = None

            for model_name, model_preds in predictions.items():
                if image_id in model_preds:
                    if 'probabilities' in model_preds[image_id]:
                        features.extend(model_preds[image_id]['probabilities'])
                    else:
                        # Use one-hot encoding for hard predictions
                        one_hot = [0] * len(self.class_names)
                        class_idx = self.class_names.index(model_preds[image_id]['predicted_class'])
                        one_hot[class_idx] = 1
                        features.extend(one_hot)

                    if true_label is None and 'true_class' in model_preds[image_id]:
                        true_label = model_preds[image_id]['true_class']

            if len(features) == len(predictions) * len(self.class_names) and true_label:
                X.append(features)
                y.append(true_label)
                image_ids.append(image_id)

        return np.array(X), y, image_ids

    def evaluate_ensemble(self, ensemble_results, ground_truth):
        """Evaluate ensemble performance"""
        from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

        y_true, y_pred = [], []

        for image_id, result in ensemble_results.items():
            if image_id in ground_truth:
                y_true.append(ground_truth[image_id])
                y_pred.append(result['predicted_class'])

        if not y_true:
            self.logger.warning("No ground truth available for evaluation")
            return {}

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=self.class_names)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix.tolist(),
            'num_samples': len(y_true)
        }

    def save_ensemble_results(self, ensemble_results, method_name):
        """Save ensemble results to file"""
        output_file = self.output_dir / f"ensemble_{method_name}_results.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for image_id, result in ensemble_results.items():
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            serializable_results[image_id] = serializable_result

        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        self.logger.info(f"Saved ensemble results to {output_file}")
        return output_file

    def run_all_ensembles(self):
        """Run all ensemble methods"""
        self.logger.info("Loading model predictions...")
        predictions = self.load_model_predictions()

        if not predictions:
            self.logger.error("No predictions found!")
            return

        self.logger.info(f"Found predictions from {len(predictions)} models")

        ensemble_methods = {
            'hard_voting': lambda: self.voting_ensemble(predictions, 'hard'),
            'soft_voting': lambda: self.voting_ensemble(predictions, 'soft'),
            'weighted': lambda: self.weighted_ensemble(predictions),
            'stacking_rf': lambda: self.stacking_ensemble(predictions, 'random_forest'),
            'stacking_lr': lambda: self.stacking_ensemble(predictions, 'logistic_regression')
        }

        results = {}
        for method_name, method_func in ensemble_methods.items():
            try:
                self.logger.info(f"Running {method_name} ensemble...")
                ensemble_results = method_func()

                if ensemble_results:
                    self.save_ensemble_results(ensemble_results, method_name)
                    results[method_name] = ensemble_results
                    self.logger.info(f"Completed {method_name} ensemble with {len(ensemble_results)} predictions")
                else:
                    self.logger.warning(f"No results from {method_name} ensemble")

            except Exception as e:
                self.logger.error(f"Error in {method_name} ensemble: {e}")

        # Generate summary report
        summary_file = self.output_dir / "ensemble_summary.json"
        summary = {
            'total_models': len(predictions),
            'ensemble_methods': list(results.keys()),
            'model_names': list(predictions.keys()),
            'results_summary': {
                method: len(result) for method, result in results.items()
            }
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Generated ensemble summary: {summary_file}")
        return results

def main():
    parser = argparse.ArgumentParser(description='Create ensemble predictions for malaria detection')
    parser.add_argument('--results_dir', type=str, default='results/current_experiments/training',
                       help='Directory containing model results')
    parser.add_argument('--output_dir', type=str, default='results/ensemble_predictions',
                       help='Output directory for ensemble results')
    parser.add_argument('--method', type=str, choices=['all', 'voting', 'weighted', 'stacking'],
                       default='all', help='Ensemble method to use')

    args = parser.parse_args()

    ensemble_predictor = EnsembleMalariaPredictor(args.results_dir, args.output_dir)

    if args.method == 'all':
        ensemble_predictor.run_all_ensembles()
    else:
        # Run specific method
        predictions = ensemble_predictor.load_model_predictions()

        if args.method == 'voting':
            results = ensemble_predictor.voting_ensemble(predictions, 'soft')
        elif args.method == 'weighted':
            results = ensemble_predictor.weighted_ensemble(predictions)
        elif args.method == 'stacking':
            results = ensemble_predictor.stacking_ensemble(predictions)

        ensemble_predictor.save_ensemble_results(results, args.method)

if __name__ == "__main__":
    main()