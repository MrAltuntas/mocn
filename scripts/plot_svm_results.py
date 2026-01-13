#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import config


class CVResultsPlotter:
    def __init__(self, results_path=None):
        if results_path is None:
            # Try CV results first
            cv_path = project_root / config.METRICS_DIR / 'experiment_results_cv.json'
            if cv_path.exists():
                results_path = str(cv_path)
            else:
                # Fallback to legacy
                results_path = str(project_root / config.METRICS_DIR / 'experiment_results.json')

        self.results_path = results_path
        self.data = None
        self.plots_dir = str(project_root / config.PLOTS_DIR)
        self.dpi = config.PLOT_DPI
        self.format = config.PLOT_FORMAT
        self.figsize = config.FIGURE_SIZE

        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 13
        plt.rcParams['axes.labelsize'] = 12

    def load_results(self):
        if not os.path.exists(self.results_path):
            raise FileNotFoundError(f"Results file not found: {self.results_path}")
        with open(self.results_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded results from: {self.results_path}")
        return self.data

    def _save_plot(self, fig, filename):
        os.makedirs(self.plots_dir, exist_ok=True)
        filepath = os.path.join(self.plots_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', format=self.format)
        plt.close(fig)
        print(f"Saved plot: {filepath}")

    def plot_confusion_matrix(self, model_key, title_suffix):
        if model_key not in self.data or self.data[model_key] is None:
            return

        # Aggregate confusion matrices from all folds
        folds = self.data[model_key]['fold_metrics']
        
        # Initialize with first fold to get shape
        total_cm = np.array(folds[0]['confusion_matrix'])
        
        # Sum the rest
        for fold in folds[1:]:
            total_cm += np.array(fold['confusion_matrix'])
            
        # Plot
        n_classes = total_cm.shape[0]
        class_names = [f'Class {i}' for i in range(n_classes)]

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Total Sample Count (All Folds)'}, ax=ax)

        ax.set_title(f'Aggregated Confusion Matrix - {title_suffix}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)

        # Add accuracy text
        accuracy = self.data[model_key]['metrics']['accuracy']
        std = self.data[model_key]['metrics'].get('accuracy_std', 0.0)
        fig.text(0.99, 0.01, f'Avg Accuracy: {accuracy:.4f} (±{std:.4f})',
                ha='right', va='bottom', fontsize=10, style='italic')

        self._save_plot(fig, f'confusion_matrix_{model_key}.png')

    def plot_metrics_comparison(self):
        models = []
        if 'classical_svm' in self.data and self.data['classical_svm']:
            models.append(('Classical SVM', self.data['classical_svm']))
        if 'quantum_svm' in self.data and self.data['quantum_svm']:
            models.append(('Quantum SVM', self.data['quantum_svm']))
            
        if not models:
            return

        # Prepare DataFrame for plotting
        metrics_to_plot = ['accuracy', 'f1', 'precision', 'recall']
        plot_data = []
        
        for name, data in models:
            m = data['metrics']
            for metric in metrics_to_plot:
                plot_data.append({
                    'Model': name,
                    'Metric': metric.capitalize(),
                    'Score': m.get(metric, 0),
                    'Std': m.get(f'{metric}_std', 0)
                })
        
        df = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar plot with error bars
        sns.barplot(data=df, x='Metric', y='Score', hue='Model', 
                    palette='muted', ax=ax, capsize=0.1)
        
        # Add error bars manually if needed, but sns.barplot with CI is tricky with pre-computed std
        # So we just trust the bars for now or overlay error bars if we had raw data.
        # Since we have pre-computed Mean/Std, we can just interpret the bar as the Mean.
        
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison (10-Fold CV)', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='lower right')
        
        # Add values on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)

        self._save_plot(fig, 'model_comparison.png')

    def generate_all_plots(self):
        print("Generating plots...")
        self.load_results()
        
        self.plot_confusion_matrix('classical_svm', 'Classical SVM')
        self.plot_confusion_matrix('quantum_svm', 'Quantum SVM')
        
        self.plot_metrics_comparison()
        print("Done.")


def main():
    try:
        plotter = CVResultsPlotter()
        plotter.generate_all_plots()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
