#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
import config


class SVMPlotter:
    def __init__(self, results_path=None):
        if results_path is None:
            results_path = project_root / config.METRICS_DIR / 'experiment_results.json'
            results_path = str(results_path)

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
        if 'classical_svm' not in self.data:
            raise ValueError("'classical_svm' key not found in JSON")
        return self.data

    def _save_plot(self, fig, filename):
        os.makedirs(self.plots_dir, exist_ok=True)
        filepath = os.path.join(self.plots_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', format=self.format)
        plt.close(fig)

    def plot_confusion_matrix(self):
        cm = np.array(self.data['classical_svm']['metrics']['confusion_matrix'])
        n_classes = cm.shape[0]
        class_names = [f'Class {i}' for i in range(n_classes)]

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Sample Count'}, ax=ax)

        ax.set_title('Confusion Matrix - Classical SVM', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)

        accuracy = self.data['classical_svm']['metrics']['accuracy']
        fig.text(0.99, 0.01, f'Accuracy: {accuracy:.4%}',
                ha='right', va='bottom', fontsize=10, style='italic')

        self._save_plot(fig, 'confusion_matrix_classical.png')

    def plot_roc_curve(self):
        if 'y_proba' not in self.data['classical_svm'] or self.data['classical_svm']['y_proba'] is None:
            return

        y_proba = np.array(self.data['classical_svm']['y_proba'])
        y_true = np.array(self.data['classical_svm']['y_true'])
        n_classes = y_proba.shape[1]

        fig, ax = plt.subplots(figsize=self.figsize)

        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            y_score = y_proba[:, i]
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve - Classical SVM (One-vs-Rest)', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        self._save_plot(fig, 'roc_curve_classical.png')

    def plot_precision_recall_curve(self):
        if 'y_proba' not in self.data['classical_svm'] or self.data['classical_svm']['y_proba'] is None:
            return

        y_proba = np.array(self.data['classical_svm']['y_proba'])
        y_true = np.array(self.data['classical_svm']['y_true'])
        n_classes = y_proba.shape[1]

        fig, ax = plt.subplots(figsize=self.figsize)

        for i in range(n_classes):
            y_true_binary = (y_true == i).astype(int)
            y_score = y_proba[:, i]
            precision, recall, _ = precision_recall_curve(y_true_binary, y_score)
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, lw=2, label=f'Class {i} (AUC = {pr_auc:.3f})')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve - Classical SVM (One-vs-Rest)',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        self._save_plot(fig, 'pr_curve_classical.png')

    def plot_metrics_overview(self):
        metrics = self.data['classical_svm']['metrics']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'MCC']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1'],
            metrics['mcc']
        ]

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(metric_names, metric_values, color='steelblue', alpha=0.8, edgecolor='black')

        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Performance Metrics - Classical SVM', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=1.0, color='red', linestyle='--', lw=1, alpha=0.5, label='Perfect Score')
        ax.legend(loc='lower right')

        self._save_plot(fig, 'metrics_overview.png')

    def generate_all_plots(self):
        self.load_results()
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.plot_metrics_overview()


def main():
    try:
        plotter = SVMPlotter()
        plotter.generate_all_plots()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
