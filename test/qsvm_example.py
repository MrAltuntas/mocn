"""
Complete Example: Training and Using Quantum SVM
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.qsvm import QuantumSVM
from config import *


def prepare_data(n_samples=200, n_features=10, n_classes=2):
    """
    Generate and prepare synthetic data for QSVM

    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        n_classes: Number of classes

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) preprocessed data
    """
    print("=" * 70)
    print("STEP 1: Data Preparation")
    print("=" * 70)

    # Generate synthetic classification data
    print(f"\nGenerating {n_samples} samples with {n_features} features...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=max(n_features - 2, 2),
        n_redundant=2,
        n_classes=n_classes,
        n_clusters_per_class=1,
        weights=None,  # Balanced classes
        flip_y=0.01,  # 1% label noise
        class_sep=1.0,
        random_state=RANDOM_STATE
    )

    print(f"✓ Generated data shape: {X.shape}")
    print(f"✓ Class distribution: {np.bincount(y)}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features standardized (mean=0, std=1)")

    # Apply PCA to reduce dimensionality
    print(f"\nApplying PCA: {n_features} → {PCA_COMPONENTS} dimensions...")
    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"✓ PCA complete")
    print(f"  Explained variance: {explained_var:.2%}")
    print(f"  New shape: {X_train_pca.shape}")

    return X_train_pca, X_test_pca, y_train, y_test, scaler, pca


def train_qsvm(X_train, y_train):
    """
    Create and train Quantum SVM

    Args:
        X_train: Training features (after PCA)
        y_train: Training labels

    Returns:
        Trained QuantumSVM model
    """
    print("\n" + "=" * 70)
    print("STEP 2: Quantum SVM Training")
    print("=" * 70)

    # Initialize QSVM
    qsvm = QuantumSVM(
        n_qubits=PCA_COMPONENTS,
        reps=QSVM_FEATURE_MAP_REPS
    )

    # Display circuit info
    print("\nQuantum Circuit Information:")
    circuit_info = qsvm.get_circuit_info()
    for key, value in circuit_info.items():
        print(f"  {key}: {value}")

    # Train
    print("\nTraining model (this may take a few minutes)...")
    qsvm.train(X_train, y_train, verbose=True)

    return qsvm


def evaluate_model(qsvm, X_test, y_test):
    """
    Evaluate the trained QSVM model

    Args:
        qsvm: Trained QuantumSVM model
        X_test: Test features
        y_test: True test labels

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 70)
    print("STEP 3: Model Evaluation")
    print("=" * 70)

    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = qsvm.predict(X_test)
    y_proba = qsvm.predict_proba(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n{'='*40}")
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"{'='*40}\n")

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    return {
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_proba,
        'confusion_matrix': cm
    }


def visualize_results(y_test, y_pred, y_proba, cm):
    """
    Create visualizations of the results

    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities
        cm: Confusion matrix
    """
    print("\n" + "=" * 70)
    print("STEP 4: Visualization")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix Heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    # Prediction Confidence Distribution
    confidence = np.max(y_proba, axis=1)
    correct = (y_test == y_pred)

    axes[1].hist(confidence[correct], alpha=0.5, label='Correct', bins=20, color='green')
    axes[1].hist(confidence[~correct], alpha=0.5, label='Incorrect', bins=20, color='red')
    axes[1].set_title('Prediction Confidence Distribution')
    axes[1].set_xlabel('Confidence')
    axes[1].set_ylabel('Count')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('qsvm_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to: qsvm_results.png")

    return fig


def save_model(qsvm, path=MODEL_SAVE_PATH):
    """
    Save the trained model

    Args:
        qsvm: Trained QuantumSVM model
        path: Save path
    """
    print("\n" + "=" * 70)
    print("STEP 5: Model Persistence")
    print("=" * 70)

    import os
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    qsvm.save(path)

    # Test loading
    print("\nTesting model loading...")
    loaded_qsvm = QuantumSVM.load(path)
    print(f"✓ Model successfully loaded")
    print(f"\n{loaded_qsvm}")


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 70)
    print("QUANTUM SVM - COMPLETE WORKFLOW")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Qubits: {PCA_COMPONENTS}")
    print(f"  Feature Map Reps: {QSVM_FEATURE_MAP_REPS}")
    print(f"  Random Seed: {RANDOM_STATE}")
    print("=" * 70)

    # Prepare data
    X_train, X_test, y_train, y_test, scaler, pca = prepare_data(
        n_samples=200,
        n_features=10,
        n_classes=2
    )

    # Train QSVM
    qsvm = train_qsvm(X_train, y_train)

    # Evaluate
    results = evaluate_model(qsvm, X_test, y_test)

    # Visualize
    try:
        visualize_results(
            y_test,
            results['predictions'],
            results['probabilities'],
            results['confusion_matrix']
        )
    except Exception as e:
        print(f"\nVisualization skipped: {e}")

    # Save model
    save_model(qsvm)

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE!")
    print("=" * 70)
    print(f"\nFinal Test Accuracy: {results['accuracy']:.2%}")
    print("\nFiles created:")
    print(f"  - {MODEL_SAVE_PATH}")
    print(f"  - qsvm_results.png (if matplotlib available)")

    return qsvm, results


if __name__ == "__main__":
    # Set numpy random seed for reproducibility
    np.random.seed(RANDOM_STATE)

    # Run the complete workflow
    qsvm, results = main()

    print("\n" + "=" * 70)
    print("Quick Reference - Using the Model:")
    print("=" * 70)
    print("""
# Load a saved model:
from qsvm import QuantumSVM
model = QuantumSVM.load('models/qsvm_model.pkl')

# Make predictions on new data (remember to preprocess first!):
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

# Get model info:
print(model)
print(model.get_params())
print(model.get_circuit_info())
    """)