# MOCN - Quantum vs Classical SVM for Intrusion Detection

Comparative study of Classical SVM vs Quantum SVM for network intrusion detection using Qiskit.

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
If you don't have data yet, run:
```bash
python scripts/download_kdd99.py
```
This downloads the KDD Cup 99 dataset (10% subset) to `data/raw/kdd99/`.

### 3. Configure Settings
Edit `config.py` to adjust parameters:
```python
DATASET_NAME = 'kdd99'        # Dataset to use
SKIP_QUANTUM = True           # Set False to train Quantum SVM
NORMALIZE = True              # Normalize features
USE_PCA = True                # Apply PCA dimensionality reduction
PCA_COMPONENTS = 10           # Number of PCA components
```

### 4. Run Experiment
```bash
python main.py
```
Results will be saved to `results/metrics/experiment_results.json`.

---

## Configuration (`config.py`)

All experiment settings are centralized in `config.py`:

**General Settings:**
- `RANDOM_SEED`: Reproducibility seed
- `TRAIN_TEST_SPLIT`: Train/test split ratio (default: 0.8)
- `SKIP_QUANTUM`: Skip quantum SVM training for faster testing
- `VERBOSE`: Print detailed results
- `SAVE_MODELS`: Save trained models to disk

**Dataset Settings:**
- `DATASET_NAME`: Which dataset to use ('kdd99', 'netflow', 'cores_iot')
- `MIN_SAMPLES_PER_CLASS`: Filter out rare attack classes

**Preprocessing:**
- `NORMALIZE`: Apply normalization (True/False)
- `NORMALIZATION_METHOD`: 'minmax' or 'standard'
- `USE_PCA`: Apply PCA dimensionality reduction
- `PCA_COMPONENTS`: Number of components to keep

**Classical SVM:**
- `SVM_KERNEL`: Kernel type ('rbf', 'linear', 'poly')
- `SVM_C`: Regularization parameter
- `SVM_GAMMA`: Kernel coefficient

**Quantum SVM:**
- `QSVM_N_QUBITS`: Number of qubits (should match `PCA_COMPONENTS`)
- `QSVM_FEATURE_MAP_REPS`: Feature map repetitions
- `QSVM_SHOTS`: None for exact simulation, or int for sampling

---

## Data Loading & Preprocessing

### KDD99 Dataset (`src/data_processing.py`)

**Loading Process:**
1. Loads raw numpy arrays from `data/raw/kdd99/X_raw.npy` and `y_raw.npy`
2. **One-hot encodes** categorical features (protocol_type, service, flag)
   - Machine learning models require numerical inputs, not text/categorical values
   - Converts categorical variables into binary vectors (0/1)
   - Results in ~120 features from original 41
3. **Filters out rare attack classes** (< `MIN_SAMPLES_PER_CLASS` samples)
   - Rare classes with insufficient samples lead to poor model generalization
   - Prevents overfitting and improves training stability
4. **Creates multi-class labels** (0-N representing different attack types)
   - **Important:** This is not just binary attack detection (attack vs normal)
   - We classify attacks into their specific types (e.g., DoS, probe, U2R, R2L)
   - Enables more granular threat identification and response

**Preprocessing Pipeline:**
1. **Train/Test Split** - Stratified split (default 80/20)
2. **Normalization** - MinMax or Standard scaling (fit on train only)
   - **Why:** Features have different scales (e.g., duration: 0-58329, protocol_type: 0-1)
   - SVM is sensitive to feature scales; normalization ensures equal contribution
   - Prevents features with larger ranges from dominating the model
3. **PCA** - Dimensionality reduction (fit on train only)
   - **Why:** Reduces computational complexity for Quantum SVM (fewer qubits needed)
   - Removes redundant/correlated features while retaining essential information
   - Reduces ~120 features to 10 components
   - Preserves ~94% of variance while making quantum circuits tractable

Key function: `preprocess_data(X, y)` returns `X_train, X_test, y_train, y_test`

---

## Training & Evaluation

### Classical SVM (`src/svm.py`)
- Uses scikit-learn's SVC with configurable kernel
- Fast training (~0.15s on small datasets)
- Supports RBF, linear, polynomial kernels

### Quantum SVM (`src/qsvm.py`)
- Uses Qiskit's quantum kernel (ZZFeatureMap)
- Requires data dimensionality = number of qubits
- Slower training (~2-5s) but explores quantum advantage
- Statevector simulator for exact computation

### Trainer (`src/trainer.py`)
Main training function: `train_and_evaluate(model, X_train, X_test, y_train, y_test)`

**Process:**
1. Train model on training data
2. Measure training time
3. Make predictions on test set
4. Calculate comprehensive metrics
5. Return results dictionary

### Evaluation Metrics (`src/evaluation.py`)
- Accuracy, Precision, Recall, F1 Score
- False Positive Rate (FPR), False Negative Rate (FNR)
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix
- Per-class metrics for multi-class problems

---

## Scripts

### `scripts/download_kdd99.py`
Downloads KDD Cup 99 dataset (10% subset) using scikit-learn.

**Usage:**
```bash
python scripts/download_kdd99.py
```

**Output:**
- `data/raw/kdd99/X_raw.npy` - Feature matrix (N x 41)
- `data/raw/kdd99/y_raw.npy` - Attack labels
- `data/raw/kdd99/kdd99_raw.pkl` - Complete dataset pickle

### `scripts/plot_svm_results.py`
Visualization script for SVM results (generate plots from saved metrics).

---

## Test Folder

### `test/qsvm_example.py`
Standalone example demonstrating complete Quantum SVM workflow:
1. Generate synthetic data
2. Preprocess (standardization + PCA)
3. Train Quantum SVM
4. Evaluate and visualize results
5. Save/load model

**Usage:**
```bash
python test/qsvm_example.py
```

**Output:**
- `test/models/qsvm_model.pkl` - Trained model
- `test/qsvm_results.png` - Visualization (confusion matrix + confidence)

### `test/test_qsvm.py`
Unit tests for Quantum SVM implementation.

---

## Dependencies

- **Quantum:** qiskit, qiskit-machine-learning, qiskit-aer
- **ML:** scikit-learn, imbalanced-learn
- **Data:** numpy, pandas
- **Viz:** matplotlib, seaborn

---

## Author

**Mustafa Altuntaş**

---

**Last Updated:** 2025-12-07