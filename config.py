# ============================================
# GENERAL SETTINGS
# ============================================
RANDOM_SEED = 42
TRAIN_TEST_SPLIT = 0.8

SKIP_QUANTUM = False  # Skip quantum SVM training (faster for testing)
VERBOSE = True  # Print detailed results
SAVE_MODELS = True  # Save trained models to disk

# ============================================
# DATASET PARAMETERS
# ============================================
DATASET_NAME = 'kdd99'  # 'netflow', 'kdd99', 'cores_iot'
MAX_SAMPLES = 3000
MIN_SAMPLES_PER_CLASS = 10  # Minimum samples required per class (rare classes will be filtered out)

DATA_RAW_DIR = 'data/raw'
NETFLOW_PATH = f'{DATA_RAW_DIR}/netflow'
KDD99_PATH = f'{DATA_RAW_DIR}/kdd99'
CORES_IOT_PATH = f'{DATA_RAW_DIR}/cores_iot'

# ============================================
# PREPROCESSING PARAMETERS
# ============================================
# Normalization
NORMALIZE = True
NORMALIZATION_METHOD = 'minmax'  # 'minmax' or 'standard'

# Balancing
USE_SMOTE = True  # Apply SMOTE to balance training data

# PCA (Dimensionality Reduction)
USE_PCA = True
PCA_COMPONENTS = 10

# Cross Validation
CV_FOLDS = 10

# ============================================
# CLASSICAL SVM PARAMETERS
# ============================================
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'
SVM_ITERATION_LOG = False

# ============================================
# QUANTUM SVM PARAMETERS
# ============================================
# Quantum backend
# PCA Configuration
# Quantum SVM Configuration
QSVM_N_QUBITS = 10  # Number of qubits in quantum circuit
QSVM_SHOTS = None  # None for exact statevector, or integer for sampling
QSVM_FEATURE_MAP_REPS = 2  # Number of feature map repetitions
BATCH_QSVM = 1000  # Batch size for quantum kernel evaluation
# QSVM_MAX_SAMPLES removed as we use global MAX_SAMPLES

# Training Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.3

# File paths
MODEL_SAVE_PATH = "models/qsvm_model.pkl"
RESULTS_PATH = "results/"

# ============================================
# RESULTS & PATHS
# ============================================
RESULTS_DIR = 'results'
MODELS_DIR = f'{RESULTS_DIR}/models'
METRICS_DIR = f'{RESULTS_DIR}/metrics'
PLOTS_DIR = f'{RESULTS_DIR}/plots'
REPORTS_DIR = f'{RESULTS_DIR}/reports'

# ============================================
# VISUALIZATION PARAMETERS
# ============================================
PLOT_DPI = 300
PLOT_FORMAT = 'png'
FIGURE_SIZE = (10, 6)
