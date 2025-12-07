import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from config import *


def load_data(dataset_name='kdd99'):
    """
    Load dataset from various sources

    Args:
        dataset_name (str): Dataset to load ('netflow', 'kdd99', 'cores_iot', 'synthetic')

    Returns:
        X (ndarray): Feature matrix
        y (ndarray): Labels
    """
    if dataset_name == 'kdd99':
        print(f"Loading KDD99 dataset from {KDD99_PATH}...")
        X, y = _load_kdd99()

    elif dataset_name == 'netflow':
        print(f"Loading NetFlow dataset from {NETFLOW_PATH}...")
        X, y = _load_netflow()

    elif dataset_name == 'cores_iot':
        print(f"Loading CORES IoT dataset from {CORES_IOT_PATH}...")
        X, y = _load_cores_iot()

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return X, y

def _load_kdd99():
    """
    Load KDD99 dataset from raw numpy files
    Performs one-hot encoding and creates multi-class labels

    Returns:
        X (ndarray): Feature matrix (one-hot encoded, ~120 features)
        y (ndarray): Multi-class labels (0-12, representing 13 attack types)
    """
    import os

    # Load raw numpy files
    X_file = os.path.join(KDD99_PATH, 'X_raw.npy')
    y_file = os.path.join(KDD99_PATH, 'y_raw.npy')

    if not os.path.exists(X_file) or not os.path.exists(y_file):
        print(f"ERROR: KDD99 raw files not found at {KDD99_PATH}")
        print("Please run: python scripts/download_kdd99.py")
        print("\nUsing synthetic data instead...")
        return

    print(f"Loading raw data from {KDD99_PATH}...")
    X_raw = np.load(X_file, allow_pickle=True)
    y_raw = np.load(y_file, allow_pickle=True)

    print(f"Loaded {len(X_raw)} samples with {X_raw.shape[1]} raw features")

    # Column names (41 features)
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
    ]

    # Create DataFrame for easier processing
    df = pd.DataFrame(X_raw, columns=columns)

    # Handle categorical features (one-hot encoding)
    # Note: protocol_type, service, flag are stored as bytes
    categorical_cols = ['protocol_type', 'service', 'flag']

    # Convert bytes to strings for proper encoding
    for col in categorical_cols:
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))

    # One-hot encode (drop_first=False to keep all information for multi-class)
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    print(f"After one-hot encoding: {df.shape[1]} features")

    # Convert to numpy array
    X = df.values.astype(np.float64)

    # Process labels - filter rare classes and create multi-class mapping
    unique_labels = np.unique(y_raw)
    print(f"\nFound {len(unique_labels)} unique attack types:")

    # Filter out rare classes (less than MIN_SAMPLES_PER_CLASS)
    label_counts = {label: np.sum(y_raw == label) for label in unique_labels}
    valid_labels = sorted([label for label, count in label_counts.items() if count >= MIN_SAMPLES_PER_CLASS])
    removed_labels = sorted([label for label, count in label_counts.items() if count < MIN_SAMPLES_PER_CLASS])

    # Print all labels with filtering info
    for label in sorted(unique_labels):
        count = label_counts[label]
        label_str = label.decode('utf-8') if isinstance(label, bytes) else str(label)
        status = "✓ KEPT" if label in valid_labels else "✗ REMOVED (rare)"
        print(f"  {label_str:20s} - {count:6d} samples ({count/len(y_raw)*100:.2f}%) {status}")

    # Filter dataset to keep only valid classes
    valid_mask = np.isin(y_raw, valid_labels)
    X = X[valid_mask]
    y_raw_filtered = y_raw[valid_mask]

    print(f"\nAfter filtering rare classes:")
    print(f"  Kept:    {len(valid_labels)} classes with {np.sum(valid_mask)} samples")
    print(f"  Removed: {len(removed_labels)} classes with {np.sum(~valid_mask)} samples")

    # Create label mapping (alphabetically sorted for consistency)
    label_mapping = {label: idx for idx, label in enumerate(valid_labels)}

    # Convert labels to integers
    y = np.array([label_mapping[label] for label in y_raw_filtered])

    print(f"\nFinal dataset shape: X={X.shape}, y={y.shape}")
    print(f"Final class distribution: {np.bincount(y)}")

    return X, y

def _load_netflow():
    """
    Load NetFlow dataset (placeholder)
    TODO: Implement actual NetFlow loading logic
    """
    print("WARNING: NetFlow loader not implemented. Using synthetic data.")
    return

def _load_cores_iot():
    """
    Load CORES IoT dataset (placeholder)
    TODO: Implement actual CORES IoT loading logic
    """
    print("WARNING: CORES IoT loader not implemented. Using synthetic data.")
    return

def preprocess_data(X, y):
    """
    Preprocessing pipeline:
    1. Train/test split
    2. Normalize features (fit only on train)
    3. Apply PCA (fit only on train)
    """
    print("\n" + "="*50)
    print("PREPROCESSING PIPELINE")
    print("="*50)

    # 1. Train/Test Split (önce)
    print(f"[1/3] Splitting data (train={TRAIN_TEST_SPLIT:.0%}, test={1-TRAIN_TEST_SPLIT:.0%})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=1-TRAIN_TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=y
    )

    print(f"      Train: {X_train.shape[0]} samples")
    print(f"      Test:  {X_test.shape[0]} samples")
    print(f"      Class distribution (train): {np.bincount(y_train)}")
    print(f"      Class distribution (test):  {np.bincount(y_test)}")

    # 2. Normalization (fit sadece train üzerinde)
    if NORMALIZE:
        print(f"[2/3] Normalizing features using {NORMALIZATION_METHOD}...")

        if NORMALIZATION_METHOD == 'minmax':
            scaler = MinMaxScaler()
        elif NORMALIZATION_METHOD == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {NORMALIZATION_METHOD}")

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        print(f"      Train range: {X_train.min():.2f} - {X_train.max():.2f}")
        print(f"      Test  range: {X_test.min():.2f} - {X_test.max():.2f}")
    else:
        print("[2/3] Normalization disabled.")

    # 3. PCA (fit yine sadece train üzerinde)
    if USE_PCA:
        print(f"[3/3] Applying PCA with {PCA_COMPONENTS} components...")
        original_features = X_train.shape[1]

        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
        pca.fit(X_train)

        X_train = pca.transform(X_train)
        X_test = pca.transform(X_test)

        variance_explained = pca.explained_variance_ratio_.sum()
        print(f"      Reduced from {original_features} to {PCA_COMPONENTS} features")
        print(f"      Explained variance (train): {variance_explained:.2%}")
    else:
        print("[3/3] PCA disabled.")

    return X_train, X_test, y_train, y_test

def get_dataset_info(X, y):
    """
    Get information about the dataset

    Args:
        X (ndarray): Feature matrix
        y (ndarray): Labels

    Returns:
        info (dict): Dataset information
    """
    info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': dict(zip(*np.unique(y, return_counts=True))),
        'feature_range': (X.min(), X.max())
    }

    return info
