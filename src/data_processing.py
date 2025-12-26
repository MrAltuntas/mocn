import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, RandomOverSampler
from config import *
import os


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

    if MAX_SAMPLES is not None and MAX_SAMPLES < len(X):
        print(f"\nSubsampling data to {MAX_SAMPLES} samples (stratified)...")
        # Use train_test_split to perform stratified subsampling
        X, _, y, _ = train_test_split(
            X, y,
            train_size=MAX_SAMPLES,
            random_state=RANDOM_SEED,
            stratify=y
        )
        print(f"  New dataset shape: X={X.shape}, y={y.shape}")
        print(f"  New class distribution: {np.bincount(y)}")

    # Filter out classes with only one sample, as this will break stratification
    unique_labels, counts = np.unique(y, return_counts=True)
    single_sample_labels = unique_labels[counts < 2]

    if len(single_sample_labels) > 0:
        print(f"\nFiltering out {len(single_sample_labels)} classes with less than 2 samples...")
        filter_mask = ~np.isin(y, single_sample_labels)
        X = X[filter_mask]
        y = y[filter_mask]
        print(f"  New dataset shape after filtering: X={X.shape}, y={y.shape}")
        print(f"  New class distribution after filtering: {np.bincount(y)}")

    return X, y

def _stratified_subsample(df, y_raw):
    """
    Subsample dataset to MAX_SAMPLES using stratified sampling.
    Handles rare classes and NaNs to prevent errors.
    """
    if MAX_SAMPLES is None or MAX_SAMPLES >= len(df):
        return df, y_raw

    print(f"\nOptimization: Subsampling raw data to {MAX_SAMPLES} samples before processing...")
    
    # Ensure y_raw matches df length
    if len(y_raw) != len(df):
        print(f"  Warning: label length ({len(y_raw)}) != dataframe length ({len(df)}). Skipping subsampling.")
        return df, y_raw

    # Pre-filter NaNs in y_raw
    valid_mask = ~pd.isnull(y_raw)
    if np.sum(~valid_mask) < len(y_raw):
        print(f"  Pre-filtering {np.sum(~valid_mask)} samples with NaN labels to enable stratification...")
        df = df[valid_mask]
        y_raw = y_raw[valid_mask]

    # Pre-filter rare classes based on configuration (must be at least 2 for stratification)
    # This unifies the "Crash Prevention" (<2) and "Experiment Logic" (<MIN_SAMPLES)
    filter_threshold = max(2, MIN_SAMPLES_PER_CLASS)
    unique_vals, counts = np.unique(y_raw, return_counts=True)
    rare_classes = unique_vals[counts < filter_threshold]
    
    if len(rare_classes) > 0:
        print(f"  Pre-filtering {len(rare_classes)} rare classes (<{filter_threshold} samples) to enable stratification...")
        mask = ~np.isin(y_raw, rare_classes)
        df = df[mask]
        y_raw = y_raw[mask]

    # Use train_test_split for stratified sampling
    try:
        df, _, y_raw, _ = train_test_split(
        df, y_raw,
        train_size=MAX_SAMPLES,
        random_state=RANDOM_SEED,
        stratify=y_raw
        )
        print(f"  Subsampled shape: {df.shape}")
    except ValueError as e:
        # Fallback if stratification fails (should be rare with above filtering)
        print(f"  Stratified sampling failed ({e}). Falling back to random sampling.")
        df, _, y_raw, _ = train_test_split(
        df, y_raw,
        train_size=MAX_SAMPLES,
        random_state=RANDOM_SEED
        )
    
    return df, y_raw

def apply_smote(X, y):
    """
    Apply SMOTE to balance the dataset
    """
    print(f"\nApplying SMOTE...")
    print(f"  Original class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    print(f"  Original class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Check minimum class size
    unique, counts = np.unique(y, return_counts=True)
    min_samples = np.min(counts)

    try:
        if min_samples < 2:
            print(f"  Warning: Found class with only {min_samples} samples. SMOTE requires at least 2.")
            print("  Switching to RandomOverSampler for this run.")
            ros = RandomOverSampler(random_state=RANDOM_SEED)
            X_res, y_res = ros.fit_resample(X, y)
        else:
            # Adjust k_neighbors if we have small classes
            # Default k is 5. We need k < n_samples.
            k_neighbors = min(5, min_samples - 1)
            if k_neighbors < 5:
                print(f"  Adjusting SMOTE k_neighbors to {k_neighbors} due to small class size.")
            
            smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k_neighbors)
            X_res, y_res = smote.fit_resample(X, y)

        print(f"  Resampled class distribution: {dict(zip(*np.unique(y_res, return_counts=True)))}")
        return X_res, y_res

    except Exception as e:
        print(f"  Oversampling failed: {e}. Returning original data.")
        return X, y

def _load_kdd99():
    """
    Load KDD99 dataset from raw numpy files
    Performs one-hot encoding and creates multi-class labels

    Returns:
        X (ndarray): Feature matrix (one-hot encoded, ~120 features)
        y (ndarray): Multi-class labels (0-12, representing 13 attack types)
    """
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

    # --- OPTIMIZATION: Subsample BEFORE preprocessing ---
    df, y_raw = _stratified_subsample(df, y_raw)
    # ----------------------------------------------------

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
    Load NetFlow dataset from raw numpy files
    Performs one-hot encoding for categorical columns (PROTOCOL, TCP_FLAGS)
    """
    # Load raw numpy files
    X_file = os.path.join(NETFLOW_PATH, 'X_raw.npy')
    y_file = os.path.join(NETFLOW_PATH, 'y_raw.npy')

    if not os.path.exists(X_file) or not os.path.exists(y_file):
        print(f"ERROR: Netflow raw files not found at {NETFLOW_PATH}")
        print("Please run: python scripts/download_netflow.py")
        return None, None

    print(f"Loading raw NetFlow data from {NETFLOW_PATH}...")
    X_raw = np.load(X_file, allow_pickle=True)
    y_raw = np.load(y_file, allow_pickle=True)

    print(f"Loaded {len(X_raw)} samples with {X_raw.shape[1]} raw features")

    # Column names (from download_netflow.py)
    # Exclude 'ANOMALY' which is the target, so we have 32 features
    columns = [
        'FLOW_ID', 'PROTOCOL_MAP', 'L4_SRC_PORT', 'IPV4_SRC_ADDR', 'L4_DST_PORT', 
        'IPV4_DST_ADDR', 'FIRST_SWITCHED', 'FLOW_DURATION_MILLISECONDS', 'LAST_SWITCHED', 
        'PROTOCOL', 'TCP_FLAGS', 'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT', 'TCP_WIN_MIN_IN', 
        'TCP_WIN_MIN_OUT', 'TCP_WIN_MSS_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_SCALE_OUT', 
        'SRC_TOS', 'DST_TOS', 'TOTAL_FLOWS_EXP', 'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN', 
        'TOTAL_PKTS_EXP', 'TOTAL_BYTES_EXP', 'IN_BYTES', 'IN_PKTS', 'OUT_BYTES', 
        'OUT_PKTS', 'ANALYSIS_TIMESTAMP', 'ALERT', 'ID'
    ]
    
    # Verify column count matches X_raw
    if X_raw.shape[1] != len(columns):
        print(f"Warning: Expected {len(columns)} columns but got {X_raw.shape[1]}. Using generic names.")
        columns = [f"feat_{i}" for i in range(X_raw.shape[1])]

    # Create DataFrame
    df = pd.DataFrame(X_raw, columns=columns)

    # Identifiers and timestamps are usually not useful for classification
    drop_cols = ['FLOW_ID', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'FIRST_SWITCHED', 'LAST_SWITCHED', 'ANALYSIS_TIMESTAMP', 'ID', 'PROTOCOL_MAP', 'ALERT']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop_cols)
    print(f"Dropped {len(existing_drop_cols)} ID/Timestamp columns.")

    # --- OPTIMIZATION: Subsample BEFORE preprocessing ---
    # This massively speeds up OHE and subsequent steps
    df, y_raw = _stratified_subsample(df, y_raw)
    # ----------------------------------------------------

    # Categorical columns for NetFlow
    # PROTOCOL and TCP_FLAGS are the primary categorical features that benefit from one-hot encoding.
    categorical_cols = ['PROTOCOL', 'TCP_FLAGS']
    
    # Ensure they exist
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # Convert to string to ensure OHE works
    for col in categorical_cols:
        df[col] = df[col].astype(str)

    # One-hot encode
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    print(f"After one-hot encoding: {df.shape[1]} features")

    # Fill NaNs if any
    df = df.fillna(0)

    # Convert to numpy
    X = df.values.astype(np.float64)

    # Process Labels (y_raw)
    # y_raw might contain NaNs
    valid_mask = ~pd.isnull(y_raw)
    if np.sum(~valid_mask) > 0:
        print(f"Dropping {np.sum(~valid_mask)} samples with missing labels.")
        X = X[valid_mask]
        y_raw = y_raw[valid_mask]

    unique_labels = np.unique(y_raw)
    
    # Filter out rare classes (less than MIN_SAMPLES_PER_CLASS)
    label_counts = {label: np.sum(y_raw == label) for label in unique_labels}
    valid_labels = sorted([label for label, count in label_counts.items() if count >= MIN_SAMPLES_PER_CLASS])
    removed_labels = sorted([label for label, count in label_counts.items() if count < MIN_SAMPLES_PER_CLASS])
    
    # Filter dataset to keep only valid classes
    if len(removed_labels) > 0:
        print(f"Filtering rare classes (MIN_SAMPLES_PER_CLASS={MIN_SAMPLES_PER_CLASS})...")
        valid_mask = np.isin(y_raw, valid_labels)
        X = X[valid_mask]
        y_raw = y_raw[valid_mask]
        print(f"  Removed {len(removed_labels)} classes with {np.sum(~valid_mask)} samples")

    print(f"Unique labels: {np.unique(y_raw)}")
    
    # Encode labels to integers
    label_dict = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_dict[l] for l in y_raw])
    
    return X, y

def _load_cores_iot():
    """
    Load CORES IoT dataset from raw numpy files
    Assumes features are numerical and last column is label (0: Normal, 1: Attack)
    """
    # Load raw numpy files
    X_file = os.path.join(CORES_IOT_PATH, 'X_raw.npy')
    y_file = os.path.join(CORES_IOT_PATH, 'y_raw.npy')

    if not os.path.exists(X_file) or not os.path.exists(y_file):
        print(f"ERROR: CoReS IoT raw files not found at {CORES_IOT_PATH}")
        print("Please run: python scripts/load_cores_iot.py")
        return None, None

    print(f"Loading raw CoReS IoT data from {CORES_IOT_PATH}...")
    X_raw = np.load(X_file, allow_pickle=True)
    y_raw = np.load(y_file, allow_pickle=True)

    print(f"Loaded {len(X_raw)} samples with {X_raw.shape[1]} raw features")

    # Create DataFrame for consistency and easy processing
    # We don't have specific column names, so we'll generate generic ones
    columns = [f"feat_{i}" for i in range(X_raw.shape[1])]
    df = pd.DataFrame(X_raw, columns=columns)

    # --- OPTIMIZATION: Subsample BEFORE preprocessing ---
    df, y_raw = _stratified_subsample(df, y_raw)
    # ----------------------------------------------------

    # CoReS IoT data is expected to be numerical.
    # Convert all columns to numeric, coercing errors to NaN
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaNs with 0
    if df.isnull().values.any():
        print(f"Filled {df.isnull().sum().sum()} missing values with 0")
        df = df.fillna(0)

    # Convert to numpy array
    X = df.values.astype(np.float64)

    # Process Labels
    # Ensure labels are integers (0 or 1)
    y_series = pd.Series(y_raw)
    y_numeric = pd.to_numeric(y_series, errors='coerce')
    
    # Check for invalid labels
    if y_numeric.isnull().any():
        print(f"Warning: Found {y_numeric.isnull().sum()} invalid labels. Dropping those samples.")
        valid_mask = ~y_numeric.isnull()
        X = X[valid_mask]
        y_numeric = y_numeric[valid_mask]

    y = y_numeric.astype(int).values
    
    unique_labels = np.unique(y)

    # Filter out rare classes (less than MIN_SAMPLES_PER_CLASS)
    label_counts = {label: np.sum(y == label) for label in unique_labels}
    valid_labels = sorted([label for label, count in label_counts.items() if count >= MIN_SAMPLES_PER_CLASS])
    removed_labels = sorted([label for label, count in label_counts.items() if count < MIN_SAMPLES_PER_CLASS])
    
    # Filter dataset to keep only valid classes
    if len(removed_labels) > 0:
        print(f"Filtering rare classes (MIN_SAMPLES_PER_CLASS={MIN_SAMPLES_PER_CLASS})...")
        valid_mask = np.isin(y, valid_labels)
        X = X[valid_mask]
        y = y[valid_mask]
        print(f"  Removed {len(removed_labels)} classes with {np.sum(~valid_mask)} samples")
        
    print(f"Unique labels: {np.unique(y)}")
    print(f"Class distribution: {np.bincount(y)}")

    return X, y

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

    # 1.5. Apply SMOTE (only on training data)
    if USE_SMOTE:
        print(f"[1.5/3] Applying SMOTE to training data...")
        X_train, y_train = apply_smote(X_train, y_train)
    else:
        print(f"[1.5/3] SMOTE disabled. Skipping balancing.")

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
