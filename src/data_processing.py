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
    Load NetFlow dataset with robust handling for missing column names.
    Prioritizes loading 'netflow_raw.pkl' to preserve metadata.
    """
    pkl_file = os.path.join(NETFLOW_PATH, 'netflow_raw.pkl')
    X_file = os.path.join(NETFLOW_PATH, 'X_raw.npy')
    y_file = os.path.join(NETFLOW_PATH, 'y_raw.npy')

    df = None
    y_raw = None

    # 1. Try loading from Pickle (Best case: preserves column names and types)
    if os.path.exists(pkl_file):
        print(f"Loading NetFlow data from {pkl_file} (preserves metadata)...")
        try:
            df = pd.read_pickle(pkl_file)
            print(f"Loaded DataFrame with shape: {df.shape}")
            
            # Identify target column
            possible_labels = ['label', 'class', 'attack', 'attack_type', 'target', 'anomaly', 'alert']
            target_col = None
            for col in df.columns:
                if str(col).lower() in possible_labels:
                    target_col = col
                    break
            
            if target_col:
                print(f"  Found target column: {target_col}")
                y_raw = df[target_col].values
                df = df.drop(columns=[target_col])
            else:
                print("  Warning: No explicit target column found in DataFrame.")
                # Try to load y_raw separately if it exists
                if os.path.exists(y_file):
                    print(f"  Loading labels from {y_file}")
                    y_raw = np.load(y_file, allow_pickle=True)
                else:
                    print("  ERROR: No labels found. Using empty labels.")
                    y_raw = np.zeros(len(df))

        except Exception as e:
            print(f"  Failed to load pickle: {e}. Falling back to raw numpy arrays.")
            df = None

    # 2. Fallback to Raw Numpy (If pickle failed or doesn't exist)
    if df is None:
        if not os.path.exists(X_file) or not os.path.exists(y_file):
            print(f"ERROR: Netflow raw files not found at {NETFLOW_PATH}")
            print("Please run: python scripts/download_netflow.py")
            return None, None

        print(f"Loading raw NetFlow data from {NETFLOW_PATH} (numpy arrays)...")
        X_raw = np.load(X_file, allow_pickle=True)
        y_raw = np.load(y_file, allow_pickle=True)
        
        # Create DataFrame with generic names
        columns = [f"feat_{i}" for i in range(X_raw.shape[1])]
        df = pd.DataFrame(X_raw, columns=columns)
        print(f"Loaded raw array: {df.shape}")

    # --- OPTIMIZATION: Subsample BEFORE preprocessing ---
    # This massively speeds up OHE and subsequent steps
    df, y_raw = _stratified_subsample(df, y_raw)
    # ----------------------------------------------------

    # 3. Robust Preprocessing (Handles both named and generic columns)
    print("Preprocessing NetFlow features...")
    
    # Drop known ID/Timestamp columns if they exist (only for named columns)
    drop_cols_names = ['FLOW_ID', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'FIRST_SWITCHED', 
                       'LAST_SWITCHED', 'ANALYSIS_TIMESTAMP', 'ID', 'PROTOCOL_MAP', 'ALERT']
    existing_drop_cols = [c for c in df.columns if str(c) in drop_cols_names]
    
    if existing_drop_cols:
        df = df.drop(columns=existing_drop_cols)
        print(f"  Dropped {len(existing_drop_cols)} known ID/timestamp columns.")

    # Auto-detect and handle column types
    # First, try to convert everything to numeric
    # Heuristic strategy:
    # 1. Columns that are fully convertible to numbers -> Convert
    # 2. Columns that fail conversion -> Treat as Categorical
    # 3. Categorical with HIGH cardinality (>50) -> Drop (likely IDs/IPs that persisted)
    # 4. Categorical with LOW cardinality -> One-Hot Encode

    new_df = pd.DataFrame(index=df.index)
    categorical_cols = []

    for col in df.columns:
        # Attempt numeric conversion
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        
        # Check if we lost too much data (indicating it was actually a string feature)
        # If previously it wasn't null, but now it is null, it was non-numeric.
        is_numeric = True
        if df[col].dtype == object:
             # If > 50% became NaN, treat as categorical string
             if numeric_series.isna().sum() > (0.5 * len(df)):
                 is_numeric = False
        
        if is_numeric:
            new_df[col] = numeric_series.fillna(0) # Fill NaNs (missing values in numeric cols)
        else:
            # It's categorical
            unique_count = df[col].nunique()
            if unique_count > 50:
                print(f"  Dropping high-cardinality categorical column '{col}' ({unique_count} unique values).")
            else:
                print(f"  Detected categorical column '{col}' ({unique_count} unique values). Keeping for OHE.")
                categorical_cols.append(col)
                new_df[col] = df[col].astype(str)

    df = new_df

    # One-hot encode identified categorical columns
    if categorical_cols:
        print(f"  One-hot encoding {len(categorical_cols)} categorical columns...")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    
    print(f"After processing: {df.shape[1]} features")

    # Convert to numpy
    X = df.values.astype(np.float64)

    # Process Labels (y_raw)
    valid_mask = ~pd.isnull(y_raw)
    if np.sum(~valid_mask) < len(y_raw):
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
    label_dict = {label: idx for idx, label in enumerate(valid_labels)}
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

def prepare_fold_data(X_train, y_train, X_val, y_val):
    """
    Process data for a specific Cross-Validation fold.
    Ensures ZERO DATA LEAKAGE by fitting transformations ONLY on X_train.
    
    Pipeline:
    1. SMOTE (Train only)
    2. Normalize (Fit Train, Transform Val)
    3. PCA (Fit Train, Transform Val)
    """
    
    # 1. SMOTE (Balancing) - Only on Training Data
    if USE_SMOTE:
        X_train, y_train = apply_smote(X_train, y_train)

    # 2. Normalization
    if NORMALIZE:
        if NORMALIZATION_METHOD == 'minmax':
            scaler = MinMaxScaler()
        elif NORMALIZATION_METHOD == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {NORMALIZATION_METHOD}")

        # FIT only on training data
        scaler.fit(X_train)
        
        # Transform both
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
    
    # 3. PCA
    if USE_PCA:
        pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_SEED)
        
        # FIT only on training data
        pca.fit(X_train)
        
        # Transform both
        X_train = pca.transform(X_train)
        X_val = pca.transform(X_val)

    return X_train, y_train, X_val, y_val

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
