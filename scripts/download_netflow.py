import os
import numpy as np
import pandas as pd
import pickle
import kagglehub
from kagglehub import KaggleDatasetAdapter

def download_netflow_raw():
    print("NetFlow v9 download (Kaggle)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    output_dir = os.path.join(project_root, 'data', 'raw', 'netflow')
    os.makedirs(output_dir, exist_ok=True)

    # Download latest version using dataset_download to get the path
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("ashtcoder/network-data-schema-in-the-netflow-v9-format")
    print(f"Dataset downloaded to: {path}")

    # Find the CSV file in the directory
    csv_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_file = os.path.join(root, file)
                break
        if csv_file:
            break
            
    if not csv_file:
        raise FileNotFoundError("No CSV file found in the downloaded dataset.")

    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print("Dataset downloaded.")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())

    # Attempt to separate X and y. 
    # Common labels in network datasets: 'Label', 'class', 'attack', 'attack_type'
    # I will look for these. Use case-insensitive search.
    target_col = None
    possible_labels = ['label', 'class', 'attack', 'attack_type', 'target', 'anomaly', 'alert']
    
    for col in df.columns:
        if col.lower() in possible_labels:
            target_col = col
            break
            
    if target_col:
        print(f"Found target column: {target_col}")
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
    else:
        print("No obvious target column found. Saving all as X, y as empty.")
        X = df.values
        y = np.array([])

    np.save(f'{output_dir}/X_raw.npy', X)
    np.save(f'{output_dir}/y_raw.npy', y)

    # Save the dataframe as pickle for convenience (similar to kdd99_raw.pkl)
    with open(f'{output_dir}/netflow_raw.pkl', 'wb') as f:
        pickle.dump(df, f)

    return X, y

if __name__ == '__main__':
    X, y = download_netflow_raw()

    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")