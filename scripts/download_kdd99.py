import os
import numpy as np
from sklearn.datasets import fetch_kddcup99
import pickle
import ssl

def download_kdd99_raw():
    print("KDD Cup 99 download (%10)")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    output_dir = os.path.join(project_root, 'data', 'raw', 'kdd99')
    os.makedirs(output_dir, exist_ok=True)

    data = fetch_kddcup99(subset='SA', percent10=True, download_if_missing=True)

    X = data.data
    y = data.target

    np.save(f'{output_dir}/X_raw.npy', X)
    np.save(f'{output_dir}/y_raw.npy', y)

    with open(f'{output_dir}/kdd99_raw.pkl', 'wb') as f:
        pickle.dump(data, f)

    return X, y

if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context

    X, y = download_kdd99_raw()

    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")