import os
import numpy as np
import pandas as pd
import pickle
import glob

def load_cores_iot_raw():
    print("CoReS IoT loading...")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    output_dir = os.path.join(project_root, 'data', 'raw', 'cores_iot')
    if not os.path.exists(output_dir):
        output_dir = os.path.join(project_root, 'data', 'raw', 'cores-iot')

    # Find CSV files in the directory
    csv_files = glob.glob(os.path.join(output_dir, '*.csv'))
            
    if not csv_files:
        print(f"No CSV file found in {output_dir}.")
        return None, None

    # Load the first CSV found
    csv_file = csv_files[0]
    print(f"Loading data from: {csv_file}")
    df = pd.read_csv(csv_file, header=None)
    
    print("Dataset loaded.")
    print(f"Shape: {df.shape}")

    print("Last column is target class column.")
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values

    np.save(f'{output_dir}/X_raw.npy', X)
    np.save(f'{output_dir}/y_raw.npy', y)

    with open(f'{output_dir}/cores_iot_raw.pkl', 'wb') as f:
        pickle.dump(df, f)

    return X, y

if __name__ == '__main__':
    X, y = load_cores_iot_raw()

    if X is not None:
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
