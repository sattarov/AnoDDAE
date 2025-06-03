import os
import glob
import numpy as np

def load_data(data_dir, dataset_name):
    """
    Loads all ADBench .npz datasets from the ad_bench folder, sorted by size.

    Args:
        data_dir (str): Path to the directory containing the 'ad_bench' folder.

    Returns:
        datasets_dict (dict): Dictionary mapping dataset names to (X, y) tuples.
    """
    data = np.load(os.path.join(data_dir, dataset_name), allow_pickle=True)
       
    X, y = data['X'], data['y']
    # dataset_name = dataset_name.replace('.npz', '')
    return X, y

def split_data(X, y, train_setting='unsupervised', random_state=None):
    """
    Splits the dataset according to the specified training setup.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels (0: normal, 1: anomaly).
        train_setting (str): 'unsupervised' or 'semi_supervised'.
        random_state (int, optional): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test: Split datasets.
    """
    if random_state is not None:
        np.random.seed(random_state)

    if train_setting == 'unsupervised':
        X_train, X_test = X, X
        y_train, y_test = y, y
    elif train_setting == 'semi-supervised':
        anomaly_indices = np.where(y == 1)[0]
        normal_indices = np.where(y == 0)[0]

        # Shuffle normal indices
        np.random.shuffle(normal_indices)
        half_normals = len(normal_indices) // 2

        train_normal_indices = normal_indices[:half_normals]
        test_normal_indices = normal_indices[half_normals:]

        # Train: only 50% normals
        train_indices = train_normal_indices

        # Test: remaining normals + all anomalies
        test_indices = np.concatenate([test_normal_indices, anomaly_indices])

        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
    else:
        raise ValueError("train_setting must be 'unsupervised' or 'semi-supervised'")

    return X_train, X_test, y_train, y_test