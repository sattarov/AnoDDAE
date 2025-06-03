"""
Utility functions for anomaly detection tasks.
This module provides functions for evaluating anomaly detection performance,
setting random seeds for reproducibility, getting the device for PyTorch operations,
and normalizing data.
""" 
import random
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler


def evaluate_anomaly_detection(scores, labels):
    """
    Evaluate anomaly detection performance using average precision (AP) and ROC-AUC.

    Args:
        scores (numpy.ndarray): Anomaly scores for the data points.
        labels (numpy.ndarray): True labels for the data points (0 for normal, 1 for anomalies).

    Returns:
        dict: Dictionary containing AP and ROC-AUC scores.
    """
    ap = average_precision_score(labels, scores)
    roc_auc = roc_auc_score(labels, scores)
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    return {"AP": ap, "ROC-AUC": roc_auc, "PR-AUC": pr_auc}

def set_seed(seed):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """
    Get the device to use for PyTorch operations.

    Returns:
        torch.device: The device (CPU or GPU) to use.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_data(X):
    """
    Normalize the data to have zero mean and unit variance.
    Args:
        X (numpy.ndarray): Input data to normalize.
    Returns:
        numpy.ndarray: Normalized data.
    """
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized

def get_batch_size(dataset_size):
    """
    Get the effective batch size based on the dataset size and specified batch size.

    Args:
        batch_size (int): The specified batch size.
        dataset_size (int): The total number of samples in the dataset.

    Returns:
        int: The effective batch size.
    """ 
    # update the batch size to the closest power of 2
    batch_sizes = [2**i for i in range(3, 14)]
    batch_size = min(batch_sizes, key=lambda x: abs(x - dataset_size//10))
    return batch_size