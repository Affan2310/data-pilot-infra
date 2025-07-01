import pandas as pd
import numpy as np

def generate_dummy_classification_data(n_samples=100, n_features=5, n_classes=3):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice(range(n_classes), n_samples)  # numeric labels
    columns = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y
    return df

def generate_dummy_regression_data(n_samples=100, n_features=5):
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = X @ np.array([2, -3, 0.5, 4, -1.5][:n_features]) + np.random.randn(n_samples) * 0.5
    columns = [f"feature_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y
    return df
