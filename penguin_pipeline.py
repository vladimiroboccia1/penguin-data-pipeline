"""Data pipeline that turns the Palmer Penguins CSV into model-ready NumPy arrays."""

import numpy as np
import pandas as pd

# Numeric measurements used as features
FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
# Species name -> class label
LABELS = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}


def load_data(path: str):
    """Read the penguins CSV file and return it as a pandas DataFrame."""
    return pd.read_csv(path)


def build_features(df: pd.DataFrame):
    """Build the feature matrix X and the label vector y.

    Drops rows with missing measurements or sex, one-hot encodes island and sex,
    and maps species to labels 0, 1, 2. Returns X (2D array) and y (1D array).
    """
    df_model = df.dropna(subset=FEATURES + ["sex"])
    island_onehot = pd.get_dummies(df_model["island"], prefix="island", dtype=int)
    sex_onehot = pd.get_dummies(df_model["sex"], prefix="sex", dtype=int)
    # Rows are aligned by index label, so each penguin keeps its own values
    df_features = pd.concat([df_model[FEATURES], island_onehot, sex_onehot], axis=1)
    X = df_features.to_numpy()
    species = df_model["species"].map(LABELS)
    y = species.to_numpy()
    return X, y


def split_data(X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8, seed: int = 42):
    """Randomly split X and y into train and test sets, reproducibly.

    Returns X_train, X_test, y_train, y_test, in this order.
    """
    rng = np.random.default_rng(seed)  # fixed seed: same split on every run
    indices = rng.permutation(len(X))
    n_train = int(train_ratio * len(X))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    # Same indices for X and y, so features and labels stay paired
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    """Standardize both sets with the mean and standard deviation of the training set.

    Using training statistics only avoids leaking information from the test set.
    Returns X_train_scaled, X_test_scaled, mean, std, in this order.
    """
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    X_train_scaled = (X_train - mu) / std
    X_test_scaled = (X_test - mu) / std
    return X_train_scaled, X_test_scaled, mu, std