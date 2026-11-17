"""Pipeline test on a small hand-written DataFrame."""
import numpy as np
import pandas as pd

from penguin_pipeline import build_features, split_data


def make_df() -> pd.DataFrame:
    """Five hand-written penguins: four complete and one with a missing value."""
    return pd.DataFrame({
        "species": ["Adelie", "Gentoo", "Chinstrap", "Adelie", "Gentoo"],
        "island": ["Torgersen", "Biscoe", "Dream", "Dream", "Biscoe"],
        "bill_length_mm": [39.1, 46.1, 48.7, 37.8, None],  # the last penguin has no measured bill
        "bill_depth_mm": [18.7, 13.2, 17.4, 18.3, 14.1],
        "flipper_length_mm": [181.0, 211.0, 195.0, 174.0, 215.0],
        "body_mass_g": [3750.0, 4500.0, 3800.0, 3400.0, 5100.0],
        "sex": ["MALE", "FEMALE", "FEMALE", "FEMALE", "MALE"],
    })


def test_build_features_shape():
    X, y = build_features(make_df())
    # 5 rows minus the one with the missing value = 4
    # 4 measurements + 3 islands + 2 sexes = 9 columns
    assert X.shape == (4, 9)


def test_build_features_labels():
    X, y = build_features(make_df())
    # Adelie 0, Gentoo 2, Chinstrap 1, Adelie 0: the fifth penguin was dropped
    assert list(y) == [0, 2, 1, 0]


def test_split_sizes():
    X = np.arange(20).reshape(10, 2)  # 10 fake examples, 2 columns
    y = np.arange(10)                 # all distinct labels: 0, 1, ..., 9
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert X_train.shape == (8, 2)    # 80% of 10
    assert X_test.shape == (2, 2)
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_split_same_seed(): 
    X = np.arange(20).reshape(10, 2)  # 10 fake examples, 2 columns
    y = np.arange(10)
    _, _, y_train,_ = split_data(X,y, seed=7)
    _, _, y_train2,_ = split_data(X,y,seed=7)
    assert np.array_equal(y_train, y_train2)


def test_split_different_seed(): 
    X = np.arange(20).reshape(10, 2)  # 10 fake examples, 2 columns
    y = np.arange(10)
    _, _, y_train,_ = split_data(X,y, seed=7)
    _, _, y_train2,_ = split_data(X,y,seed=9)
    assert not np.array_equal(y_train, y_train2)


def test_split_no_loss(): 
    X = np.arange(20).reshape(10, 2)  # 10 fake examples, 2 columns
    y = np.arange(10)
    _, _, y_train, y_test = split_data(X,y)
    assert sorted(list(y_train) + list(y_test)) == list(range(10))

