"""Data loading and the single train/test split used everywhere."""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, RANDOM_STATE, TARGET, TEST_SIZE


def load_data(path=DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=TARGET)
    return X, y


def split_data(X, y):
    """Stratified hold-out. The test set is touched ONCE, at the very end."""
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
