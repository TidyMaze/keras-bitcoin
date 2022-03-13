from datetime import datetime
from typing import Tuple
import numpy as np
from fetch_prices import load
from processing import compute_with_fibo_history

from itertools import chain

def load_training_data_for_regression() -> Tuple[np.ndarray, list[datetime]]:
    pair_history = load()
    seq = compute_with_fibo_history(pair_history.history, 10)
    seq_with_only_prices = [[col.price for col in rows] for rows in seq]
    train_data = np.array(seq_with_only_prices)
    dates = [col.timestamp for col in pair_history.history][-len(train_data):]

    assert len(train_data) == len(dates)

    return train_data, dates


def load_training_data_for_classification() -> Tuple[np.ndarray, list[datetime]]:
    pair_history = load()
    rows = compute_with_fibo_history(pair_history.history, 20)
    seq_with_only_prices = [
        list(chain(*[[col.price_last, col.price_first, col.price_min, col.price_max] for col in row]))
        for row in rows
    ]
    train_data = np.array(seq_with_only_prices)
    dates = [col.timestamp for col in pair_history.history][-len(train_data):]

    assert len(train_data) == len(dates)

    return train_data, dates
