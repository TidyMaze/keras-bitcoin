from datetime import datetime
from multiprocessing.context import assert_spawning
from typing import Tuple
import numpy as np
from fetch_prices import load
from processing import computeWithFiboHistory


def loadTrainingData() -> Tuple[np.ndarray, list[datetime]]:
    pairHistory = load()
    # lines = pairHistoryMovingAverageRange(pairHistory, start=2, end=10, step=3)
    seq = computeWithFiboHistory(pairHistory.history, 10)
    seqWithOnlyPrices = [[col.price for col in rows] for rows in seq]

    # for i in range(len(seqWithOnlyPrices)):
    # print(f'{i}: {seqWithOnlyPrices[i]}')

    trainData = np.array(seqWithOnlyPrices)
    dates = [col.timestamp for col in pairHistory.history][-len(trainData):]

    assert len(trainData) == len(dates)

    return trainData, dates
