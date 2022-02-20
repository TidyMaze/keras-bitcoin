from datetime import datetime
from typing import Tuple

from pair_history import PairHistory
from price_item import PriceItem


def movingAverage(data: list[tuple[datetime, float]], points: int) -> list[tuple[datetime, float]]:
    """
    Calculates the moving average of a list of (timestamp, price) tuples.
    """
    if len(data) < points:
        raise ValueError(
            f"Not enough data points to calculate moving average of {points} points.")
    return [(data[i][0], sum(price for _, price in data[i:i + points]) / points) for i in range(len(data) - points + 1)]


def pairHistoryMovingAverage(pairHistory: PairHistory, points: int) -> PairHistory:
    """
    Calculates the moving average of a pair history.
    """

    historyAsTupleList = [(item.timestamp, item.price)
                          for item in pairHistory.history]

    averageAsTupleList = movingAverage(historyAsTupleList, points)

    averageHistory = [
        PriceItem(
            timestamp=timestamp,
            price=price
        ) for (timestamp, price) in averageAsTupleList
    ]

    return PairHistory(f'ma({points}) of {pairHistory.pair}', averageHistory)


def pairHistoryMovingAverageRange(pairHistory: PairHistory, start: int, end: int, step: int) -> PairHistory:
    return [pairHistoryMovingAverage(pairHistory, points) for points in range(start, end, step)]

# given a list of price item, compute each point with t-1, t-2, t-3, t-5, t-8, t-13


def computeWithFiboHistory(data: list[PriceItem], points: int) -> list[PriceItem]:
    indices = fibanacciSequence(points)[2:]

    print(indices)
    print(len(data))

    indicesFromEnd = [len(data) - i for i in indices]
    print(indicesFromEnd)

    dataAtIndices = [data[i] for i in indicesFromEnd]

    return dataAtIndices


def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibanacciSequence(n):
    return [fibonacci(i) for i in range(n)]
