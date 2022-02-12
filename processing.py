from datetime import datetime

from pair_history import PairHistory


def movingAverage(data: list[(datetime, float)], points: int) -> list[(datetime, float)]:
    """
    Calculates the moving average of a list of (timestamp, price) tuples.
    """
    if len(data) < points:
        raise ValueError(
            f"Not enough data points to calculate moving average of {points} points.")
    return [(data[i][0], sum(price for _, price in data[i:i + points]) / points) for i in range(len(data) - points + 1)]


def pairHistoryMovingAverage(pairHistory: PairHistory, points: int):
    """
    Calculates the moving average of a pair history.
    """

    historyAsTupleList = [(item.timestamp, item.price)
                          for item in pairHistory.history]

    history = movingAverage(historyAsTupleList, points)

    return PairHistory(f'ma({points}) of {pairHistory.pair}', history)
