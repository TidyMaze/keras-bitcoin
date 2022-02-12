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
