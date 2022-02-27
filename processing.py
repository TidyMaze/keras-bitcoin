from datetime import datetime

from pair_history import PairHistory
from price_item import PriceItem


def moving_average(data: list[tuple[datetime, float]], points: int) -> list[tuple[datetime, float]]:
    """
    Calculates the moving average of a list of (timestamp, price) tuples.
    """
    if len(data) < points:
        raise ValueError(
            f"Not enough data points to calculate moving average of {points} points.")
    return [(data[i][0], sum(price for _, price in data[i:i + points]) / points) for i in range(len(data) - points + 1)]


def pair_history_moving_average(pair_history: PairHistory, points: int) -> PairHistory:
    """
    Calculates the moving average of a pair history.
    """

    history_as_tuple_list = [(item.timestamp, item.price)
                          for item in pair_history.history]

    average_as_tuple_list = moving_average(history_as_tuple_list, points)

    average_history = [
        PriceItem(
            timestamp=timestamp,
            price=price
        ) for (timestamp, price) in average_as_tuple_list
    ]

    return PairHistory(f'ma({points}) of {pair_history.pair}', average_history)


def pair_history_moving_average_range(pair_history: PairHistory, start: int, end: int, step: int) -> list[PairHistory]:
    return [pair_history_moving_average(pair_history, points) for points in range(start, end, step)]

# given a list of price item, compute each point with t-1, t-2, t-3, t-5, t-8, t-13


def get_offset_list(data, indices, reference):
    res = [data[i] for i in [reference - i for i in indices]]
    res.reverse()
    return res + [data[reference]]


def compute_with_fibo_history(data: list[PriceItem], points: int) -> list[list[PriceItem]]:
    indices = fibonacci_sequence(points)[2:]

    print(indices)
    # print(len(data))

    return [
        get_offset_list(data, indices, reference)
        for reference in range(max(indices), len(data))
    ]


def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_sequence(n):
    return [fibonacci(i) for i in range(n)]
