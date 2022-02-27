import matplotlib.pyplot as plt

from pair_history import PairHistory


def plot_pair_history(pair_history: PairHistory, block: bool = True) -> None:
    x_axis = [item.timestamp for item in pair_history.history]
    y_axis = [item.price for item in pair_history.history]
    plt.plot(x_axis, y_axis, label=pair_history.pair)


def plot_multi_pair_history(pair_histories: list[PairHistory]) -> None:
    for pairHistory in pair_histories:
        plot_pair_history(pairHistory, block=False)

    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.show(block=True)
