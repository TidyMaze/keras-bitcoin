import matplotlib.pyplot as plt

from pair_history import PairHistory


def plotPairHistory(pairHistory: PairHistory, block: bool = True) -> None:
    xAxis = [item.timestamp for item in pairHistory.history]
    yAxis = [item.price for item in pairHistory.history]
    plt.plot(xAxis, yAxis, label=pairHistory.pair)


def plotMultiPairHistory(pairHistories: list[PairHistory]) -> None:
    for pairHistory in pairHistories:
        plotPairHistory(pairHistory, block=False)

    plt.xlabel('date')
    plt.ylabel('price')
    plt.legend()
    plt.show(block=True)
