import matplotlib.pyplot as plt

from pair_history import PairHistory


def plotPairHistory(pairHistory: PairHistory):
    xAxis = [item.timestamp for item in pairHistory.history]
    yAxis = [item.price for item in pairHistory.history]

    plt.plot(xAxis, yAxis)
    plt.title(f'price evolution of {pairHistory.pair}')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.show()
