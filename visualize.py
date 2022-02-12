import matplotlib.pyplot as plt


def plotPairHistory(pairHistory):
    xAxis = list(map(lambda item: item.timestamp, pairHistory.history))
    yAxis = list(map(lambda item: item.price, pairHistory.history))

    plt.plot(xAxis, yAxis)
    plt.title(f'price evolution of {pairHistory.pair}')
    plt.xlabel('date')
    plt.ylabel('price')
    plt.show()
