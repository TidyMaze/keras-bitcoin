from fetch_prices import load
from processing import movingAverage, pairHistoryMovingAverage
from visualize import plotMultiPairHistory, plotPairHistory

pairHistory = load()

# print(data)


lines = [pairHistoryMovingAverage(pairHistory, points)
         for points in range(2, 11, 3)]
plotMultiPairHistory([pairHistory] + lines)
