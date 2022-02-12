from fetch_prices import load
from processing import movingAverage, pairHistoryMovingAverage, pairHistoryMovingAverageRange
from visualize import plotMultiPairHistory, plotPairHistory

pairHistory = load()

# print(data)


lines = pairHistoryMovingAverageRange(pairHistory, start=2, end=10, step=3)

plotMultiPairHistory([pairHistory] + lines)
