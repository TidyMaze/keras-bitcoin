from fetch_prices import load
from processing import computeWithFiboHistory, fibanacciSequence, movingAverage, pairHistoryMovingAverage, pairHistoryMovingAverageRange
from visualize import plotMultiPairHistory, plotPairHistory

pairHistory = load()

# plotMultiPairHistory([pairHistory])

lines = pairHistoryMovingAverageRange(pairHistory, start=2, end=10, step=3)

seq = computeWithFiboHistory(pairHistory.history, 6)

for el in seq:
    print(el)
