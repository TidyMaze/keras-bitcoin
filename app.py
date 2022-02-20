from fetch_prices import load
from processing import computeWithFiboHistory, fibanacciSequence, movingAverage, pairHistoryMovingAverage, pairHistoryMovingAverageRange
from visualize import plotMultiPairHistory, plotPairHistory

pairHistory = load()

# plotMultiPairHistory([pairHistory])

lines = pairHistoryMovingAverageRange(pairHistory, start=2, end=10, step=3)

seq = computeWithFiboHistory(pairHistory.history, 6)

seqWithOnlyPrices = [
    [col.price for col in rows]
    for rows in seq
]

for i in range(len(seqWithOnlyPrices)):
    print(f'{i}: {seqWithOnlyPrices[i]}')
