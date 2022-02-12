from fetch_prices import load
from processing import movingAverage, pairHistoryMovingAverage
from visualize import plotPairHistory

pairHistory = load()

# print(data)

plotPairHistory(pairHistory)

btcUsdMovingAverage = pairHistoryMovingAverage(pairHistory, 10)

plotPairHistory(btcUsdMovingAverage)
