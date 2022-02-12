from fetch_prices import load
from visualize import plotPairHistory

data = load()

# print(data)

plotPairHistory(data)
