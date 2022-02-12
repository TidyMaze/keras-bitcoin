from fetch_prices import load

import matplotlib.pyplot as plt

data = load()

print(data)

xAxis = list(map(lambda item: item.timestamp, data.history))
yAxis = list(map(lambda item: item.price, data.history))

plt.plot(xAxis, yAxis)
plt.title(f'price evolution of {data.pair}')
plt.xlabel('date')
plt.ylabel('price')
plt.show()
