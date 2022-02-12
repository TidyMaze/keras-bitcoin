from fetch_prices import load

import matplotlib.pyplot as plt

data = load()

print(data)

xAxis = list(map(lambda item: item.timestamp, data))
yAxis = list(map(lambda item: item.price, data))

plt.plot(xAxis, yAxis)
plt.title('price evolution')
plt.xlabel('date')
plt.ylabel('price')
plt.show()
