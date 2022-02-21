from matplotlib.pyplot import axis
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from keras.layers import Normalization
from keras.layers import LeakyReLU
from pair_history import PairHistory
from price_item import PriceItem
from visualize import plotMultiPairHistory
import numpy as np
import matplotlib.pyplot as plt

from training_data_loader import loadTrainingData

trainData, dates = loadTrainingData()

print(dates)

# all but first and last column in numpy array
X = trainData[:, :-1]
# last column in numpy array
y = trainData[:, -1]

print(X)
print(y)

rows = y.tolist()

print(rows)

assert len(rows) == len(dates)

yListItems = [PriceItem(dates[i], rows[i]) for i in range(len(rows))]
yPairHistory = PairHistory('real', yListItems)

normalization_layer = Normalization(axis=-1)
normalization_layer.adapt(X)

print(normalization_layer(X))

activationFn = tf.keras.layers.LeakyReLU(alpha=0.3)
# activationFn = 'relu'

model = Sequential()
model.add(tf.keras.Input(shape=(8,)))
model.add(normalization_layer)
model.add(Dense(8))
model.add(Activation(activationFn))
model.add(Dense(4))
model.add(Activation(activationFn))
model.add(Dense(1))
model.add(Activation('linear'))

sgd = SGD(learning_rate=0.005, clipnorm=1.0)
model.compile(loss='mean_squared_error', optimizer=sgd)

print(np.any(np.isnan(X)))
print(np.any(np.isnan(y)))

history = model.fit(X, y, epochs=3000, batch_size=32,
                    validation_split=0.1, verbose=2)

print(history.history.keys())

predicted = model.predict(X).tolist()

for i in range(len(X)):
    print(
        f'at {dates[i]}: input {X[i]} => predicted {predicted[i]} (real {y[i]})')

print(len(predicted))

# print(model.predict(np.array([[123, 456]])))

print(len(dates))

predictedListItems = [PriceItem(dates[i], predicted[i])
                      for i in range(len(predicted))]

print(len(predictedListItems))

predictedPairHistory = PairHistory('predicted', predictedListItems)

# print(predictedPairHistory.history)
print(len(predictedPairHistory.history))

plotMultiPairHistory([yPairHistory, predictedPairHistory])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()
