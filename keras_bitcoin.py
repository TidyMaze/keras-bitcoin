from matplotlib.pyplot import axis
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from keras.layers import Normalization
from keras.layers import LeakyReLU
import numpy as np

from training_data_loader import loadTrainingData

trainData = loadTrainingData()

# all but last column in numpy array
X = trainData[:, :-1]
# last column in numpy array
y = trainData[:, -1]

print(X)
print(y)

normalization_layer = Normalization(axis=None)
normalization_layer.adapt(X)

print(normalization_layer(X))

model = Sequential()
model.add(tf.keras.Input(shape=(4,)))
model.add(normalization_layer)
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('linear'))

sgd = SGD(learning_rate=0.01, clipnorm=1.0)
model.compile(loss='mean_squared_error', optimizer=sgd)

print(np.any(np.isnan(X)))
print(np.any(np.isnan(y)))

model.fit(X, y, batch_size=10, epochs=1000)


predicted = model.predict(X)

for i in range(len(X)):
    print(X[i], predicted[i], y[i])

# print(model.predict(np.array([[123, 456]])))
