from matplotlib.pyplot import axis
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from keras.layers import Normalization
from keras.layers import LeakyReLU
import numpy as np

from training_data_loader import loadTrainingData

trainData, dates = loadTrainingData()

print(dates)

# all but first and last column in numpy array
X = trainData[:, :-1]
# last column in numpy array
y = trainData[:, -1]

print(X)
print(y)

normalization_layer = Normalization(axis=None)
normalization_layer.adapt(X)

print(normalization_layer(X))

activationFn = tf.keras.layers.LeakyReLU(alpha=0.3)

model = Sequential()
model.add(tf.keras.Input(shape=(8,)))
model.add(normalization_layer)
model.add(Dense(16))
model.add(Activation(activationFn))
model.add(Dense(16))
model.add(Activation(activationFn))
model.add(Dense(1))
model.add(Activation('linear'))

sgd = SGD(learning_rate=0.01, clipnorm=1.0)
model.compile(loss='mean_squared_error', optimizer=sgd)

print(np.any(np.isnan(X)))
print(np.any(np.isnan(y)))

es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1000, patience=200, verbose=0, mode='auto')

model.fit(X, y, batch_size=32, epochs=10000,
          validation_split=0.2, callbacks=[es])


predicted = model.predict(X)

for i in range(len(X)):
    print(X[i], predicted[i], y[i])

# print(model.predict(np.array([[123, 456]])))
