from matplotlib.pyplot import axis
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from keras.layers import Normalization
from keras.layers import LeakyReLU
import numpy as np

X = np.random.randint(0, 100, (100, 2))

# build a new y np array with each element being the product of each item squared plus 10
y = np.array([(x[0] + x[1]) for x in X])

# print(X)
print(y)

normalization_layer = Normalization(axis=None)
normalization_layer.adapt(X)

print(normalization_layer(X))

model = Sequential()
model.add(tf.keras.Input(shape=(2,)))
model.add(normalization_layer)
model.add(Dense(16))
model.add(Activation('LeakyReLU'))
model.add(Dense(1))
model.add(Activation('linear'))

sgd = SGD(learning_rate=0.01)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(X, y, batch_size=10, epochs=1000)

predicted = model.predict(X)

for l, i in X:
    print(X[i], predicted[i], y[i])

print(model.predict([123, 456]))
