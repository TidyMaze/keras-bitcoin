from keras.callbacks import History
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers import Normalization
from pair_history import PairHistory
from price_item import PriceItem
from visualize import plot_multi_pair_history
import numpy as np
import matplotlib.pyplot as plt

from training_data_loader import load_training_data_for_regression


def run():
    train_data, dates = load_training_data_for_regression()

    print(dates)

    # all but first and last column in numpy array
    x = train_data[:, :-1]
    # last column in numpy array
    y = train_data[:, -1]

    print(x)
    print(y)

    rows = y.tolist()

    print(rows)

    assert len(rows) == len(dates)

    y_list_items = [PriceItem(dates[i], rows[i]) for i in range(len(rows))]
    y_pair_history = PairHistory('real', y_list_items)

    normalization_layer = Normalization(axis=-1)
    normalization_layer.adapt(x)

    print(normalization_layer(x))

    activation_fn = tf.keras.layers.LeakyReLU(alpha=0.3)

    model = Sequential()
    model.add(tf.keras.Input(shape=(8,)))
    model.add(normalization_layer)
    model.add(Dense(32))
    model.add(Activation(activation_fn))
    model.add(Dense(32))
    model.add(Activation(activation_fn))
    model.add(Dense(1))
    model.add(Activation('linear'))

    sgd = SGD(learning_rate=0.01, clipnorm=1.0)
    adam = Adam(learning_rate=0.01, clipnorm=1.0)
    model.compile(loss='mean_squared_error', optimizer=adam)

    print(np.any(np.isnan(x)))
    print(np.any(np.isnan(y)))

    history: History = model.fit(x, y, epochs=1000,
                                 validation_split=0.1, verbose=2)

    print(history.history.keys())

    predicted = model.predict(x).tolist()

    print(len(predicted))

    print(len(dates))

    predicted_list_items = [PriceItem(dates[i], predicted[i])
                            for i in range(len(predicted))]

    print(len(predicted_list_items))

    predicted_pair_history = PairHistory('predicted', predicted_list_items)

    print(len(predicted_pair_history.history))

    plot_multi_pair_history([y_pair_history, predicted_pair_history])

    show_train_history_loss(history)


def show_train_history_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()


if __name__ == '__main__':
    run()
