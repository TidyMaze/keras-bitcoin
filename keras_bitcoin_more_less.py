import time

from keras import regularizers
from keras.callbacks import History
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from keras.layers.core import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers import Normalization, Dropout
from keras.callbacks import EarlyStopping
from pair_history import PairHistory
from price_item import PriceItem
from visualize import plot_multi_pair_history
import numpy as np
import matplotlib.pyplot as plt

from training_data_loader import load_training_data_for_regression, load_training_data_for_classification


def run():
    train_data, dates = load_training_data_for_classification()

    # shiuffle training data
    np.random.shuffle(train_data)

    x = train_data[:, :-1]

    # last column compared to previous in numpy array
    last_column = train_data[:, -1]
    last_column2 = train_data[:, -2]
    pre_y = np.greater_equal(last_column, last_column2)

    callback = EarlyStopping(monitor='val_accuracy', patience=50, verbose=1, restore_best_weights=True)

    encoder = LabelEncoder()
    encoder.fit(pre_y)
    y = encoder.transform(pre_y)

    print({'train_data': train_data, 'last': last_column, 'last-1': last_column2, 'x': x, 'y': y})

    # time.sleep(5)

    rows = y.tolist()

    assert len(rows) == len(dates)

    normalization_layer = Normalization(axis=-1)
    normalization_layer.adapt(x)

    print(normalization_layer(x))

    activation_fn = tf.keras.layers.LeakyReLU(alpha=0.3)

    dropout = 0.001
    l2 = 0.0001
    reg = regularizers.l2(l2)
    # reg = None

    hidden_layer_size = 16

    model = Sequential()
    model.add(tf.keras.Input(shape=(18,), ))
    model.add(normalization_layer)
    model.add(Dropout(dropout))
    model.add(Dense(hidden_layer_size, activation=activation_fn, kernel_regularizer=reg))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_layer_size, activation=activation_fn, kernel_regularizer=reg))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    # sgd = SGD(learning_rate=0.01, clipnorm=1.0)
    adam = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    print(np.any(np.isnan(x)))
    print(np.any(np.isnan(y)))

    history: History = model.fit(
        x,
        y,
        epochs=1000,
        validation_split=0.2,
        verbose=2,
        batch_size=128,
        shuffle=True,
        callbacks=[callback]
    )

    print(history.history.keys())

    # plot_multi_pair_history([y_pair_history, predicted_pair_history])

    # show_train_history_loss(history)
    show_train_history_accuracy(history)


def show_train_history_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()


def show_train_history_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
    plt.show()


if __name__ == '__main__':
    run()
