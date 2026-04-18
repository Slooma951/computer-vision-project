'''Trains a simple deep NN on the MNIST dataset.
Baseline model used to verify TensorFlow/Keras setup before applying to pneumonia dataset.
Includes additional evaluation metrics (precision, recall, F1-score) as required.
'''

from __future__ import print_function

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.metrics import classification_report
import numpy as np


batch_size = 512
num_classes = 10
epochs = 10

with tf.device('/cpu:0'):
    # load dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape and normalise
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # one-hot encoding
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    # model
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # train
    history = model.fit(x_train, y_train_cat,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test_cat))

    # evaluate
    score = model.evaluate(x_test, y_test_cat, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # predictions for metrics
    predictions = model.predict(x_test)
    y_pred = np.argmax(predictions, axis=1)

    # classification report (IMPORTANT for assignment)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))