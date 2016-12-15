from keras.optimizers import SGD
from keras.utils import np_utils

import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split


import theano
import h5py as h5
import numpy as np
import time

import config
import util


if __name__ == "__main__":
    print "Loading data.."
    data, labels, lz = util.load_data()
    data = data.astype('float32')
    data /= 255
    lz = np.array(lz)
    print lz.shape
    print "Data loaded !"

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    Y_train = np_utils.to_categorical(y_train, config.nb_class)
    Y_test = np_utils.to_categorical(y_test, config.nb_class)

    model = util.load_alexnet_model(weights_path=config.alexnet_weights_path, nb_class=config.nb_class)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=1e-6, momentum=0.9),
        metrics=['accuracy'])

    print "Fine-tuning CNN.."

    hist = model.fit(X_train, Y_train,
              nb_epoch=2, batch_size=32,verbose=1,
              validation_data=(X_test, Y_test))

    out = model.predict(X_train[0])
