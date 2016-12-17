from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedKFold
from keras.utils import np_utils

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm

import h5py as h5
import numpy as np
import time

import config
import util

fold_count = 1

def tune(X_train, X_test, y_train, y_test):
    Y_train = np_utils.to_categorical(y_train, config.nb_class)
    Y_test = np_utils.to_categorical(y_test, config.nb_class)

    model = util.load_alexnet_model(weights_path=config.alexnet_weights_path, nb_class=config.nb_class)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=SGD(lr=1e-6, momentum=0.9),
        metrics=['accuracy'])

    print "Fine-tuning CNN.."


    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Softmax %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    X_train = None
    Y_train = None
    svm_train = None
    svm_test = None

    return scores[1]


if __name__ == "__main__":
    total_scores = 0

    print "Loading data.."
    data, labels, lz = util.load_data()
    data = data.astype('float32')
    data /= 255
    lz = np.array(lz)
    print lz.shape
    print "Data loaded !"

    skf = StratifiedKFold(y=lz, n_folds=config.n_folds, shuffle=False)

    for i, (train, test) in enumerate(skf):
        print "Test train Shape: "
        print data[train].shape
        print data[test].shape
        print ("Running Fold %d / %d" % (i+1, config.n_folds))

        scores = tune(data[train], data[test],labels[train], labels[test])
        total_scores = total_scores + scores
        fold_count = fold_count + 1
    print("Average acc : %.2f%%" % (total_scores/config.n_folds*100))
