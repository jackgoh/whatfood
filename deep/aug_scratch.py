from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedKFold
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

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

    model = None
    model = util.load_alexnet_model(weights_path=None, nb_class=config.nb_class)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    print "Training from scratch CNN.."

    datagen = ImageDataGenerator(rotation_range=40,
                                 width_shift_range=0.3,
                                 height_shift_range=0.3,
                                 horizontal_flip=True,
                                 zoom_range = 0.25,
                                 shear_range = 0.25,
                                 fill_mode='nearest')
    datagen.fit(X_train)
    hist = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32), nb_epoch=400,
                        samples_per_epoch=X_train.shape[0], validation_data = (X_test,y_test))


    util.save_history(hist,"aug_alex_scratch_fold"+ str(fold_count),fold_count)

    #scores = model.evaluate(X_test, Y_test, verbose=0)
    #print("Softmax %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.save_weights("models/aug_alex_scratch_weights"+ str(fold_count) +".h5")

    # Clear memory
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None

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

        tune(data[train], data[test],labels[train], labels[test])
        #total_scores = total_scores + scores
        #fold_count = fold_count + 1
    #print("Average acc : %.2f%%" % (total_scores/config.n_folds*100))
