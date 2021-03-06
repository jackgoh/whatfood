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
    model = util.load_alexnet_model_finetune56(weights_path=config.alexnet_weights_path, nb_class=config.nb_class)

    model.compile(

        loss='sparse_categorical_crossentropy',
        optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    print "Fine-tuning CNN.."

    #Real-time Data Augmentation using In-Built Function of Keras
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

    #hist = model.fit(X_train, Y_train,
    #          nb_epoch=400, batch_size=32,verbose=1,
    #          validation_data=(X_test, Y_test))

    util.save_history(hist,"alex_finetune67_aug_fold"+ str(fold_count),fold_count)

    model.save_weights("models/alex_finetune67_aug_weights"+ str(fold_count) +".h5")

    #scores = model.evaluate(X_test, y_test, verbose=0)
    #print("Softmax %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Clear memory
    model= None
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None


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
