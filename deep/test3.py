import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.applications.vgg16 import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import util

try:
    import cPickle as pickle
except:
    import pickle

seed = 7
np.random.seed(seed)
nb_train_samples = 1700
nb_validation_samples = 300

def load_data():
    # load your data using this function
    f = open("../dataset/myfood10-224.pkl", 'rb')
    d = pickle.load(f)
    data = d['trainFeatures']
    labels = d['trainLabels']
    lz = d['labels']
    data = data.reshape(data.shape[0], 3, 224, 224)
    #data = data.transpose(0, 2, 3, 1)

    return data,labels,lz

def save_bottleneck_features(X_train, X_test, y_train, y_test):
    model = VGG16(weights='imagenet', include_top=False)

    bottleneck_features_train = model.predict(X_train)
    np.save(open("bottleneck_features_train.npy", 'w'), bottleneck_features_train)


    bottleneck_features_validation = model.predict(X_test)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)


def train_top_model(y_train, y_test):
    train_data = np.load(open("bottleneck_features_train.npy", 'rb'))
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    print train_data.shape
    print validation_data.shape

    train_labels = y_train
    validation_labels =  y_test


    model = util.get_top_model_for_VGG16(shape=train_data.shape[1:], nb_class=10, W_regularizer=True)
    rms = RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(verbose=1, patience=20, monitor='acc')
    model_checkpoint = ModelCheckpoint("top-model-weights.h5", save_best_only=True, save_weights_only=True, monitor='acc')
    callbacks_list = [early_stopping, model_checkpoint]

    history = model.fit(
        train_data,
        train_labels,
        nb_epoch=100,
        validation_data=(validation_data, validation_labels),
        callbacks=callbacks_list)



if __name__ == "__main__":
    print "Loading data.."
    data, labels, lz = load_data()
    data = data.astype('float32')
    data /= 255
    lz = np.array(lz)
    print "Data loaded !"

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.15, random_state=seed)
    print X_train.shape
    print X_test.shape
    print "Test train splitted !"


    #save_bottleneck_features(X_train, X_test, y_train, y_test)
    train_top_model(y_train, y_test)
