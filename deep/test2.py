from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.utils import np_utils
from sklearn import svm

import json
import os.path

import h5py as h5
import numpy as np

import util
import config

try:
    import cPickle as pickle
except:
    import pickle


seed = 7
np.random.seed(seed)
# path to the model weights file.
weights_path = '../dataset/alexnet_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
img_width, img_height = 224, 224
nb_train_samples = 1500
nb_validation_samples = 500
nb_class = 10
nb_epoch = 40

def load_data():
    # load your data using this function
    f = open("../dataset/myfood10-227.pkl", 'rb')
    d = pickle.load(f)
    data = d['trainFeatures']
    labels = d['trainLabels']
    lz = d['labels']
    data = data.reshape(data.shape[0], 3, 227, 227)
    #data = data.transpose(0, 2, 3, 1)

    return data,labels,lz

def get_top_model_for_alexnet(nb_class=None, shape=None, W_regularizer=False, weights_file_path=None, input=None, output=None):
    if not output:
        inputs = Input(shape=shape)

    dense_3 = Dense(10,name='dense_3')(inputs)
    predictions = Activation("softmax",name="softmax")(dense_3)
    model = Model(input=input or inputs, output=predictions)

    return model

def load_model(nb_class, weights_path=None):

    inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    conv_5 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)



    dense_1 = Flatten(name="flatten")(conv_5)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(nb_class,name='dense_3')(dense_3)
    prediction = Activation("softmax",name="softmax")(dense_3)


    base_model = Model(input=inputs, output=prediction)

    if weights_path:
        base_model.load_weights(weights_path)

    base_model = Model(input=inputs, output=dense_2)

    return base_model



def save_bottlebeck_features(X_train, X_test, y_train, y_test):
    model = load_model(nb_class=nb_class, weights_path=weights_path)
    '''
    j = 0
    for i in X_train:
        temp = X_train[j]
        temp = temp[None, ...]

        bottleneck_features_train.append(model.predict(temp, batch_size=32)[0])
        j+1
    bottleneck_features_train = np.array(bottleneck_features_train)
    np.save(open('alex_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    j = 0
    for i in X_test:
        temp = X_train[j]
        temp = temp[None, ...]
        bottleneck_features_validation.append(model.predict(temp, batch_size=32)[0])
        j+1
    bottleneck_features_validation = np.array(bottleneck_features_validation)
    np.save(open('alex_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
    '''
    bottleneck_features_train = model.predict(X_train, batch_size=32)
    np.save(open('alex_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


    bottleneck_features_validation = model.predict(X_test, batch_size=32)
    np.save(open('alex_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
    print "deep features extracted (x,4096)"


def train_top_model(y_train, y_test):
    X_train = np.load(open('alex_bottleneck_features_train.npy' , 'rb'))
    X_test = np.load(open('alex_bottleneck_features_validation.npy', 'rb'))

    #svm_train_data = X_train.reshape(nb_train_samples,9216)
    #svm_test_data = X_test.reshape(nb_validation_samples,9216)

    print "Training SVM.."
    clf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)

    clf.fit(X_train, y_train.ravel())
    #y_pred = clf.predict(test_data)
    score = clf.score(X_test, y_test.ravel())
    print("%s: %.2f%%" % ("acc: ", score*100))


    print "Training CNN.."
    y_train = np_utils.to_categorical(y_train, nb_class)
    y_test = np_utils.to_categorical(y_test, nb_class)

    shape=X_train.shape[1:]

    model = get_top_model_for_alexnet(
        shape=shape,
        nb_class=nb_class)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    model.fit(X_train, y_train,
              nb_epoch=100, batch_size=32,verbose=1)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    y_proba = model.predict(X_test)
    y_pred = np_utils.probas_to_classes(y_proba)

    target_names = ['class 0(BIKES)', 'class 1(CARS)', 'class 2(HORSES)', 'class 2(HORSES)', 'class 2(HORSES)', 'class 2(HORSES)', 'class 2(HORSES)', 'class 2(HORSES)', 'class 2(HORSES)', 'class 2(HORSES)']
    print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
    print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

    model.save_weights(top_model_weights_path)

if __name__ == "__main__":
    print "Loading data.."
    data, labels, lz = load_data()
    data = data.astype('float32')
    data /= 255
    lz = np.array(lz)
    print "Data loaded !"

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
    print X_train.shape
    print X_test.shape
    print "Test train splitted !"

    save_bottlebeck_features(X_train, X_test, y_train, y_test)
    train_top_model(y_train, y_test)
