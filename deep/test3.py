import os
import h5py
from sklearn.cross_validation import StratifiedKFold
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D

from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
splittensor, Softmax4D

from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD,RMSprop
from keras.utils import np_utils
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

batch_size = 32
nb_classes = 10
nb_epoch = 100
data_augmentation = False

# input image dimensions
img_rows, img_cols = 227,227
img_channels = 3

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

def save_bottlebeck_features(X_train, X_test, y_train, y_test):
    datagen = ImageDataGenerator(rescale=1./255)

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

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)


    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(nb_classes,name='dense_3')(dense_3)
    prediction = Activation("softmax",name="softmax")(dense_3)


    model = Model(input=inputs, output=prediction)
    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    model.load_weights('../dataset/alexnet_weights.h5')
    print('Model loaded.')

    model.pop()
    model.pop()

    bottleneck_features_train = model.predict(X_train, batch_size=32)
    np.save(open('alex_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


    bottleneck_features_validation = model.predict(X_test, batch_size=32)
    np.save(open('alex_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)



def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    opt = SGD(lr=0.01)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
    #rms = RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=0.01)
    #model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print "Model loaded."

    model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size,verbose=1)

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    return scores

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

    save_bottlebeck_features(X_train, X_test, y_train, y_test)
    #train_top_model(y_train, y_test)
