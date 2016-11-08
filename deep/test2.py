import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.utils import column_or_1d
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import SGD
from sklearn import svm
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix

import util

try:
    import cPickle as pickle
except:
    import pickle

seed = 7
np.random.seed(seed)
# path to the model weights file.
weights_path = '../dataset/vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 224, 224
nb_classes = 10

nb_train_samples = 1700
nb_validation_samples = 300
nb_epoch = 40


def get_layer_weights(weights_file=None, layer_name=None):
    if not weights_file or not layer_name:
        return None
    else:
        g = weights_file[layer_name]
        weights = [g[p] for p in g]
        print 'Weights for "{}" are loaded'.format(layer_name)
        return weights

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

def save_bottlebeck_features(X_train, X_test, y_train, y_test):
    weights_path='../dataset/vgg16_weights.h5'
    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

    bottleneck_features_train = model.predict(X_train, batch_size=32)
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)


    bottleneck_features_validation = model.predict(X_test, batch_size=32)
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.jet):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ['AisKacang' , 'AngKuKueh' , 'ApamBalik' , 'Asamlaksa' , 'Bahulu' , 'Bakkukteh',
    'BananaLeafRice' , 'Bazhang' , 'BeefRendang' , 'BingkaUbi'], rotation=45)
    plt.yticks(tick_marks, ['AisKacang' , 'AngKuKueh' , 'ApamBalik' , 'Asamlaksa' , 'Bahulu' , 'Bakkukteh',
    'BananaLeafRice' , 'Bazhang' , 'BeefRendang' , 'BingkaUbi'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def train_top_model(y_train, y_test):
    train_data = np.load(open('bottleneck_features_train.npy' , 'rb'))
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))


    print train_data.shape
    print validation_data.shape
    print "Training CNN.."

    input = None
    shape=train_data.shape[1:]
    output = None
    W_regularizer=True
    weights_file_path = None
    weights_file = None

    model = Sequential()
    model.add(Flatten(name='flatten', input_shape=shape))
    W_regularizer = l2(1e-2)
    model.add(Dense(4096, activation='relu',W_regularizer=W_regularizer))
    model.add(Dropout(0.6))
    W_regularizer = l2(1e-2)
    model.add(Dense(4096, activation='relu',W_regularizer=W_regularizer))
    model.add(Dense(nb_classes,activation='softmax'))

    rms = RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, y_train,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, y_test))

    y_pred = model.predict_classes(validation_data)
    print(y_pred)

    p=model.predict_proba(validation_data) # to predict probability

    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cm)

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

    #save_bottlebeck_features(X_train, X_test, y_train, y_test)
    train_top_model(y_train, y_test)
