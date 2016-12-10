from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.models import Model
from keras import regularizers
from KerasLayers.Custom_layers import LRN2D
from keras.optimizers import SGD,RMSprop

from sklearn.cross_validation import StratifiedKFold
from keras.utils import np_utils

import hickle as hkl
import numpy as np

# global constants
NB_CLASS = 10       # number of classes
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
ALPHA = 0.0001
BETA = 0.75
GAMMA = 0.1
DROPOUT = 0.5
WEIGHT_DECAY = 0.0005
LRN2D_norm = True       # whether to use batch normalization
# Theano - 'th' (channels, width, height)
# Tensorflow - 'tf' (width, height, channels)
DIM_ORDERING = 'th'


def conv2D_lrn2d(x, nb_filter, nb_row, nb_col,
                 border_mode='same', subsample=(1, 1),
                 activation='relu', LRN2D_norm=True,
                 weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORDERING):
    '''
        Info:
            Function taken from the Inceptionv3.py script keras github
            Utility function to apply to a tensor a module Convolution + lrn2d
            with optional weight decay (L2 weight regularization).
    '''
    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      bias=False,
                      dim_ordering=dim_ordering)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    if LRN2D_norm:

        x = LRN2D(alpha=ALPHA, beta=BETA)(x)
        x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    return x


def create_model(weights_path=None):
    # Define image input layer
    if DIM_ORDERING == 'th':
        INP_SHAPE = (3, 224, 224)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 1
    elif DIM_ORDERING == 'tf':
        INP_SHAPE = (224, 224, 3)  # 3 - Number of RGB Colours
        img_input = Input(shape=INP_SHAPE)
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORDERING))

    # Channel 1 - Convolution Net Layer 1
    x = conv2D_lrn2d(
        img_input, 3, 11, 11, subsample=(
            1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            4, 4), pool_size=(
                4, 4), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 2
    x = conv2D_lrn2d(x, 48, 55, 55, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 3
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 4
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Convolution Net Layer 5
    x = conv2D_lrn2d(x, 192, 13, 13, subsample=(1, 1), border_mode='same')
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Cov Net Layer 6
    x = conv2D_lrn2d(x, 128, 27, 27, subsample=(1, 1), border_mode='same')
    x = MaxPooling2D(
        strides=(
            2, 2), pool_size=(
                2, 2), dim_ordering=DIM_ORDERING)(x)
    x = ZeroPadding2D(padding=(1, 1), dim_ordering=DIM_ORDERING)(x)

    # Channel 1 - Cov Net Layer 7
    x = Flatten()(x)
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Channel 1 - Cov Net Layer 8
    x = Dense(2048, activation='relu')(x)
    x = Dropout(DROPOUT)(x)

    # Final Channel - Cov Net 9
    prediction = Dense(output_dim=NB_CLASS,
              activation='softmax')(x)

    if weights_path:
        x.load_weights(weights_path)

    model = Model(input=img_input,
                  output=prediction)

    return model


def check_print():
    # Create the Model
    x, img_input, CONCAT_AXIS, INP_SHAPE, DIM_ORDERING = create_model()

    # Create a Keras Model - Functional API
    model = Model(input=img_input,
                  output=[x])
    model.summary()

    # Save a PNG of the Model Build

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy')
    print "Check print"

def load_data():
    # load your data using this function
    d = hkl.load('../dataset/myfood4-224.hkl')
    data = d['trainFeatures']
    labels = d['trainLabels']
    lz = d['labels']
    data = data.reshape(data.shape[0], 3, 224, 224)
    #data = data.transpose(0, 2, 3, 1)

    return data,labels,lz

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    opt = SGD(lr=0.0001)
    model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
    #rms = RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=0.01)
    #model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])

    Y_train = np_utils.to_categorical(y_train, NB_CLASS)
    Y_test = np_utils.to_categorical(y_test, NB_CLASS)

    print "Training CNN.."
    hist = model.fit(X_train, Y_train, nb_epoch=250, validation_data=(X_test, Y_test),batch_size=32,verbose=1)
    #visualize_loss(hist)
    model.save_weights("food10x_scratch_weights.h5")

    scores = model.evaluate(X_test, Y_test, verbose=1)
    print("Softmax %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return scores


if __name__ == "__main__":
    n_folds = 2
    cvscores = []
    print "Loading data.."
    data, labels, lz = load_data()
    print "Data loaded !"
    data = data.astype('float32')
    data /= 255
    lz = np.array(lz)
    total_scores = 0


    skf = StratifiedKFold(y=lz, n_folds=n_folds, shuffle=False)

    for i, (train, test) in enumerate(skf):
        print ("Running Fold %d / %d" % (i+1, n_folds))
        model = None # Clearing the NN.
        #model = create_model(weights_path=None, heatmap=False)
        model = create_model(weights_path=None)
        scores = train_and_evaluate_model(model, data[train], labels[train], data[test], labels[test])
        #print "Score for fold ", i+1, "= ", scores*100
        total_scores = total_scores + scores

    print("Average acc : %.2f%%" % (total_scores/n_folds*100))
