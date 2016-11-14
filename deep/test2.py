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
nb_train_samples = 1500
nb_validation_samples = 500
nb_class = 10
top_model_weights_path = "bottleneck_fc_model.h5"

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
    f = open("../dataset/myfood10-227.pkl", 'rb')
    d = pickle.load(f)
    data = d['trainFeatures']
    labels = d['trainLabels']
    lz = d['labels']
    data = data.reshape(data.shape[0], 3, 227, 227)
    #data = data.transpose(0, 2, 3, 1)

    return data,labels,lz

''''
def get_top_model_for_alexnet(nb_class=None, shape=None, W_regularizer=False, weights_file_path=None, input=None, output=None):
    if not output:
        inputs = Input(shape=shape)
        x = Flatten(name='flatten')(inputs)
    else:
        x = Flatten(name='flatten', input_shape=shape)(output)

    if weights_file_path:
        weights_file = h5.File(top_model_weights_path)

    weights_1 = get_layer_weights(weights_file, 'dense_1')
    dense_1 = Dense(4096, activation='relu',name='dense_1',weights=weights_1)(x)
    dense_2 = Dropout(0.5)(dense_1)


    weights_2 = get_layer_weights(weights_file, 'dense_2')
    dense_2 = Dense(4096, activation='relu',name='dense_2',weights=weights_2)(dense_2)
    dense_3 = Dropout(0.5)(dense_2)

    weights_3 = get_layer_weights(weights_file, 'softmax')
    dense_3 = Dense(10,name='dense_3',weights=weights_3)(dense_3)
    predictions = Activation("softmax",name="softmax")(dense_3)
    model = Model(input=input or inputs, output=predictions)


    if weights_file:
        weights_file.close()

    return model
'''

def get_top_model_for_alexnet(nb_class=None, shape=None, W_regularizer=False, weights_file_path=None, input=None, output=None):

    if not output:
        inputs = Input(shape=shape)
        x = Flatten(name='flatten')(inputs)
    else:
        x = Flatten(name='flatten', input_shape=shape)(output)


    dense_1 = Dense(4096, activation='relu',name='dense_1')(x)
    dense_2 = Dropout(0.5)(dense_1)


    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)


    dense_3 = Dense(10,name='dense_3')(dense_3)
    predictions = Activation("softmax",name="softmax")(dense_3)
    model = Model(input=input, output=predictions)

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


    model = Model(input=inputs, output=conv_5)

    base_model = Model(input=inputs, output=conv_5)

    for layer in base_model.layers:
        layer.trainable = False


    model = get_top_model_for_alexnet(
        shape=base_model.output_shape[1:],
        nb_class=nb_class,
        #weights_file_path="bottleneck_fc_model.h5",
        input=base_model.input,
        output=base_model.output)

    return model

def tune(X_train, X_test, y_train, y_test):
    y_train = np_utils.to_categorical(y_train, nb_class)
    y_test = np_utils.to_categorical(y_test, nb_class)

    model = load_model(nb_class=nb_class, weights_path="../dataset/alexnet_weights.h5")

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    print "Fine-tuning CNN.."

    model.fit(X_train, y_train,
              nb_epoch=100, batch_size=32,verbose=0)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    y_proba = model.predict(X_test)
    y_pred = np_utils.probas_to_classes(y_proba)

    target_names = ['AisKacang' , 'AngKuKueh' , 'ApamBalik' , 'Asamlaksa' , 'Bahulu' , 'Bakkukteh',
         'BananaLeafRice' , 'Bazhang' , 'BeefRendang' , 'BingkaUbi' , 'Buburchacha',
         'Buburpedas' , 'Capati' , 'Cendol' , 'ChaiTowKuay' , 'CharKuehTiao' , 'CharSiu',
         'CheeCheongFun' , 'ChiliCrab' , 'Chweekueh' , 'ClayPotRice' , 'CucurUdang',
         'CurryLaksa' , 'CurryPuff' , 'Dodol' , 'Durian' , 'DurianCrepe' , 'FishHeadCurry',
         'Guava' , 'HainaneseChickenRice' , 'HokkienMee' , 'Huatkuih' , 'IkanBakar',
         'Kangkung' , 'KayaToast' , 'Keklapis' , 'Ketupat' , 'KuihDadar' , 'KuihLapis',
         'KuihSeriMuka' , 'Langsat' , 'Lekor' , 'Lemang' , 'LepatPisang' , 'LorMee',
         'Maggi goreng' , 'Mangosteen' , 'MeeGoreng' , 'MeeHoonKueh' , 'MeeHoonSoup',
         'MeeJawa' , 'MeeRebus' , 'MeeRojak' , 'MeeSiam' , 'Murtabak' , 'Murukku',
         'NasiGorengKampung' , 'NasiImpit' , 'Nasikandar' , 'Nasilemak' , 'Nasipattaya',
         'Ondehondeh' , 'Otakotak' , 'OysterOmelette' , 'PanMee' , 'PineappleTart',
         'PisangGoreng' , 'Popiah' , 'PrawnMee' , 'Prawnsambal' , 'Puri' , 'PutuMayam',
         'PutuPiring' , 'Rambutan' , 'Rojak' , 'RotiCanai' , 'RotiJala' , 'RotiJohn',
         'RotiNaan' , 'RotiTissue' , 'SambalPetai' , 'SambalUdang' , 'Satay' , 'Sataycelup',
         'SeriMuka' , 'SotoAyam' , 'TandooriChicken' , 'TangYuan' , 'TauFooFah',
         'TauhuSumbat' , 'Thosai' , 'TomYumSoup' , 'Wajik' , 'WanTanMee' , 'WaTanHo' , 'Wonton',
         'YamCake' , 'YongTauFu' , 'Youtiao' , 'Yusheng']

    print(classification_report(np.argmax(y_test,axis=1), y_pred,target_names=target_names))
    print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

    model.save_weights("alex_finetune_weights.h5")


print "Loading data.."
data, labels, lz = load_data()
data = data.astype('float32')
data /= 255
lz = np.array(lz)
print "Data loaded !"

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25)
tune(X_train, X_test, y_train, y_test)
