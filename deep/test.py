from keras.optimizers import SGD
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import StratifiedKFold
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
from sklearn.metrics import classification_report,confusion_matrix
from keras.utils import np_utils
from sklearn import svm

import sys, glob, argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import json
import os.path

import h5py as h5
import numpy as np

import hickle as hkl

seed = 7
np.random.seed(seed)
# path to the model weights file.
weights_path = '../dataset/alexnet_weights.h5'
nb_class = 100
nb_epoch = 150
fold_count = 1

def get_layer_weights(weights_file=None, layer_name=None):
    if not weights_file or not layer_name:
        return None
    else:
        g = weights_file[layer_name]
        weights = [g[p] for p in g]
        print 'Weights for "{}" are loaded'.format(layer_name)
    return weights

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.jet):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(100)
    plt.xticks(tick_marks, ['AisKacang' , 'AngKuKueh' , 'ApamBalik' , 'Asamlaksa' , 'Bahulu' , 'Bakkukteh',
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
         'YamCake' , 'YongTauFu' , 'Youtiao' , 'Yusheng'], rotation=45)
    plt.yticks(tick_marks, ['AisKacang' , 'AngKuKueh' , 'ApamBalik' , 'Asamlaksa' , 'Bahulu' , 'Bakkukteh',
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
         'YamCake' , 'YongTauFu' , 'Youtiao' , 'Yusheng'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def visualize_loss(hist):
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
    plt.show()

def load_data():
    # load your data using this function
    d = hkl.load('../dataset/myfood100-227.hkl')
    data = d['trainFeatures']
    labels = d['trainLabels']
    lz = d['labels']
    data = data.reshape(data.shape[0], 3, 227, 227)
    #data = data.transpose(0, 2, 3, 1)

    return data,labels,lz

def get_top_model_for_alexnet(nb_class=None, shape=None, W_regularizer=False, weights_file_path=None, input=None, output=None):
    if not output:
        inputs = Input(shape=shape)


    dense_1 = Flatten(name="flatten")(inputs)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(nb_class,name='dense_3')(dense_3)
    predictions = Activation("softmax",name="softmax")(dense_3)
    model = Model(input=input or inputs, output=predictions)

    return model

# This function load top model pretrained weights, will be merge with alexnet bottom model weight
def get_top_model_for_alexnet2(nb_class=None, shape=None, W_regularizer=False, weights_file_path=None, input=None, output=None):
    if not output:
        inputs = Input(shape=shape)
        dense_1 = Flatten(name='flatten')(inputs)
    else:
        dense_1 = Flatten(name='flatten', input_shape=shape)(output)

    if weights_file_path:
        weights_file = h5.File(weights_file_path)

    weights_1 = get_layer_weights(weights_file, 'dense_1')
    dense_1 = Dense(4096, activation='relu',name='dense_1',weights=weights_1)(dense_1)
    dense_2 = Dropout(0.5)(dense_1)

    weights_2 = get_layer_weights(weights_file, 'dense_2')
    dense_2 = Dense(4096, activation='relu',name='dense_2',weights=weights_2)(dense_2)
    dense_3 = Dropout(0.5)(dense_2)

    weights_3 = get_layer_weights(weights_file, 'dense_3')
    dense_3 = Dense(nb_class,name='dense_3',weights=weights_3)(dense_3)
    predictions = Activation("softmax",name="softmax")(dense_3)

    model = Model(input=input or inputs, output=predictions)


    if weights_file:
        weights_file.close()

    return model

# Model for extracting bottleneck
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


    base_model.load_weights(weights_path)


    #model = Model(input=inputs, output=conv_5)

    base_model = Model(input=inputs, output=conv_5)

    for layer in base_model.layers[:17]:
        print layer
        layer.trainable = False

    model = get_top_model_for_alexnet2(
        shape=base_model.output_shape[1:],
        nb_class=nb_class,
        weights_file_path="models/alex_finetune67_weights" + str(fold_count) + ".h5",
        input= base_model.input,
        output= base_model.output)

    return model

# Model for merging bottom alexnet model weights and finetuned top model weights
def load_svm_model(nb_class, weights_path=None):

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


    #model = Model(input=inputs, output=conv_5)

    base_model = Model(input=inputs, output=dense_2)

    '''
    for layer in base_model.layers:
        layer.trainable = False

    model = get_top_model_for_svm(
        shape=base_model.output_shape[1:],
        nb_class=nb_class,
        weights_file_path="model/alex_topmodel" + str(fold_count) + ".h5",
        input= base_model.input,
        output= base_model.output)
    '''

    return base_model


# Save CNN bottleneck for finetune top model
def save_bottleneck_features(X_train, X_test, y_train, y_test):
    model = load_model(nb_class=nb_class, weights_path=weights_path)

    bottleneck_features_train = model.predict(X_train)
    np.save(open('alex_bottleneck_features_train'+ str(fold_count) +'.npy', 'wb'), bottleneck_features_train)


    bottleneck_features_validation = model.predict(X_test)
    np.save(open('alex_bottleneck_features_validation'+ str(fold_count) + '.npy', 'wb'), bottleneck_features_validation)
    print "deep features extracted", bottleneck_features_train.shape[1:]

# Save finetuned CNN for svm classification
def save_bottleneck_svmfeatures(X_train, X_test, y_train, y_test, pretrained_weights):
    model = load_svm_model(nb_class=nb_class, weights_path=pretrained_weights)

    bottleneck_features_train = model.predict(X_train)
    np.save(open('alex_doblefinetune_svmfeatures_train'+ str(fold_count) +'.npy', 'wb'), bottleneck_features_train)


    bottleneck_features_validation = model.predict(X_test)
    np.save(open('alex_doblefinetune_svmfeatures_validation'+ str(fold_count) + '.npy', 'wb'), bottleneck_features_validation)
    print "deep features extracted", bottleneck_features_train.shape[1:]

# Train top model and save weithgs
def train_top_model(X_train, X_test, y_train, y_test):

    model = load_model(nb_class=nb_class, weights_path=weights_path)

    print "\nTraining CNN.."
    y_train = np_utils.to_categorical(y_train, nb_class)
    y_test = np_utils.to_categorical(y_test, nb_class)

    shape=X_train.shape[1:]

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    hist = model.fit(X_train, y_train,
              nb_epoch=10, batch_size=32,verbose=1)

    #visualize_loss(hist) # Removing this because we not training CNN, just classification

    scores = model.evaluate(X_test, y_test, verbose=0)
    model.save_weights("models/alex_finetune56_finetune567" + str(fold_count) + ".h5")
    model = None
    #model.save_weights("model/alex_topmodel" + str(fold_count) + ".h5")
    #print("CNN %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    return scores[1]

# SVM classification
def train_svm(y_train, y_test):
    svm_train = np.load(open('alex_doblefinetune_svmfeatures_train'+ str(fold_count) +'.npy' , 'rb'))
    svm_test = np.load(open('alex_doblefinetune_svmfeatures_validation'+ str(fold_count) + '.npy', 'rb'))


    print "\nTraining SVM.."
    clf = svm.SVC(kernel='linear', gamma=0.7, C=1.0)

    clf.fit(svm_train, y_train.ravel())
    #y_pred = clf.predict(test_data)
    score = clf.score(svm_test, y_test.ravel())
    print("SVM %s: %.2f%%" % ("acc: ", score*100))

    y_pred = clf.predict(svm_test)

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
    cm = confusion_matrix(y_test, y_pred)
    print(classification_report(y_test, y_pred,target_names=target_names))
    print(cm)

    return score

if __name__ == "__main__":
    n_folds = 2
    total_scores = 0

    print "Loading data.."
    data, labels, lz = load_data()
    data = data.astype('float32')
    data /= 255
    lz = np.array(lz)
    print lz.shape
    print "Data loaded !"

    skf = StratifiedKFold(y=lz, n_folds=n_folds, shuffle=False)

    for i, (train, test) in enumerate(skf):
        print "Test train Shape: "
        print data[train].shape
        print data[test].shape
        print ("Running Fold %d / %d" % (i+1, n_folds))

        #save_bottleneck_features(data[train], data[test],labels[train], labels[test])
        scores = train_top_model(data[train], data[test],labels[train], labels[test])

        save_bottleneck_svmfeatures(data[train], data[test],labels[train], labels[test],"models/alex_finetune56_finetune567" + str(fold_count) + ".h5")
        svm_scores = train_svm(labels[train], labels[test])

        total_scores = total_scores + svm_scores
        fold_count = fold_count + 1
    print("Average acc : %.2f%%" % (total_scores/n_folds*100))
