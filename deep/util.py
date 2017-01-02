import matplotlib

matplotlib.use('Agg')  # fixes issue if no GUI provided

import matplotlib.pyplot as plt
import numpy as np
import h5py as h5
import os
import glob

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
import hickle as hkl
import config

def load_data():
    # load your data using this function
    d = hkl.load(config.data_path)
    data = d['trainFeatures']
    labels = d['trainLabels']
    lz = d['labels']
    data = data.reshape(data.shape[0], 3, 227, 227)
    #data = data.transpose(0, 2, 3, 1)
    return data,labels,lz

def get_layer_weights(weights_file=None, layer_name=None):
    if not weights_file or not layer_name:
        return None
    else:
        g = weights_file[layer_name]
        weights = [g[p] for p in g]
        print 'Weights for "{}" are loaded'.format(layer_name)
    return weights

def save_history(history, prefix, fold_count):
    if 'acc' not in history.history:
        return

    train_loss = history.history['loss']
    val_loss=history.history['val_loss']
    train_acc=history.history['acc']
    val_acc=history.history['val_acc']

    img_path = '{}/{}-%s.jpg'.format("hist", prefix)
    # save hist to numpy
    data={'train_loss': train_loss, 'val_loss':  val_loss, 'train_acc': train_acc, 'val_acc': val_acc }
    np.save(open(prefix + "_hist" + str(fold_count) +'.npy', 'wb'), data)

    # summarize history for accuracy
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(img_path % 'accuracy')
    plt.close()

    # summarize history for loss
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.savefig(img_path % 'loss')
    plt.close()

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

def get_top_model_for_alexnet_finetune567(nb_class=None, shape=None, weights_file_path=None, input=None, output=None):
    x = ZeroPadding2D((1,1))(output)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1),W_regularizer=l2(0.0002))(
            splittensor(ratio_split=2,id_split=i)(x)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    conv_5 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)


    dense_1 = Flatten(name="flatten")(conv_5)

    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)


    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)


    dense_3 = Dense(nb_class,name='dense_3')(dense_3)
    predictions = Activation("softmax",name="softmax")(dense_3)
    model = Model(input=input, output=predictions)

    return model

def get_top_model_for_alexnet_finetune56(nb_class=None, shape=None, weights_file_path=None, input=None, output=None):
    dense_1 = Flatten(name="flatten")(output)

    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)


    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)


    dense_3 = Dense(nb_class,name='dense_3')(dense_3)
    predictions = Activation("softmax",name="softmax")(dense_3)
    model = Model(input=input, output=predictions)

    return model

def get_top_model_for_alex_finetune67(nb_class=None, shape=None, W_regularizer=False, weights_file_path=None, input=None, output=None):
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

def load_alexnet_model_finetune567(weights_path=None, nb_class=None):

    inputs = Input(shape=(3,227,227))
    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),
                            activation='relu',
                            name='conv_1',
                            W_regularizer=l2(0.0002))(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,
                      activation="relu",
                      name='conv_2_'+str(i+1),
                      W_regularizer=l2(0.0002))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")
    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3', W_regularizer=l2(0.0002))(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1),W_regularizer=l2(0.0002))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1),W_regularizer=l2(0.0002))(
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
    base_model = Model(input=inputs, output=conv_4)

    for layer in base_model.layers:
        layer.trainable = False


    model = get_top_model_for_alexnet_finetune567(
        shape=base_model.output_shape[1:],
        nb_class=nb_class,
        #weights_file_path="bottleneck_fc_model.h5",
        input=base_model.input,
        output=base_model.output)

    return model

def load_alexnet_model_finetune56(weights_path=None, nb_class=None):

    inputs = Input(shape=(3,227,227))
    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),
                            activation='relu',
                            name='conv_1',
                            W_regularizer=l2(0.0002))(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,
                      activation="relu",
                      name='conv_2_'+str(i+1),
                      W_regularizer=l2(0.0002))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")
    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3', W_regularizer=l2(0.0002))(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1),W_regularizer=l2(0.0002))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1),W_regularizer=l2(0.0002))(
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
    base_model = Model(input=inputs, output=conv_5)

    for layer in base_model.layers:
        layer.trainable = False


    model = get_top_model_for_alexnet_finetune56(
        shape=base_model.output_shape[1:],
        nb_class=nb_class,
        #weights_file_path="bottleneck_fc_model.h5",
        input=base_model.input,
        output=base_model.output)

    return model

# Model for extracting bottleneck
def load_alex_finetune56_finetune567(nb_class, weights_path=None, top_model_weight_path=None):

    inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
                           name='conv_1', W_regularizer=l2(0.0002))(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1), W_regularizer=l2(0.0002))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3', W_regularizer=l2(0.0002))(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1), W_regularizer=l2(0.0002))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1), W_regularizer=l2(0.0002))(
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

    model = get_top_model_for_alex_finetune67(
        shape=base_model.output_shape[1:],
        nb_class=nb_class,
        weights_file_path=top_model_weight_path,
        input= base_model.input,
        output= base_model.output)

    return model

# Model for extracting bottleneck
def load_deep_features_model(nb_class, weights_path=None):

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

    base_model = Model(input=inputs, output=conv_5)

    return base_model

#### SVM ####

def get_top_model_for_svm(nb_class=None, shape=None, W_regularizer=False, weights_file_path=None, input=None, output=None):
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
    model = Model(input=input or inputs, output=dense_2)


    if weights_file:
        weights_file.close()

    return model

# Model for merging bottom alexnet model weights and finetuned top model weights
def load_svm_deep_features_model(nb_class, weights_path=None, top_model_weight_path=None):

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

    base_model = Model(input=inputs, output=conv_5)

    for layer in base_model.layers:
        layer.trainable = False

    model = get_top_model_for_svm(
        shape=base_model.output_shape[1:],
        nb_class=nb_class,
        weights_file_path=top_model_weight_path,
        input= base_model.input,
        output= base_model.output)

    return model

def load_alex_model(weights_path=None, nb_class=None):

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
        
    return base_model

def load_svm_alex_model(weights_path=None, nb_class=None):

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
