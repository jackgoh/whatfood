from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
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


# Save finetuned CNN for svm classification
def save_bottleneck_svmfeatures(X_train, X_test, y_train, y_test, pretrained_weights):
    model = util.load_svm_alex_model(nb_class=config.nb_class, weights_path=pretrained_weights)

    bottleneck_features_train = model.predict(X_train)
    np.save(open('alex_finetune56_finetune567_svmfeatures_train'+ str(fold_count) +'.npy', 'wb'), bottleneck_features_train)


    bottleneck_features_validation = model.predict(X_test)
    np.save(open('alex_finetune56_finetune567_svmfeatures_validation'+ str(fold_count) + '.npy', 'wb'), bottleneck_features_validation)
    print "Deep features extracted ", bottleneck_features_train.shape[1:]

# Train top model and save weithgs
def tune(X_train, X_test, y_train, y_test):

    model = util.load_alex_finetune56_finetune567(nb_class=config.nb_class, weights_path=config.alexnet_weights_path,top_model_weight_path="models/alex_finetune67_aug_weights" + str(fold_count) + ".h5")

    print "\nTraining CNN.."
    Y_train = np_utils.to_categorical(y_train, config.nb_class)
    Y_test = np_utils.to_categorical(y_test, config.nb_class)

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

    util.save_history(hist,"alex_finetune67_567_aug_fold"+ str(fold_count),fold_count)

    model.save_weights("models/alex_finetune67_567_aug_weights"+ str(fold_count) +".h5")

    #scores = model.evaluate(X_test, y_test, verbose=0)
    #print("Softmax %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    # Clear memory
    model= None
    X_train = None
    Y_train = None
    X_test = None
    Y_test = None


# SVM classification
def train_svm(y_train, y_test):
    svm_train = np.load(open('alex_finetune56_finetune567_svmfeatures_train'+ str(fold_count) +'.npy' , 'rb'))
    svm_test = np.load(open('alex_finetune56_finetune567_svmfeatures_validation'+ str(fold_count) + '.npy', 'rb'))


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
    total_scores = 0

    print "Loading data.."
    data, labels, lz = util.load_data()
    data = data.astype('float32')
    data /= 255
    lz = np.array(lz)
    print lz.shape
    print "Data loaded !"

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5)
    print "Test train Shape: "
    print X_train.shape
    print X_test.shape
    tune(X_train, X_test, y_train, y_test)
