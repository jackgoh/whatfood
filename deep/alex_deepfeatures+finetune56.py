from keras.optimizers import SGD
from sklearn.metrics import classification_report
from sklearn.cross_validation import StratifiedKFold
from keras.utils import np_utils

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import svm

import h5py as h5
import numpy as np
import time

import config
import util

fold_count = 1

# Save CNN bottleneck for finetune top model
def save_bottleneck_features(X_train, X_test, y_train, y_test):
    model = util.load_deep_features_model(nb_class=config.nb_class, weights_path=config.alexnet_weights_path)

    bottleneck_features_train = model.predict(X_train)
    np.save(open('alex_bottleneck_features_train'+ str(fold_count) +'.npy', 'wb'), bottleneck_features_train)


    bottleneck_features_validation = model.predict(X_test)
    np.save(open('alex_bottleneck_features_validation'+ str(fold_count) + '.npy', 'wb'), bottleneck_features_validation)
    print "Deep features extracted ", bottleneck_features_train.shape[1:]

# Save finetuned CNN for svm classification
def save_bottleneck_svmfeatures(X_train, X_test, y_train, y_test):
    model = util.load_svm_deep_features_model(nb_class=config.nb_class, weights_path=config.alexnet_weights_path, top_model_weight_path= "models/alex_topmodel_deepfeatures+finetune56" + str(fold_count) + ".h5")

    bottleneck_features_train = model.predict(X_train)
    np.save(open('alex_bottleneck_svmfeatures+finetune56_train'+ str(fold_count) +'.npy', 'wb'), bottleneck_features_train)


    bottleneck_features_validation = model.predict(X_test)
    np.save(open('alex_bottleneck_svmfeatures+finetune56_validation'+ str(fold_count) + '.npy', 'wb'), bottleneck_features_validation)
    print "SVM features extracted ", bottleneck_features_train.shape[1:]

# Train top model and save weithgs
def train_top_model(y_train, y_test):
    X_train = np.load(open('alex_bottleneck_features_train'+ str(fold_count) + '.npy' , 'rb'))
    X_test = np.load(open('alex_bottleneck_features_validation'+ str(fold_count) + '.npy', 'rb'))

    print "\nTraining CNN.."
    Y_train = np_utils.to_categorical(y_train, config.nb_class)
    Y_test = np_utils.to_categorical(y_test, config.nb_class)

    shape=X_train.shape[1:]

    model = None # Clear Model

    model = util.get_top_model_for_alexnet(
        shape=shape,
        nb_class=config.nb_class)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
        metrics=['accuracy'])

    hist = model.fit(X_train, Y_train,
              nb_epoch=150, batch_size=32,verbose=1,
              validation_data=(X_test, Y_test))

    util.save_history(hist,"deeepfeatures+finetune56_fold"+ str(fold_count),fold_count)

    scores = model.evaluate(X_test, Y_test, verbose=0)
    model.save_weights("models/alex_topmodel_deepfeatures+finetune56" + str(fold_count) + ".h5")
    print("Softmax %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    return scores[1]

# SVM classification
def train_svm(y_train, y_test):
    svm_train = np.load(open('alex_bottleneck_svmfeatures+finetune56_train'+ str(fold_count) +'.npy' , 'rb'))
    svm_test = np.load(open('alex_bottleneck_svmfeatures+finetune56_validation'+ str(fold_count) + '.npy', 'rb'))


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
    data, labels, lz = util.load_data()
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

        save_bottleneck_features(data[train], data[test],labels[train], labels[test])
        scores = train_top_model(labels[train], labels[test])

        save_bottleneck_svmfeatures(data[train], data[test],labels[train], labels[test])
        svm_scores = train_svm(labels[train], labels[test])

        total_scores = total_scores + svm_scores
        fold_count = fold_count + 1
    print("Average acc : %.2f%%" % (total_scores/n_folds*100))
