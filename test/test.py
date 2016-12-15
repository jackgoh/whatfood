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

def tune(X_train, X_test, y_train, y_test):
    Y_train = np_utils.to_categorical(y_train, config.nb_class)
    Y_test = np_utils.to_categorical(y_test, config.nb_class)

    model = util.load_alexnet_model(weights_path=config.alexnet_weights_path, nb_class=config.nb_class)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=SGD(lr=1e-6, momentum=0.9),
        metrics=['accuracy'])

    print "Fine-tuning CNN.."

    hist = model.fit(X_train, Y_train,
              nb_epoch=4000, batch_size=32,verbose=1,
              validation_data=(X_test, Y_test))

    util.save_history(hist,"finetune567_fold"+ str(fold_count),fold_count)

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Softmax %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    model.save_weights("models/finetune567_weights"+ str(fold_count) +".h5")

    model = None
    model = util.load_svm_alex_model(weights_path=config.alexnet_weights_path, nb_class=config.nb_class)
    print "Generating train features for SVM.."
    svm_train = model.predict(X_train)
    print svm_train.shape
    print "Generating test features for SVM.."
    svm_test = model.predict(X_test)
    print svm_test.shape

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

    # Visualization of confusion matrix
    #np.set_printoptions(precision=2)
    #plt.figure()
    #plot_confusion_matrix(cm)
    #plt.show()

    # Clear memory
    X_train = None
    Y_train = None
    svm_train = None
    svm_test = None

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

    skf = StratifiedKFold(y=lz, n_folds=config.n_folds, shuffle=False)

    for i, (train, test) in enumerate(skf):
        print "Test train Shape: "
        print data[train].shape
        print data[test].shape
        print ("Running Fold %d / %d" % (i+1, config.n_folds))

        scores = tune(data[train], data[test],labels[train], labels[test])
        total_scores = total_scores + scores
        fold_count = fold_count + 1
    print("Average acc : %.2f%%" % (total_scores/config.n_folds*100))
