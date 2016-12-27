#Author: Jacob Gildenblat, 2014
#License: you may use this for whatever you like
import sys, glob, argparse
import matplotlib.pyplot as plt
import numpy as np
import math, cv2
import csv
import time
import pickle
import hickle as hkl

from scipy.stats import multivariate_normal
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
from _vlfeat import *
'''
def dictionary(descriptors, N):
    em = cv2.EM(N)
    em.train(descriptors)

    return np.float32(em.getMat("means")), \
        np.float32(em.getMatVector("covs")), np.float32(em.getMat("weights"))[0]
'''

def dictionary(descriptors, N):
    means, covs, priors, _ = vl_gmm(descriptors, N)
    #save("means.gmm", gmm.means_)
    #save("covs.gmm", gmm.covars_)
    #save("weights.gmm", gmm.weights_)
    return means, covs, priors

def image_descriptors(file):
    img = cv2.imread(file, 0)
    #img = cv2.resize(img, (256, 256))
    img = np.array(img, 'f', order='F') # 'F' = column-major order!
    img = np.array(img, 'float32')

    f, descriptors = vl_sift(img,floatDescriptors=True, verbose=False) #0.7225 acc
    #descriptor = cv2.DescriptorExtractor_create("OpponentSURF")
    #_ , descriptors = cv2.SIFT().detectAndCompute(img, None)
    #descriptors = apply_pca(descriptors)
    #f, descriptors = vl_dsift(img, fast=False, norm=True, step=100, floatDescriptors=True, verbose=False, size=5) #0.33 acc
    #f, descriptors = vl_phow(img, verbose=False) #0.73125 128x128
    descriptors = np.swapaxes(descriptors,0,1)
    return descriptors

def folder_descriptors(folder):
    files = glob.glob(folder + "/*.jpg")
    print "Calculating SIFT descriptors. Number of images in "+ folder +" is " + str(len(files))
    return np.concatenate([image_descriptors(file) for file in files])

def fisher_vector(samples, means, covs, w):
    samples = np.swapaxes(samples,0,1)
    fv = vl_fisher(samples, means, covs, w, fast=True, improved=True)
    test = []

    for i in fv:
        test = np.append(test , i[0])

    return test

def apply_pca(image_descriptors):
    pca = PCA(n_components=64)
    return (pca.fit_transform(image_descriptors))

def generate_gmm(input_folder, N):
    loadfeature = False

    # start count execution time
    start_time = time.time()
    words = load_feature() if loadfeature else np.concatenate([folder_descriptors(folder) for folder in glob.glob(input_folder + '/*')])
    print("Feature extration: %s seconds" % (time.time() - start_time))
    hkl.dump(words, 'sift_feature.h5', mode='w')

    words = np.swapaxes(words,0,1)
    print "Number of words ", words.shape
    print("Training GMM of size", N)
    start_time = time.time()
    means, covs, weights = dictionary(words, N)
    print("GMM: %s seconds" % (time.time() - start_time))

    print means.shape
    print covs.shape
    print weights.shape

    np.save("means.gmm", means)
    np.save("covs.gmm", covs)
    np.save("weights.gmm", weights)
    return means, covs, weights

def get_fisher_vectors_from_folder(folder, gmm):
    files = glob.glob(folder + "/*.jpg")
    return np.float32([fisher_vector(image_descriptors(file), *gmm) for file in files])

def fisher_features(folder, gmm):
    print "Encoding FV"
    folders = glob.glob(folder + "/*")
    start_time = time.time()
    features = {f : get_fisher_vectors_from_folder(f, gmm) for f in folders}
    print("Fisher Vector: %s seconds" % (time.time() - start_time))

    f = open('sift_fv.pkl', 'wb')
    pickle.dump(features, f)
    #hkl.dump(features, "sift_fv.h5", mode='w')

    return features

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

def train(gmm, features):
    X = np.concatenate(features.values())
    y = np.concatenate([np.float32([i]*len(v)) for i,v in zip(range(0, len(features)), features.values())])
    print y.shape

    k = 2
    sfold = StratifiedKFold(y, n_folds=k)

    total_score = 0
    fold_count = 1

    for train_index, test_index in sfold:
        print "Training SVM.."
        train_data, test_data = X[train_index], X[test_index]
        train_label, test_label = y[train_index], y[test_index]
        start_time = time.time()
        clf = svm.SVC(kernel='linear', C=1.0, probability=True)
        clf.fit(train_data, train_label)
        print("SVM: %s seconds" % (time.time() - start_time))

        #print "Saving SIFT SVM model.."
        #hkl.dump(clf, "sift_svm_"+ str(fold_count) +".h5")

        y_pred = clf.predict(test_data)

        # Compute confusion matrix
        cm = confusion_matrix(test_label, y_pred)
        np.set_printoptions(precision=2)
        #print('Confusion matrix, without normalization')
        #print(cm)
        plt.figure()
        plot_confusion_matrix(cm)

        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print('Normalized confusion matrix')
        #print(cm_normalized)
        plt.figure()
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

        plt.show()


        score = clf.score(test_data,test_label)
        print "Acc for fold ", fold_count, "= ", score

        scores = clf.predict_proba(test_data)
        n = 5
        indices = np.argsort(scores)[:,:-n-1:-1]

        # Get accuracy
        top1 = 0.0
        top5 = 0.0

        correct_predict_top1 = np.zeros((100,), dtype=np.int)
        correct_predict_top5 = np.zeros((100,), dtype=np.int)

        for image_index, index_list in enumerate(indices):
            if test_label[image_index] == index_list[0]:
                top1 += 1.0
            if test_label[image_index] in index_list:
                top5 += 1.0

        image_index = None
        index_list = None
        start_index = 0
        end_index = 99

        for class_label in range(0,100):
            for image_index in range(start_index,end_index+1):
                if test_label[image_index] == indices[image_index][0]:
                    correct_predict_top1[class_label] += 1
                if test_label[image_index] in indices[image_index]:
                    correct_predict_top5[class_label] += 1
            start_index += 100
            end_index += 100

        objects = ['AisKacang' , 'AngKuKueh' , 'ApamBalik' , 'Asamlaksa' , 'Bahulu' , 'Bakkukteh',
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
        y_pos = np.arange(len(objects))
        performance = correct_predict_top1

        rects1 = plt.bar(y_pos, performance)
        plt.xticks(y_pos, objects, rotation='vertical')
        plt.ylabel('Total true positive')
        plt.title('Total true positive per sample')

        autolabel(rects1)
        plt.savefig('barchart_deep_feaures'+ str(fold_count) +'.png')
        plt.show()

        print correct_predict_top1
        print correct_predict_top5

        print('Top-1 Accuracy: ' + str(top1 / len(test_label) * 100.0) + '%')
        print('Top-5 Accuracy: ' + str(top5 / len(test_label) * 100.0) + '%')
        total_score = total_score + top1 / len(test_label)
        fold_count = fold_count + 1

    print "Accuracy : ", total_score/k
    return clf

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



def load_gmm(folder = ""):
    print "Loading GMM.."
    f = file("means.gmm.npy","rb")
    means = np.load(f)

    f = file("covs.gmm.npy","rb")
    covs = np.load(f)

    f = file("weights.gmm.npy","rb")
    weights = np.load(f)

    return means, covs, weights

def load_fv():
    print "Loading SIFT fisher vector.."
    with open('sift_fv.pkl', 'rb') as f:
        fv = pickle.load(f)
    #fv = hkl.load("sift_fv.h5")
    return fv

def load_feature():
    print "Loading SIFT Feature.."
    with open('sift_feature.pkl', 'rb') as f:
        feature = pickle.load(f)
    #feature = hkl.load('sift_feature.pkl')
    return feature

number = 32
working_folder = "../dataset/food100"
gengmm_folder = "../dataset/food100"
loadgmm = True
loadfv = True

total_time = time.time()

gmm = load_gmm(gengmm_folder) if loadgmm else generate_gmm(gengmm_folder, number)
fisher_features = load_fv() if loadfv else fisher_features(working_folder, gmm)
classifier = train(gmm, fisher_features)

print("Total: %s seconds" % (time.time() - total_time))
