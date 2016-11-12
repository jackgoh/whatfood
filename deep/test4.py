from keras.optimizers import SGD, RMSprop
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from convnetskeras.convnets import preprocess_image_batch, convnet
import scipy.misc
from keras.regularizers import l2
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn import svm
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

nb_train_samples = 1340
nb_validation_samples = 660
nb_epoch = 200

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
    model = convnet('alexnetfc',weights_path="../dataset/alexnet_weights.h5", heatmap=False)

    bottleneck_features_train = []
    bottleneck_features_validation = []

    j = 0
    for i in X_train:
        temp = X_train[j]
        temp = temp[None, ...]

        bottleneck_features_train.append(model.predict(temp, batch_size=32)[0])
        j+1
    bottleneck_features_train = np.array(bottleneck_features_train)
    np.save(open('alex_bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    j = 0
    for i in X_test:
        temp = X_train[j]
        temp = temp[None, ...]
        bottleneck_features_validation.append(model.predict(temp, batch_size=32)[0])
        j+1
    bottleneck_features_validation = np.array(bottleneck_features_validation)
    np.save(open('alex_bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)


    print bottleneck_features_train.shape
    print bottleneck_features_validation.shape


def train_top_model(y_train, y_test):
    train_data = np.load(open('alex_bottleneck_features_train.npy' , 'rb'))
    #train_labels = np.array([0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('alex_bottleneck_features_validation.npy', 'rb'))
    #validation_labels = np.array([0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    svm_train_data = train_data.reshape(1340,9216)
    svm_test_data = validation_data.reshape(660,9216)

    shape=train_data.shape[1:]

    print "Training CNN.."
    model = Sequential()
    model.add(Flatten(name='flatten', input_shape=shape))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes,activation='softmax'))
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = SGD(lr=0.01)
    rms = RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=0.01)
    model.compile(optimizer=rms, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data, y_train,
              nb_epoch=nb_epoch, batch_size=32,
              validation_data=(validation_data, y_test), verbose=0)

    scores = model.evaluate(validation_data, y_test, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


    print "Training SVM.."
    clf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0)

    clf.fit(svm_train_data, y_train.ravel())
    #y_pred = clf.predict(test_data)
    score = clf.score(svm_test_data, y_test.ravel())
    print score

if __name__ == "__main__":
    print "Loading data.."
    data, labels, lz = load_data()
    data = data.astype('float32')
    data /= 255
    lz = np.array(lz)
    print "Data loaded !"

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.33, random_state=seed)
    print X_train.shape
    print X_test.shape
    print "Test train splitted !"

    #save_bottlebeck_features(X_train, X_test, y_train, y_test)
    train_top_model(y_train, y_test)
