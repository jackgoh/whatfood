from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import BatchNormalization
from sklearn.svm import SVC
import theano
from keras.utils import np_utils


def svc(traindata,trainlabel,testdata,testlabel):
    print("Start training SVM...")
    svcClf = SVC(C=1.0,kernel="rbf",cache_size=3000)
    svcClf.fit(traindata,trainlabel)

    pred_testlabel = svcClf.predict(testdata)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("cnn-svm Accuracy:",accuracy)

#each add as one layer
model = Sequential()

#1 .use convolution,pooling,full connection
model.add(Convolution2D(5, 3, 3,border_mode='valid',input_shape=(1, 28, 28),activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(10, 3, 3,activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(100,activation='tanh')) #Full connection

model.add(Dense(10,activation='softmax'))

#2 .just only user full connection
# model.add(Dense(100,input_dim = 784, init='uniform',activation='tanh'))
# model.add(Dense(100,init='uniform',activation='tanh'))
# model.add(Dense(10,init='uniform',activation='softmax'))

# sgd = SGD(lr=0.2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

(X_train, y_train), (X_test, y_test) = mnist.load_data()
#change data type,keras category need ont hot
#2 reshape
#X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]) #X_train.shape[0] 60000 X_train.shape[1] 28  X_train.shape[2] 28
#1 reshape
X_train = X_train.reshape(X_train.shape[0],1,X_train.shape[1],X_train.shape[2])

Y_train = np_utils.to_categorical(y_train, 10)

#new label for svm
y_train_new = y_train[0:42000]
y_test_new = y_train[42000:]

#new train and test data
X_train_new = X_train[0:42000]
X_test = X_train[42000:]
Y_train_new = Y_train[0:42000]
Y_test = Y_train[42000:]

model.fit(X_train, Y_train, validation_split=0.33, nb_epoch=2, batch_size=10)

print("Validation...")
val_loss,val_accuracy = model.evaluate(X_test, Y_test)
print "val_loss: %f" %val_loss
print "val_accuracy: %f" %val_accuracy

#define theano funtion to get output of FC layer
get_feature = theano.function([model.layers[0].input],model.layers[6].output,allow_input_downcast=False)
FC_train_feature = get_feature(X_train_new)
FC_test_feature = get_feature(X_test)
svc(FC_train_feature,y_train_new,FC_test_feature,y_test_new)