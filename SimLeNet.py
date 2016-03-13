from Helper import Helper
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np

batch_size = 100
nb_classes = 2
nb_epoch = 80
data_augmentation = False

# input image dimensions
img_rows, img_cols = 64, 64
img_channels = 3


train_path = "/scratch/basag/train_data/"
test_path = "/scratch/basag/test_data/"
train_labels_path = "/scratch/basag/train_labels.csv"
test_labels_path = "/scratch/basag/test_labels.csv"

X_train, Y_train, X_test, Y_test = Helper.load_data(train_path, train_labels_path, test_path, test_labels_path)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

#print(np.shape(X_train))
#print(len(Y_train))
#print(np.shape(X_test))
#print(len(Y_test))


model = Graph()
model.input(shape=(img_channels, img_rows, img_cols), name='input')
model.node(Convolution2D(32, 7, 7), name='conv11', input='input', activation='relu')
model.node(Convolution2D(32, 3, 3), name='conv12', input='conv11', activation='relu')
model.node(MaxPooling2D(poolsize=(2, 2)), name='pool1', input='conv12'))

model.node(Convolution2D(32, 1, 1), name='conv21', input='pool1', activation='relu')
model.node(Convolution2D(32, 1, 1), name='conv22', input='pool1', activation='relu')
model.node(Convolution2D(32, 1, 1), name='conv23', input='pool1', activation='relu')
model.node(MaxPooling2D(poolsize=(2, 2)), name='pool21', input='pool1'))

model.node(Convolution2D(32, 3, 3), name='conv31', input='conv22', activation='relu')
model.node(Convolution2D(32, 5, 5), name='conv32', input='conv23', activation='relu')
model.node(Convolution2D(32, 1, 1), name='conv33', input='pool21', activation='relu')

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


model.compile(loss={'conv31':'mae', 'conv32':'mse', 'conv33': 'mse'}, loss_merge='sum', optimizer='sgd')
model.fit(train={'input1':X1, 'input2':X2, 'output1':Y1, 'conv24_output':Y2})
