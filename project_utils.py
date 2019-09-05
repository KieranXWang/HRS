import numpy as np
from keras.datasets import cifar10, mnist
from keras.utils import np_utils

from Model.CIFAR_model_utils import choose_defense_model as choose_cifar
from Model.MNIST_model_utils import choose_defense_model as choose_mnist


def load_cifar_data(one_hot=True, scale1=True):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    if one_hot:
        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_test = np_utils.to_categorical(Y_test, 10)
    else:
        Y_train = np.reshape(Y_train, (Y_train.shape[0],))
        Y_test = np.reshape(Y_test, (Y_test.shape[0],))


    if scale1:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

    return X_train, X_test, Y_train, Y_test


def load_mnist_data(one_hot=True, scale1=True):
    # the defualt is 0-255, not one hot coding
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # reshape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))


    if one_hot:
        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_test = np_utils.to_categorical(Y_test, 10)

    if scale1:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

    return X_train, X_test, Y_train, Y_test

def get_data(dataset, scale1=True, one_hot=False, percentage=None):
    if dataset == 'CIFAR':
        [X_train, X_test, Y_train, Y_test] = load_cifar_data(scale1=scale1, one_hot=one_hot)
        if percentage:
            samples = X_train.shape[0]
            use_samples = int(samples * percentage)
            X_train = X_train[0:use_samples]
            Y_train = Y_train[0:use_samples]

    elif dataset == 'MNIST':
        [X_train, X_test, Y_train, Y_test] = load_mnist_data(scale1=scale1, one_hot=one_hot)
        if percentage:
            samples = X_train.shape[0]
            use_samples = int(samples * percentage)
            X_train = X_train[0:use_samples]
            Y_train = Y_train[0:use_samples]

    return [X_train, X_test, Y_train, Y_test]


def get_dimensions(dataset):
    '''

    Args:
        dataset: CIFAR or MNIST

    Returns: [height, width, channels]

    '''
    if dataset == 'CIFAR':
        return [32,32,3]
    elif dataset == 'MNIST':
        return [28,28,1]

