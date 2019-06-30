import numpy as np
from Model.CIFAR_model_utils import load_cifar_data
from Model.CIFAR_model_utils import choose_defense_model as choose_cifar
from Model.MNIST_model_utils import load_mnist_data
from Model.MNIST_model_utils import choose_defense_model as choose_mnist


def test_acc(indicator, weights=None, dataset='CIFAR'):
    '''

    Args:
        indicator: 'single' ...
        model: keras model weight file dir
        dataset: 'MNIST' or 'CIFAR'

    Returns:

    '''

    if dataset == 'CIFAR':
        keras_model = choose_cifar(indicator)
        [X_train, X_test, Y_train, Y_test] = load_cifar_data(scale1=True, one_hot=False)
    elif dataset == 'MNIST':
        keras_model = choose_mnist(indicator)
        [X_train, X_test, Y_train, Y_test] = load_mnist_data(scale1=True, one_hot=False)

    if weights:
        keras_model.load_weights(weights)

    score = []
    for i in range(X_test.shape[0]):
        x = X_test[i:i + 1]
        y = Y_test[i]
        # pred = keras_model.predict(x)
        pred = np.argmax(keras_model.predict(x)[0])
        if np.array_equal(y, pred):
            score.append(1)
        else:
            score.append(0)

    acc = np.mean(np.array(score))

    return acc


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

