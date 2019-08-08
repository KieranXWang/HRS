import argparse
import numpy as np
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, Reshape, Conv2D, MaxPooling2D, concatenate

from keras_utils import construct_hrs_model
from project_utils import get_data

'''
block definition
'''
# block definitions
def block_0():
    channel = Sequential()
    channel.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3)))
    channel.add(Activation('relu'))
    channel.add(Conv2D(64, (3, 3)))
    channel.add(Activation('relu'))
    channel.add(MaxPooling2D(pool_size=(2, 2)))
    channel.add(Conv2D(128, (3, 3)))
    channel.add(Activation('relu'))
    channel.add(Conv2D(128, (3, 3)))
    channel.add(Activation('relu'))
    channel.add(MaxPooling2D(pool_size=(2, 2)))
    channel.add(Flatten())
    channel.add(Dense(256))
    channel.add(Activation('relu'))
    channel.add(Dropout(0.7))

    return channel


def block_1():
    channel = Sequential()
    channel.add(Dense(256, input_shape=(256,)))
    channel.add(Activation('relu'))
    channel.add(Dense(10))

    return channel


generate_blocks = [block_0, block_1]


def test_acc(MODEL_INDICATOR, DATASET):
    # construct model
    keras.backend.set_learning_phase(0)
    model = construct_hrs_model(dataset=DATASET, model_indicator=MODEL_INDICATOR, blocks_definition=generate_blocks)

    # get data
    [X_train, X_test, Y_train, Y_test] = get_data(dataset=DATASET, scale1=True, one_hot=False, percentage=0.01)

    # note: it is more accurate to feed data points one by one, because of the randomness of the model
    # PS: you don't want to get the acc just for a single model realization
    score = []
    for i in range(X_test.shape[0]):
        x = X_test[i:i + 1]
        y = Y_test[i]
        # pred = keras_model.predict(x)
        pred = np.argmax(model.predict(x)[0])
        if np.array_equal(y, pred):
            score.append(1)
        else:
            score.append(0)

    acc = np.mean(np.array(score))

    print('Test Acc. of Model: %s is %.2f' % (MODEL_INDICATOR, acc))
    return acc




if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--model_indicator', default='test_hrs[10][10]', help='model indicator, format: model_name[5][5] for'
                                                                            'a HRS model with 5 by 5 channels')
    parser.add_argument('--dataset', default='CIFAR', help='CIFAR or MNIST')

    args = parser.parse_args()
    test_acc(MODEL_INDICATOR=args.model_indicator,
             DATASET=args.dataset)