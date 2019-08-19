import argparse
import os
import keras
import tensorflow as tf
import numpy as np

from keras.models import Sequential, Model
from keras.layers import ZeroPadding2D, LocallyConnected2D, Activation
from keras.optimizers import SGD

from project_utils import get_data
from keras_utils import construct_hrs_model
from block_split_config import get_split


def defend_adversarial_reprogramming(model_indicator, split, epochs):
    save_dir = './Adversarial_Reprogramming/' + args.model_indicator + '/'
    try: os.makedirs(save_dir)
    except: pass

    # get MNIST data
    [X_train, X_test, Y_train, Y_test] = get_data(dataset='MNIST', scale1=True, one_hot=False, percentage=0.01)

    # input transfer model
    input_transfer = Sequential()
    input_transfer.add(ZeroPadding2D(padding=3, input_shape=(28, 28, 1)))
    input_transfer.add(LocallyConnected2D(3, (3, 3), activation='relu'))
    input_transfer.add(Activation('tanh'))

    # get block definitions
    blocks_definition = get_split(split, 'CIFAR')

    # target model to reprogram
    keras.backend.set_learning_phase(0)
    model = construct_hrs_model(dataset='CIFAR', model_indicator=model_indicator, blocks_definition=blocks_definition)
    # set layer untrainable
    for layer in model.layers:
        layer.trainable = False

    # overall model
    output = model(input_transfer.output)
    adv_model = Model(input_transfer.input, output)

    # optimizer and loss
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

    # compile
    adv_model.compile(loss=fn,
                      optimizer=sgd,
                      metrics=['accuracy'])

    # training the input transfer
    hist = adv_model.fit(X_train, Y_train,
                         batch_size=128,
                         validation_data=(X_test, Y_test),
                         nb_epoch=epochs,
                         shuffle=True)

    # save training history
    train_acc = hist.history['acc']
    test_acc = hist.history['val_acc']

    np.save(save_dir + 'hist.npy', np.array(hist.history))
    print('Perform adversarial reprogramming to model %s' % model_indicator)
    print('Reprogramming Train Acc. after %d epochs of training is %.4f' % (epochs, train_acc[-1]))
    print('Reprogramming Test Acc. after %d epochs of training is %.4f' % (epochs, test_acc[-1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_indicator', default='test_hrs[10][10]', help='model indicator, format: model_name[5][5] for'
                                                                            'a HRS model with 5 by 5 channels')
    parser.add_argument('--split', default='default', help='the indicator of channel structures in each block')
    parser.add_argument('--epochs', default=50, help='the number of epochs to train (reprogram).')

    args = parser.parse_args()
    defend_adversarial_reprogramming(model_indicator=args.model_indicator,
                                     split=args.split,
                                     epochs=args.epochs)



