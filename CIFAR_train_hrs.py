import numpy as np
import os
import tensorflow as tf
import keras
import keras.backend as K
import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, Reshape, Conv2D, MaxPooling2D, concatenate
from keras.optimizers import SGD

from project_utils import get_data, get_dimensions
from keras_utils import construct_model_by_blocks, construct_switching_blocks, construct_switching_block


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


def train_hrs(MODEL_INDICATOR, TRAINING_EPOCH, blocks_definition=generate_blocks, DATASET='CIFAR'):
    # parse structure
    STRUCTURE = [int(ss[:-1]) for ss in MODEL_INDICATOR.split('[')[1:]]
    nb_block = len(STRUCTURE)

    # create weights save dir
    SAVE_DIR = './Model/%s_models/' % DATASET + MODEL_INDICATOR + '/'
    try:
        os.makedirs('./Model/%s_models/' % DATASET + MODEL_INDICATOR + '/')
    except: pass

    # dataset and input dimensions
    [X_train, X_test, Y_train, Y_test] = get_data(dataset=DATASET, scale1=True, one_hot=True, percentage=0.05)
    img_rows, img_cols, img_channels = get_dimensions(DATASET)


    # loss definition
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

    # optimizer
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    '''
    Training HRS
    '''
    # start the training process
    for block_idx in range(nb_block):
        print("start training the %d\'s block" % block_idx)

        # construct the trained part:
        # switching blocks up to the (block_idx - 1)'s block
        if block_idx == 0:
            model_input = InputLayer(input_shape=(img_rows, img_cols, img_channels))
            # note: for InputLayer the input and output tensors are the same one.
            trained_blocks_output = model_input.output
        else:
            model_input = InputLayer(input_shape=(img_rows, img_cols, img_channels))

            # build switching blocks
            block_input = model_input.output
            for i in range(block_idx):
                block_output = construct_switching_block(block_input, STRUCTURE[i], blocks_definition[i])
                block_input = block_output
            trained_blocks_output = block_output

        # construct the part to train
        # normal blocks (with only one channel) from block_idx to the end
        for channel_idx in range(STRUCTURE[block_idx]):
            block_input = trained_blocks_output
            # the channel to train
            channel_to_train = blocks_definition[block_idx]()
            block_output = channel_to_train(block_input)
            # add following blocks in any
            for j in range(block_idx+1, nb_block):
                channel = blocks_definition[j]()
                block_output = channel(block_input)
                block_input = block_output

                pass

            # construct the model object
            model = Model(input=model_input.input, output=block_output)
            # training
            model.compile(loss=fn, optimizer=sgd, metrics=['accuracy'])
            model.fit(X_train, Y_train, batch_size=128, validation_data=(X_test, Y_test),
                      nb_epoch=TRAINING_EPOCH[block_idx], shuffle=True)

            # save weights of this channel
            channel_to_train.save_weights(SAVE_DIR + '%d_%d' % (block_idx, channel_idx))

        # after training all channels in this block, reset tf graph
        K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_indicator', default='test_hrs[5][5]', help='model indicator, format: model_name[5][5] for'
                                                                            'a HRS model with 5 by 5 channels')
    parser.add_argument('--train_schedule', default=[2, 2], help='number of epochs for training each block', type=int,
                        nargs='*')
    parser.add_argument('--dataset', default='CIFAR', help='CIFAR or MNIST')

    args = parser.parse_args()
    train_hrs(MODEL_INDICATOR=args.model_indicator,
              TRAINING_EPOCH=args.train_schedule)
    pass







