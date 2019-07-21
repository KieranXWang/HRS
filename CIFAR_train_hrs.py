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
from keras_utils import construct_model_by_blocks

# CIFAR data and information
[X_train, X_test, Y_train, Y_test] = get_data(dataset='CIFAR', scale1=True, one_hot=True, percentage=0.05)
img_rows, img_cols, img_channels = get_dimensions('CIFAR')

'''
block definition
'''
# block definitions
def block_0():
    block = Sequential()
    block.add(Conv2D(64, (3, 3), input_shape=(img_rows, img_cols, img_channels)))
    block.add(Activation('relu'))
    block.add(Conv2D(64, (3, 3)))
    block.add(Activation('relu'))
    block.add(MaxPooling2D(pool_size=(2, 2)))
    block.add(Conv2D(128, (3, 3)))
    block.add(Activation('relu'))
    block.add(Conv2D(128, (3, 3)))
    block.add(Activation('relu'))
    block.add(MaxPooling2D(pool_size=(2, 2)))
    block.add(Flatten())
    block.add(Dense(256))
    block.add(Activation('relu'))
    block.add(Dropout(0.7))

    return block


def block_1():
    channel = Sequential()
    channel.add(Dense(256, input_shape=(256,)))
    channel.add(Activation('relu'))
    channel.add(Dense(10))

    return channel


generate_blocks = [block_0, block_1]


def train_hrs(MODEL_INDICATOR, TRAINING_EPOCH, blocks_definition=generate_blocks):
    # FLAGS
    MODEL_INDICATOR = 'test_hrs[5][5]'
    STRUCTURE = [5, 5]
    nb_block = len(STRUCTURE)
    TRAINING_EPOCH = [2, 2]
    try:
        os.makedirs('./Model/CIFAR_models/' + MODEL_INDICATOR + '/')
    except: pass


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

        # construct switching blocks up to the (block_idx - 1)'s blocks
        if block_idx == 0:
            model_input = InputLayer(input_shape=(img_rows, img_cols, img_channels))
            trained_blocks = [model_input]
        else:
            #todo
            trained_blocks = None
        # for each channel to train, construct the following blocks
        for channel_idx in range(STRUCTURE[block_idx]):
            following_blocks = [blocks_definition[b]() for b in [block_idx, nb_block]]
            channel_to_train = following_blocks[0]
            model_structure_list = trained_blocks + following_blocks
            model = construct_model_by_blocks(model_structure_list)

            # training
            model.compile(loss=fn, optimizer=sgd, metrics=['accuracy'])
            model.fit(X_train, Y_train, batch_size=128, validation_data=(X_test, Y_test),
                      nb_epoch=TRAINING_EPOCH[block_idx], shuffle=True)

            # save weights of this channel
            channel_to_train.save_weights('%s_%d_%d' % (MODEL_INDICATOR, block_idx, channel_idx))

            # freeze weights of this channel
            for layer in HRS_channels[block_idx][channel_idx].layers:
                layer.trainable = False

            # reset weights of other channels
            session = K.get_session()
            for channel in [HRS_channels[i][channel_idx] for i in range(1, nb_block)]:
                for layer in channel.layers:
                    if hasattr(layer, 'kernel_initializer'):
                        layer.kernel.initializer.run(session=session)
                    if hasattr(layer, 'bias_initializer'):
                        layer.bias.initializer.run(session=session)

            else:
                model_structure_list = [trained_blocks] + [block[channel_idx] for block in HRS_channels[block_idx:]]
                model = construct_model_by_blocks(model_structure_list)

                # training
                model.compile(loss=fn, optimizer=sgd, metrics=['accuracy'])
                model.fit(X_train, Y_train, batch_size=128, validation_data=(X_test, Y_test),
                          nb_epoch=TRAINING_EPOCH[block_idx], shuffle=True)

                # save weights of this channel
                HRS_channels[block_idx][channel_idx].save_weights('%s_%d_%d' % (MODEL_INDICATOR, block_idx, channel_idx))

                # freeze weights of this channel
                for layer in HRS_channels[block_idx][channel_idx].layers:
                    layer.trainable = False

                # reset weights of other channels
                session = K.get_session()
                for channel in [HRS_channels[i][channel_idx] for i in range(block_idx + 1, nb_block)]:
                    for layer in channel.layers:
                        if hasattr(layer, 'kernel_initializer'):
                            layer.kernel.initializer.run(session=session)
                        if hasattr(layer, 'bias_initializer'):
                            layer.bias.initializer.run(session=session)




                print('debug')

        # after training all channels in this block, construct switching block using a random mask layer
        if trained_blocks is None:
            Input = InputLayer(input_shape=(img_rows, img_cols, img_channels))
            block_output_list = []
            for channel in HRS_channels[0]:
                o = channel(Input.output)
                block_output_list.append(o)
            o = concatenate(block_output_list)


            print('debug')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_indicator', default='test_hrs[5][5]', help='model indicator, format: model_name[5][5] for'
                                                                            'a HRS model with 5 by 5 channels')
    parser.add_argument('train_schedule', default='2 2', help='number of epochs for training each block', type=int)

    args = parser.parse_args()
    MODEL_INDICATOR, TRAIN_SCHEDULE = args[0], args[1]
    train_hrs(MODEL_INDICATOR, TRAIN_SCHEDULE)
    pass







