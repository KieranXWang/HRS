import numpy as np
import os
import tensorflow as tf
import keras
import keras.backend as K

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, Reshape, Conv2D, MaxPooling2D
from keras.optimizers import SGD

from project_utils import get_data, get_dimensions
from keras_utils import construct_model_by_blocks

# CIFAR data and information
[X_train, X_test, Y_train, Y_test] = get_data(dataset='CIFAR', scale1=True, one_hot=True, percentage=0.05)
img_rows, img_cols, img_channels = get_dimensions('CIFAR')


# FLAGS
MODEL_INDICATOR = 'test_hrs[5][5]'
STRUCTURE = [5, 5]
nb_block = len(STRUCTURE)
TRAINING_EPOCH = [2, 2]
try:
    os.makedirs('./Model/CIFAR_models/' + MODEL_INDICATOR + '/')
except: pass

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

block_definitions = [block_0, block_1]

# loss definition
def fn(correct, predicted):
    return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)
# optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# start the training process
trained_blocks = None

# crete all channels
HRS_channels = []
for block_idx in range(nb_block):
    block_channels = []
    for channel_idx in range(STRUCTURE[block_idx]):
        channel = block_definitions[block_idx]()
        block_channels.append(channel)
    HRS_channels.append(block_channels)

for block_idx in range(nb_block):
    for channel_idx in range(STRUCTURE[block_idx]):
        if trained_blocks is None:
            Input = InputLayer(input_shape=(img_rows, img_cols, img_channels))
            model_structure_list = [Input] + [block[channel_idx] for block in HRS_channels]
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
            for channel in [HRS_channels[i][channel_idx] for i in range(1, nb_block)]:
                for layer in channel.layers:
                    if hasattr(layer, 'kernel_initializer'):
                        layer.kernel.initializer.run(session=session)
                    if hasattr(layer, 'bias_initializer'):
                        layer.bias.initializer.run(session=session)

        else:
            # model_structure_list = [trained_blocks] + [block[channel_idx] for ]


            print('debug')








