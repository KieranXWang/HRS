import numpy as np
import os
import tensorflow as tf
import keras

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, Reshape, Conv2D, MaxPooling2D
from keras.optimizers import SGD

from project_utils import get_data, get_dimensions
from keras_utils import construct_model_by_blocks

# CIFAR data and information
[X_train, X_test, Y_train, Y_test] = get_data(dataset='CIFAR', scale1=True, one_hot=True)
img_rows, img_cols, img_channels = get_dimensions('CIFAR')


# FLAGS
STRUCTURE = [5, 5]
nb_block = len(STRUCTURE)

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

for train_block_idx in range(nb_block):
    for channel_idx in range(STRUCTURE[train_block_idx]):
        if trained_blocks is None:
            Input = InputLayer(input_shape=(img_rows, img_cols, img_channels))
            model_structure_list = [Input] + [channels[channel_idx] for channels in HRS_channels]
            model = construct_model_by_blocks(model_structure_list)

            print('debug')








