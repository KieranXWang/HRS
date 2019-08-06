import keras
import os
from keras.models import Model, Sequential
from keras.layers import InputLayer

from project_utils import get_dimensions
from Model.CIFAR_model_utils import RandomMask


def construct_model_by_blocks(block_list):
    # connect all blocks
    if len(block_list) == 1:
        i = block_list[0].input
        o = block_list[0].output
    else:
        i = block_list[0].input
        o = block_list[0].output
        idx = 1
        while idx < len(block_list):
            o = block_list[idx](o)
            idx += 1
    model = Model(input=i, output=o)

    return model


def construct_switching_blocks(dataset, indicator, structure, blocks_definition, load_weights=True):
    '''
    Note: structure can be different from indicator, indicator will only be used to load weights (if load_weights ==
    True). This is so designed because this function may be used to construct the switching part of HRS model under
    training.
    '''

    # assert nb blocks
    assert len(structure) == len(blocks_definition), 'arg structure and block_definition need to have the same length'
    nb_block = len(structure)

    # assert weights exist
    weights_dir = './Model/%s_Models/%s' % (dataset, indicator)
    assert os.path.exists(weights_dir), '%s does not exist' % weights_dir

    # input
    img_rows, img_cols, img_channels = get_dimensions(dataset)
    model_input = InputLayer(input_shape=(img_rows, img_cols, img_channels))

    # loop over blocks
    block_input = model_input
    for i in range(nb_block):
        for j in range(structure[i]):
            channel = blocks_definition[i]()


def construct_switching_block(input, nb_channels, channel_definition, weights, freeze_channel=True):

    channel_output_list = []
    for channel_idx in range(nb_channels):
        channel = channel_definition()
        channel_output = channel(input)
        channel_output_list.append(channel_output)
        # load weights
        if weights:
            channel.load_weights(weights % channel_idx)
        if freeze_channel:
            for layer in channel.layers:
                layer.trainable = False

    # using a random mask to mask inactive channels
    block_output = RandomMask(nb_channels)(channel_output_list)
    return block_output

