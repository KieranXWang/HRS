import keras
from keras.models import Model


def construct_model_by_blocks(block_list):
    # concatenate all blocks
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


def construct_switching_blocks(indicator, structure, blocks_definition, load_weights=True):
    '''
    Note: structure can be different from indicator, indicator will only be used to load weights (if load_weights ==
    True). This is so designed because this function may be used to construct the switching part of HRS model under
    training.
    '''

    # assert nb blocks
    try:
        assert len(structure) == len(blocks_definition)
