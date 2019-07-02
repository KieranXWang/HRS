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