import numpy as np
import os
import tensorflow as tf
import keras
import keras.backend as K
import argparse

from keras.models import Model
from keras.layers import InputLayer
from keras.optimizers import SGD

from project_utils import get_data, get_dimensions
from keras_utils import construct_switching_block
from block_split_config import get_split


def train_hrs(model_indicator, training_epoch, split='default', dataset='CIFAR'):
    # get block definitions
    blocks_definition = get_split(split, dataset)

    # parse structure
    structure = [int(ss[:-1]) for ss in model_indicator.split('[')[1:]]
    nb_block = len(structure)

    # make sure model_indicator, training_epoch and split all have the same number of blocks
    assert nb_block == len(training_epoch) == len(blocks_definition), "The number of blocks indicated by " \
                                                                      "model_indicator, training_epoch and split must " \
                                                                      "be the same!"

    # create weights save dir
    save_dir = './Model/%s_models/' % dataset + model_indicator + '/'
    try:
        os.makedirs('./Model/%s_models/' % dataset + model_indicator + '/')
    except: pass

    # dataset and input dimensions
    [X_train, X_test, Y_train, Y_test] = get_data(dataset=dataset, scale1=True, one_hot=True, percentage=1)
    img_rows, img_cols, img_channels = get_dimensions(dataset)

    # loss definition
    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct, logits=predicted)

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
                weight_dir = save_dir + '%d_' % i + '%d'
                block_output = construct_switching_block(block_input, structure[i], blocks_definition[i], weight_dir)
                block_input = block_output
            trained_blocks_output = block_output

        # construct the part to train
        # normal blocks (with only one channel) from block_idx to the end
        for channel_idx in range(structure[block_idx]):
            block_input = trained_blocks_output
            # the channel to train
            channel_to_train = blocks_definition[block_idx]()
            block_output = channel_to_train(block_input)
            block_input = block_output
            # add following blocks in any
            for j in range(block_idx+1, nb_block):
                channel = blocks_definition[j]()
                block_output = channel(block_input)
                block_input = block_output

            # construct the model object
            model = Model(input=model_input.input, output=block_output)
            # optimizer
            sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            # training
            model.compile(loss=fn, optimizer=sgd, metrics=['accuracy'])
            model.fit(X_train, Y_train, batch_size=128, validation_data=(X_test, Y_test),
                      nb_epoch=training_epoch[block_idx], shuffle=True)

            # save weights of this channel
            channel_to_train.save_weights(save_dir + '%d_%d' % (block_idx, channel_idx))

        # after training all channels in this block, reset tf graph
        K.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_indicator', default='test_hrs[10][10]', help='model indicator, format: model_name[5][5] for'
                                                                            'a HRS model with 5 by 5 channels')
    parser.add_argument('--split', default='default', help='the structures of channels in each block')
    parser.add_argument('--train_schedule', default=[40, 40], help='number of epochs for training each block', type=int,
                        nargs='*')
    parser.add_argument('--dataset', default='CIFAR', help='CIFAR or MNIST')

    args = parser.parse_args()
    train_hrs(model_indicator=args.model_indicator,
              training_epoch=args.train_schedule,
              dataset=args.dataset,
              split=args.split)
    pass







