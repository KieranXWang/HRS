import argparse
import numpy as np
import keras

from keras_utils import construct_hrs_model
from project_utils import get_data
from block_split_config import get_split


def test_acc(model_indicator, split, dataset):
    # get block definitions
    blocks_definition = get_split(split, dataset)

    # construct model
    keras.backend.set_learning_phase(0)
    model = construct_hrs_model(dataset=dataset, model_indicator=model_indicator, blocks_definition=blocks_definition)

    # get data
    [X_train, X_test, Y_train, Y_test] = get_data(dataset=dataset, scale1=True, one_hot=False, percentage=0.01)

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

    print('Test Acc. of Model: %s is %.2f' % (model_indicator, acc))
    return acc




if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--model_indicator', default='test_hrs[10][10]', help='model indicator, format: model_name[5][5] for'
                                                                            'a HRS model with 5 by 5 channels')
    parser.add_argument('--split', default='default', help='the indicator of channel structures in each block')
    parser.add_argument('--dataset', default='CIFAR', help='CIFAR or MNIST')

    args = parser.parse_args()
    test_acc(model_indicator=args.model_indicator,
             dataset=args.dataset,
             split=args.split)