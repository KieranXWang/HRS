import numpy as np
import os
import tensorflow as tf
import keras

from project_utils import get_data

# CIFAR data
[X_train, X_test, Y_train, Y_test] = get_data('CIFAR', True, True, 0.5)
