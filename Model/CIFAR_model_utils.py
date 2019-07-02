# all keras code
import os
from keras.models import Sequential
# from CIFAR_model_trans_utils import *
import keras.regularizers as regs
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, InputLayer, Reshape, GaussianNoise, UpSampling2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Lambda
from keras.layers.merge import Average, add
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.utils import np_utils
import tensorflow as tf
import keras
from keras.engine.topology import Layer
import numpy as np
from keras.layers.convolutional import _Conv
# model general parameters
nb_classes = 10
img_rows = 32
img_cols = 32
channels = 3



# define Conv2D + Gaussian noise
# class _Conv_noise(_Conv):
#
#
#     def build(self, input_shape):
#         if self.data_format == 'channels_first':
#             channel_axis = 1
#         else:
#             channel_axis = -1
#         if input_shape[channel_axis] is None:
#             raise ValueError('The channel dimension of the inputs '
#                              'should be defined. Found `None`.')
#         input_dim = input_shape[channel_axis]
#         kernel_shape = self.kernel_size + (input_dim, self.filters)
#
#         self.kernel = self.add_weight(shape=kernel_shape,
#                                       initializer=self.kernel_initializer,
#                                       name='kernel',
#                                       regularizer=self.kernel_regularizer,
#                                       constraint=self.kernel_constraint)
#
#         self.noise = tf.random_normal(shape=kernel_shape, mean=0.0, stddev=0.01, dtype=tf.float32, seed=None,
#                                       name=None)
#
#         if self.use_bias:
#             self.bias = self.add_weight(shape=(self.filters,),
#                                         initializer=self.bias_initializer,
#                                         name='bias',
#                                         regularizer=self.bias_regularizer,
#                                         constraint=self.bias_constraint)
#         else:
#             self.bias = None
#         # Set input spec.
#         self.input_spec = keras.layers.InputSpec(ndim=self.rank + 2,
#                                     axes={channel_axis: input_dim})
#         self.built = True
#
#     def call(self, inputs):
#         if self.rank == 1:
#             outputs = keras.backend.conv1d(
#                 inputs,
#                 self.kernel+self.noise,
#                 strides=self.strides[0],
#                 padding=self.padding,
#                 data_format=self.data_format,
#                 dilation_rate=self.dilation_rate[0])
#         if self.rank == 2:
#             outputs = keras.backend.conv2d(
#                 inputs,
#                 self.kernel+self.noise,
#                 strides=self.strides,
#                 padding=self.padding,
#                 data_format=self.data_format,
#                 dilation_rate=self.dilation_rate)
#         if self.rank == 3:
#             outputs = keras.backend.conv3d(
#                 inputs,
#                 self.kernel+self.noise,
#                 strides=self.strides,
#                 padding=self.padding,
#                 data_format=self.data_format,
#                 dilation_rate=self.dilation_rate)
#
#         if self.use_bias:
#             outputs = keras.backend.bias_add(
#                 outputs,
#                 self.bias,
#                 data_format=self.data_format)
#
#         if self.activation is not None:
#             return self.activation(outputs)
#         return outputs

class Conv2D_noise(_Conv):
    def __init__(self, filters,
                 kernel_size, stddev = 0.01,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.stddev = stddev

        super(Conv2D_noise, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)


    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.noise = tf.random_normal(shape=kernel_shape, mean=0.0, stddev=self.stddev, dtype=tf.float32, seed=None,
                                      name=None)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = keras.layers.InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.rank == 1:
            outputs = keras.backend.conv1d(
                inputs,
                self.kernel+self.noise,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = keras.backend.conv2d(
                inputs,
                self.kernel+self.noise,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = keras.backend.conv3d(
                inputs,
                self.kernel+self.noise,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = keras.backend.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


    def get_config(self):
        config = super(Conv2D_noise, self).get_config()
        config.pop('rank')
        return config



# define Dense layer + Gaussian noise

class Dense_noise(Dense):
    def __init__(self, stddev = 0.01, **kwargs):
        self.stddev = stddev
        super(Dense_noise, self).__init__(**kwargs)

    # def build(self, input_shape):
    #     self.noise = keras.layers.GaussianNoise(0.01)

    # def call(self, inputs):
    #     output = keras.backend.dot(inputs, self.kernel+self.noise)
    #     return output

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        self.noise = tf.random_normal(shape=(input_dim, self.units), mean=0.0, stddev=self.stddev, dtype=tf.float32, seed=None, name=None)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, x):
        output = keras.backend.dot(x, self.kernel+self.noise)
        if self.use_bias:
            output = keras.backend.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


# define random squeezing layer CIFAR

class Random_Squeezing_Layer(Layer):
    def __init__(self, **kwargs):
        super(Random_Squeezing_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Random_Squeezing_Layer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        rnd = tf.random_uniform([None, 32, 32, 3], minval=0, maxval=1, dtype=tf.float32, seed=None, name=None)
        rnd_x = x + rnd
        rnd_x = tf.clip_by_value(rnd_x, 0, 1, name=None)
        return rnd_x



# define random resizing and padding layer CIFAR

class Resizing_Layer_CIFAR(Layer):
    def __init__(self, pad_value=0.5, **kwargs):
        self.pad_value = pad_value
        super(Resizing_Layer_CIFAR, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Resizing_Layer_CIFAR, self).build(input_shape)  # Be sure to call this somewhere!



    def call(self, x):
        # self.rescaled = tf.image.crop_and_resize(x, [[0, 0, 1, 1]], [0], [self.rnd, self.rnd])
        # x = self.rescaled(x)
        rnd = tf.random_uniform((), 29, 32, dtype=tf.int32)
        rescaled = tf.image.crop_and_resize(x, [[0, 0, 1, 1]], [0], [rnd, rnd])
        h_rem = 32 - rnd
        w_rem = 32 - rnd
        pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=self.pad_value)
        padded.set_shape((None, 32, 32, 3))
        return padded


class Resize_general(Layer):
    def __init__(self, pad_value=0.5, scale=1.0, **kwargs):
        '''

        Args:
            pad_value:
            scale: float (0,1], the smallest resize percentage
            **kwargs:
        '''
        self.pad_value = pad_value
        self.scale = scale
        super(Resize_general, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Resize_general, self).build(input_shape)  # Be sure to call this somewhere!



    def call(self, x):
        # self.rescaled = tf.image.crop_and_resize(x, [[0, 0, 1, 1]], [0], [self.rnd, self.rnd])
        # x = self.rescaled(x)
        '''

        Args:
            x: 4-D tensor (batch, row, column, channel)

        Returns:

        '''
        shape = x.get_shape().as_list()
        edge = x.get_shape().as_list()[1]
        low_bound = int(edge * self.scale)
        rnd = tf.random_uniform((), low_bound, edge, dtype=tf.int32)
        rescaled = tf.image.crop_and_resize(x, [[0, 0, 1, 1]], [0], [rnd, rnd])
        h_rem = edge - rnd
        w_rem = edge - rnd
        pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=self.pad_value)
        padded.set_shape((None, shape[1], shape[2], shape[3]))
        return padded


# define VAE

# class DenoisingAutoEncoder:
#
#     def __init__(self, input_shape,
#                  structure,
#                  v_noise=0.0,
#                  activation="relu",
#                  reg_strength=0.0):
#
#         h, w, c = input_shape
#         self.image_shape = input_shape
#         # self.model_dir = model_dir
#         self.v_noise = v_noise
#
#         input_img = Input(shape=self.image_shape)
#         x = input_img
#
#         for layer in structure:
#             if isinstance(layer, int):
#                 x = Conv2D(layer, (3, 3), activation=activation, padding="same",
#                            activity_regularizer=regs.l2(reg_strength))(x)
#             elif layer == "max":
#                 x = MaxPooling2D((2, 2), padding="same")(x)
#             elif layer == "average":
#                 x = AveragePooling2D((2, 2), padding="same")(x)
#             else:
#                 print(layer, "is not recognized!")
#                 exit(0)
#
#         for layer in reversed(structure):
#             if isinstance(layer, int):
#                 x = Conv2D(layer, (3, 3), activation=activation, padding="same",
#                            activity_regularizer=regs.l2(reg_strength))(x)
#             elif layer == "max" or layer == "average":
#                 x = UpSampling2D((2, 2))(x)
#
#         decoded = Conv2D(c, (3, 3), activation='sigmoid', padding='same',
#                          activity_regularizer=regs.l2(reg_strength))(x)
#         self.model = Model(input_img, decoded)

#define mask layer
class Mask_Layer(Layer):

    def __init__(self, nb_submodels, **kwargs):
        self.nb_submodels = nb_submodels
        super(Mask_Layer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Mask_Layer, self).build(input_shape) # Be sure to call this somewhere!

        # build random mask
        ones = tf.ones([1, input_shape[2]])
        zeros = tf.zeros([self.nb_submodels-1, input_shape[2]])
        mask = tf.concat([ones, zeros], 0)

        # shuffle
        self.mask = tf.random_shuffle(mask)


    def call(self, x):
        x = x * self.mask
        x = tf.reduce_sum(x, axis=1)
        return x

    def compute_output_shape(self, input_shape):
        return(input_shape[0], input_shape[2])

# define sap layer
class sap_dense(Layer):

    def __init__(self, **kwargs):
        super(sap_dense, self).__init__(**kwargs)

    def build(self, input_shape):
        super(sap_dense, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):

        r = tf.check_numerics(x, "okay")
        p = keras.backend.abs(r) / tf.reduce_sum(keras.backend.abs(r), axis=1, keep_dims=True)
        N = sum(p.get_shape().as_list()[1:]) * 2
        p_keep = 1 - tf.exp(-N * p)
        rand = keras.backend.random_uniform(keras.backend.shape(p_keep))
        keep = rand < p_keep
        r = keras.backend.cast(keep, tf.float32) * r / (p_keep + 1e-8)
        r = tf.check_numerics(r, "OH NO")
        return r

class sap_conv(Layer):
    def __init__(self, **kwargs):
        super(sap_conv, self).__init__(**kwargs)

    def build(self, input_shape):
        super(sap_conv, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        r = tf.check_numerics(x, "okay")
        p = keras.backend.abs(r) / tf.reduce_sum(keras.backend.abs(r), axis=[1,2,3], keep_dims=True)
        w, h, c = p.get_shape().as_list()[1:]
        N = w * h * c * 2
        p_keep = 1 - tf.exp(-N * p)
        rand = keras.backend.random_uniform(keras.backend.shape(p_keep))
        keep = rand < p_keep
        r = keras.backend.cast(keep, tf.float32) * r / (p_keep + 1e-8)
        r = tf.check_numerics(r, "OH NO")
        return r

# define random projection layer (L_inf version)
class Projection_l_inf(Layer):

    def __init__(self, bound, **kwargs):
        self.bound = bound
        super(sap_dense, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Mask_Layer, self).build(input_shape)  # Be sure to call this somewhere!

        # create random perturbation



def load_cifar_data(one_hot=True, scale1=True):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    if one_hot:
        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_test = np_utils.to_categorical(Y_test, 10)
    else:
        Y_train = np.reshape(Y_train, (Y_train.shape[0],))
        Y_test = np.reshape(Y_test, (Y_test.shape[0],))


    if scale1:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

    return X_train, X_test, Y_train, Y_test




def generate_dense_conv_noise_model(stddev = 0.01):
    model = Sequential()

    model.add(Conv2D_noise(64, (3, 3), stddev,
                     input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))
    model.add(Conv2D_noise(64, (3, 3), stddev))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D_noise(128, (3, 3), stddev))
    model.add(Activation('relu'))
    model.add(Conv2D_noise(128, (3, 3), stddev))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense_noise(stddev=stddev,units=256))
    model.add(Activation('relu'))
    # model.add(Dropout(0.7))
    model.add(Dense_noise(stddev=stddev,units=256))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))

    return model

def generate_single_model():

    model = Sequential()

    model.add(Conv2D(64, (3, 3),
                      input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    #model.add(Dropout(0.7))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))

    return model


def generate_dropout_model(dropout_rate):
    model = Sequential()

    model.add(Conv2D(64, (3, 3),
                      input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))

    return model


def generate_dropout_model_v2(dropout_rate):
    model = Sequential()

    model.add(Conv2D(64, (3, 3),
                      input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(nb_classes))

    return model



def generate_sap_model():

    model = Sequential()

    model.add(Conv2D(64, (3, 3),
                      input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(sap_dense())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))

    return model

def generate_sap_all():
    model = Sequential()

    model.add(Conv2D(64, (3, 3),
                      input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))
    model.add(sap_conv())
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(sap_conv())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(sap_conv())
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(sap_conv())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(sap_dense())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(sap_dense())
    model.add(Dense(nb_classes))

    return model

def generate_resizing_model(pad_value=0.5, scale=0.8):

    model = Sequential()
    model.add(Resize_general(pad_value=pad_value, scale=scale, input_shape=(img_rows, img_cols, channels)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))

    return model

def generate_resize_noise_model(init=0.02, inner=0.01, pad_value=0.5, scale=0.8):
    model = Sequential()
    model.add(Resize_general(pad_value=pad_value, scale=scale, input_shape=(img_rows, img_cols, channels)))
    model.add(GaussianNoise(init))
    model.add(Conv2D(64, (3, 3),))
    model.add(Activation('relu'))
    # model.add(Resize_general(pad_value=pad_value, scale=scale))
    model.add(GaussianNoise(inner))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Resize_general(pad_value=pad_value, scale=scale))
    model.add(GaussianNoise(inner))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    # model.add(Resize_general(pad_value=pad_value, scale=scale))
    model.add(GaussianNoise(inner))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(GaussianNoise(inner))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))

    return model


def generate_switch_model(nb_submodels, load_submodels_weights_path=None, load_upper_model_weights=None):

    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []

    for i in range(nb_submodels):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(64, (3, 3),
                          input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(256))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.7))

        model0.add(Dense(256))
        model0.add(Activation('relu'))

        model0.add(Dense(nb_classes))

        if load_submodels_weights_path:
            model0.load_weights('./CIFAR_regulars/cifar_cnn_%2d.h5' % i)

        output0 = model0.layers[15].output

        submodel_outputs.append(output0)

        output_shape = model0.layers[15].output_shape

    # top model
    top_model = Sequential()
    top_model.add(Reshape((nb_submodels, output_shape[1]), input_shape=(output_shape[1] * nb_submodels,)))
    top_model.add(Mask_Layer(nb_submodels))
    top_model.add(Dense(256))
    top_model.add(Activation('relu'))
    top_model.add(Dense(nb_classes))

    # combine model
    big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
        keras.layers.concatenate(submodel_outputs)))

    return big_model

# # define switching model using different input transformations
# def generate_trans_switch_model():
#     # common input
#     Input = InputLayer(input_shape=(img_rows, img_cols, channels))
#
#     # generate sub_transformations, add all transformations, get the list of outputs
#     subtrans_outputs = []
#
#     output0 = Bit_Depth_Layer_CIFAR(depth=depth, input_shape=(img_rows, img_cols, channels)
#     subtrans_outputs.append(output0)
#
#
#     output1 = Jpeg_Layer_CIFAR(input_shape=(img_rows, img_cols, channels))
#     subtrans_outputs.append(output1)
#
#     output2 = Resize_general(pad_value=pad_value, scale=scale, input_shape=(img_rows, img_cols, channels))
#     subtrans_outputs.append(output2)
#
#     output3 = Crop_Layer_CIFAR(crop_size = crop_size , ensemble_size = ensemble_size , input_shape=(img_rows, img_cols, channels))
#     subtrans_outputs.append(output3)
#
#     output4 = TV_Denoising_Layer_CIFAR(keep_prob = keep_prob, lambda_tv = lambda_tv, input_shape=(img_rows, img_cols, channels))
#     subtrans_outputs.append(output4)
#
#     model = Sequential()
#     model.add(Mask_Layer(subtrans_outputs))
#
#     # output_shape = model.Inputlayer.output_shape
#
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(Conv2D(64, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(128, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(Conv2D(128, (3, 3)))
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dense(256))
#     model.add(Activation('relu'))
#     model.add(Dense(nb_classes))
#
#     # combine model
#     big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
#         keras.layers.concatenate(subtrans_outputs)))
#
#     return model




def generate_switch_model_v3(nb_subs, load_subs=None, load_upper=None):
    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []

    for i in range(load_subs):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(64, (3, 3),
                          input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(256))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.7))

        model0.add(Dense(256))
        model0.add(Activation('relu'))

        model0.add(Dense(nb_classes))

        if load_subs:
            model0.load_weights(load_subs + 'cifar_cnn_%2d.h5' % i)

        output0 = model0.layers[14].output

        submodel_outputs.append(output0)

        output_shape = model0.layers[14].output_shape

    # top model
    top_model = Sequential()
    top_model.add(Reshape((load_subs, output_shape[1]), input_shape=(output_shape[1] * load_subs,)))
    top_model.add(Mask_Layer(load_subs))
    top_model.add(Dense(256))
    top_model.add(Activation('relu'))
    top_model.add(Dense(nb_classes))

    if load_upper:
        top_model.load_weights(load_upper + 'upper0.h5')
    # combine model
    big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
        keras.layers.concatenate(submodel_outputs)))

    return big_model






# def generate_h_switch_model(nb_low, nb_up, load_submodels_weights_path=None, load_upper_weights_path=None):
#
#     # common input
#     Input = InputLayer(input_shape=(img_rows, img_cols, channels))
#
#     # generate submodels, loop over submodels, get the list of outputs
#     submodel_outputs = []
#     upper_outputs = []
#
#     for i in range(nb_low):
#         model0 = Sequential()
#
#         model0.add(Input)
#         model0.add(Conv2D(64, (3, 3),
#                          input_shape=(img_rows, img_cols, channels)))
#         model0.add(Activation('relu'))
#         model0.add(Conv2D(64, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model0.add(Conv2D(128, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(Conv2D(128, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model0.add(Flatten())
#         model0.add(Dense(256))
#         model0.add(Activation('relu'))
#         model0.add(Dropout(0.3))
#         model0.add(Dense(256))
#         model0.add(Activation('relu'))
#         model0.add(Dense(nb_classes))
#
#         if load_submodels_weights_path:
#             model0.load_weights(load_submodels_weights_path + 'cifar_cnn_%2d.h5' % i)
#
#         output0 = model0.layers[15].output
#
#         submodel_outputs.append(output0)
#
#         output_shape = model0.layers[15].output_shape
#
#     for i in range(nb_up):
#         # top model
#         top_model = Sequential()
#         top_model.add(Reshape((nb_low, output_shape[1]), input_shape=(output_shape[1] * nb_low,)))
#         top_model.add(Mask_Layer(nb_low))
#         top_model.add(Dense(256))
#         top_model.add(Activation('relu'))
#         top_model.add(Dense(nb_classes))
#
#         if load_upper_weights_path:
#             top_model.load_weights(load_upper_weights_path + 'upper%d.h5' % i)
#         # combine model
#         big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
#             keras.layers.concatenate(submodel_outputs)))
#
#         upper_output = big_model.output
#         upper_outputs.append(upper_output)
#
#     # final output
#     out_model = Sequential()
#     out_model.add(Reshape((nb_up, 10), input_shape=(10*nb_up,)))
#     out_model.add(Mask_Layer(nb_up))
#
#     total_model = keras.models.Model(inputs=Input.input, outputs=out_model(keras.layers.concatenate(upper_outputs)))
#
#
#     return total_model
#
#
# def generate_h_switch_model_v2(nb_low, nb_up, load_submodels_weights_path=None, load_upper_weights_path=None):
#     # common input
#     Input = InputLayer(input_shape=(img_rows, img_cols, channels))
#
#     # generate submodels, loop over submodels, get the list of outputs
#     submodel_outputs = []
#     upper_outputs = []
#
#     for i in range(nb_low):
#         model0 = Sequential()
#
#         model0.add(Input)
#         model0.add(Conv2D(64, (3, 3),
#                           input_shape=(img_rows, img_cols, channels)))
#         model0.add(Activation('relu'))
#         model0.add(Conv2D(64, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model0.add(Conv2D(128, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(Conv2D(128, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model0.add(Flatten())
#         model0.add(Dense(256))
#         model0.add(Activation('relu'))
#         model0.add(Dropout(0.3))
#         model0.add(Dense(256))
#         model0.add(Activation('relu'))
#         model0.add(Dense(nb_classes))
#
#         if load_submodels_weights_path:
#             model0.load_weights(load_submodels_weights_path + 'cifar_cnn_%2d.h5' % i)
#
#         output0 = model0.layers[11].output
#
#         submodel_outputs.append(output0)
#
#         output_shape = model0.layers[11].output_shape
#
#     for i in range(nb_up):
#         # top model
#         top_model = Sequential()
#         top_model.add(Reshape((nb_low, output_shape[1]), input_shape=(output_shape[1] * nb_low,)))
#         top_model.add(Mask_Layer(nb_low))
#         top_model.add(Dense(256))
#         top_model.add(Activation('relu'))
#         top_model.add(Dropout(0.3))
#         top_model.add(Dense(256))
#         top_model.add(Activation('relu'))
#         top_model.add(Dense(nb_classes))
#
#         if load_upper_weights_path:
#             top_model.load_weights(load_upper_weights_path + 'upper%d.h5' % i)
#         # combine model
#         big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
#             keras.layers.concatenate(submodel_outputs)))
#
#         upper_output = big_model.output
#         upper_outputs.append(upper_output)
#
#     # final output
#     out_model = Sequential()
#     out_model.add(Reshape((nb_up, 10), input_shape=(10 * nb_up,)))
#     out_model.add(Mask_Layer(nb_up))
#
#     total_model = keras.models.Model(inputs=Input.input, outputs=out_model(keras.layers.concatenate(upper_outputs)))
#
#     return total_model
#
#
# def generate_h_switch_model_v3(nb_low, nb_up, load_submodels_weights_path=None, load_upper_weights_path=None):
#     # common input
#     Input = InputLayer(input_shape=(img_rows, img_cols, channels))
#
#     # generate submodels, loop over submodels, get the list of outputs
#     submodel_outputs = []
#     upper_outputs = []
#
#     for i in range(nb_low):
#         model0 = Sequential()
#
#         model0.add(Input)
#         model0.add(Conv2D(64, (3, 3),
#                           input_shape=(img_rows, img_cols, channels)))
#         model0.add(Activation('relu'))
#         model0.add(Conv2D(64, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model0.add(Conv2D(128, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(Conv2D(128, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model0.add(Flatten())
#         model0.add(Dense(256))
#         model0.add(Activation('relu'))
#         model0.add(Dropout(0.7))
#         model0.add(Dense(256))
#         model0.add(Activation('relu'))
#         model0.add(Dense(nb_classes))
#
#         if load_submodels_weights_path:
#             model0.load_weights(load_submodels_weights_path + 'cifar_cnn_%2d.h5' % i)
#
#         output0 = model0.layers[14].output
#
#         submodel_outputs.append(output0)
#
#         output_shape = model0.layers[14].output_shape
#
#     for i in range(nb_up):
#         # top model
#         top_model = Sequential()
#         top_model.add(Reshape((nb_low, output_shape[1]), input_shape=(output_shape[1] * nb_low,)))
#         top_model.add(Mask_Layer(nb_low))
#         top_model.add(Dense(256))
#         top_model.add(Activation('relu'))
#         top_model.add(Dense(nb_classes))
#
#         if load_upper_weights_path:
#             top_model.load_weights(load_upper_weights_path + 'upper%d.h5' % i)
#         # combine model
#         big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
#             keras.layers.concatenate(submodel_outputs)))
#
#         upper_output = big_model.output
#         upper_outputs.append(upper_output)
#
#     # final output
#     out_model = Sequential()
#     out_model.add(Reshape((nb_up, 10), input_shape=(10 * nb_up,)))
#     out_model.add(Mask_Layer(nb_up))
#
#     total_model = keras.models.Model(inputs=Input.input, outputs=out_model(keras.layers.concatenate(upper_outputs)))
#
#     return total_model
#
#
def generate_h_switch_model_v4(nb_low, nb_up, load_submodels_weights_path=None, load_upper_weights_path=None):
    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []
    upper_outputs = []

    for i in range(nb_low):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(64, (3, 3),
                          input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(256))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.7))
        model0.add(Dense(256))
        model0.add(Activation('relu'))
        model0.add(Dense(nb_classes))

        if load_submodels_weights_path:
            model0.load_weights(load_submodels_weights_path + 'cifar_cnn_%2d.h5' % i)

        output0 = model0.layers[13].output

        submodel_outputs.append(output0)

        output_shape = model0.layers[13].output_shape

    for i in range(nb_up):
        # top model
        top_model = Sequential()
        top_model.add(Reshape((nb_low, output_shape[1]), input_shape=(output_shape[1] * nb_low,)))
        top_model.add(Mask_Layer(nb_low))
        top_model.add(Dense(256))
        top_model.add(Activation('relu'))
        top_model.add(Dense(nb_classes))

        if load_upper_weights_path:
            top_model.load_weights(load_upper_weights_path + 'upper%d.h5' % i)
        # combine model
        big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
            keras.layers.concatenate(submodel_outputs)))

        upper_output = big_model.output
        upper_outputs.append(upper_output)

    # final output
    out_model = Sequential()
    out_model.add(Reshape((nb_up, 10), input_shape=(10 * nb_up,)))
    out_model.add(Mask_Layer(nb_up))

    total_model = keras.models.Model(inputs=Input.input, outputs=out_model(keras.layers.concatenate(upper_outputs)))

    return total_model
#
#
# def generate_h_switch_model_v3_no_dropout(nb_low, nb_up, load_submodels_weights_path=None, load_upper_weights_path=None):
#     # common input
#     Input = InputLayer(input_shape=(img_rows, img_cols, channels))
#
#     # generate submodels, loop over submodels, get the list of outputs
#     submodel_outputs = []
#     upper_outputs = []
#
#     for i in range(nb_low):
#         model0 = Sequential()
#
#         model0.add(Input)
#         model0.add(Conv2D(64, (3, 3),
#                           input_shape=(img_rows, img_cols, channels)))
#         model0.add(Activation('relu'))
#         model0.add(Conv2D(64, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model0.add(Conv2D(128, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(Conv2D(128, (3, 3)))
#         model0.add(Activation('relu'))
#         model0.add(MaxPooling2D(pool_size=(2, 2)))
#
#         model0.add(Flatten())
#         model0.add(Dense(256))
#         model0.add(Activation('relu'))
#         #model0.add(Dropout(0.7))
#         model0.add(Dense(256))
#         model0.add(Activation('relu'))
#         model0.add(Dense(nb_classes))
#
#         if load_submodels_weights_path:
#             model0.load_weights(load_submodels_weights_path + 'cifar_cnn_%2d.h5' % i)
#
#         # care_layer = model0.layers[0]
#         output0 = model0.layers[13].output
#
#         submodel_outputs.append(output0)
#
#         output_shape = model0.layers[13].output_shape
#
#     for i in range(nb_up):
#         # top model
#         top_model = Sequential()
#         top_model.add(Reshape((nb_low, output_shape[1]), input_shape=(output_shape[1] * nb_low,)))
#         top_model.add(Mask_Layer(nb_low))
#         top_model.add(Dense(256))
#         top_model.add(Activation('relu'))
#         top_model.add(Dense(nb_classes))
#
#         if load_upper_weights_path:
#             top_model.load_weights(load_upper_weights_path + 'upper%d.h5' % i)
#         # combine model
#         big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
#             keras.layers.concatenate(submodel_outputs)))
#
#         upper_output = big_model.output
#         upper_outputs.append(upper_output)
#
#     # final output
#     out_model = Sequential()
#     out_model.add(Reshape((nb_up, 10), input_shape=(10 * nb_up,)))
#     out_model.add(Mask_Layer(nb_up))
#
#     total_model = keras.models.Model(inputs=Input.input, outputs=out_model(keras.layers.concatenate(upper_outputs)))
#
#     return total_model


def generate_random_projection_model_l_inf(l_inf):

    model = Sequential()

    model.add(Projection_l_inf(bound, input_shape=(img_rows, img_cols, channels)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))

    return model


def generate_avg_model(nb_submodels, merge_at='logit', sub_model_weights_dir='../Model/CIFAR_regular_30/'):
    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []

    for i in range(nb_submodels):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(64, (3, 3),
                          input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(256))
        model0.add(Activation('relu'))
        # model0.add(Dropout(0.7))

        model0.add(Dense(256))
        model0.add(Activation('relu'))

        model0.add(Dense(nb_classes))
        model0.add(Activation('softmax'))

        model0.load_weights(sub_model_weights_dir + 'cifar_cnn_%2d.h5' % i)

        if merge_at == 'logit':
            output0 = model0.layers[16].output
        elif merge_at == 'softmax':
            output0 = model0.layers[17].output
        else:
            print('unrecognized merge_at parameter, use logit instead')
            output0 = model0.layers[16].output

        submodel_outputs.append(output0)

        # output_shape = model0.layers[15].output_shape

    model_output = keras.layers.average(submodel_outputs)
    # model_output_softmax = keras.layers.Activation('softmax')(model_output)

    avg_model = keras.models.Model(inputs=Input.input, outputs=model_output)

    return avg_model


def generate_gaussian_model(init=0.02, inner=0.01):
    model = Sequential()

    model.add(GaussianNoise(init, input_shape=(img_rows, img_cols, channels)))
    model.add(Conv2D(64, (3, 3),))
    model.add(Activation('relu'))
    model.add(GaussianNoise(inner))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GaussianNoise(inner))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(GaussianNoise(inner))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))

    return model


def generate_gaussian_dropout_model(init=0.2, inner=0.1):
    model = Sequential()

    model.add(GaussianNoise(init, input_shape=(img_rows, img_cols, channels)))
    model.add(Conv2D(64, (3, 3),))
    model.add(Activation('relu'))
    model.add(GaussianNoise(inner))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GaussianNoise(inner))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(GaussianNoise(inner))
    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))

    return model


def generate_hsw_v3_fix(nb_low, nb_up,lower_folder='../Model/CIFAR_regular_30/', upper_folder='../Model/CIFAR_uppers_v3/'):
    lower_path = lower_folder + 'cifar_cnn_ 0.h5'
    upper_path = upper_folder + '%d/' % nb_up + 'upper0.h5'

    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    model0 = Sequential()

    model0.add(Input)
    model0.add(Conv2D(64, (3, 3),
                      input_shape=(img_rows, img_cols, channels)))
    model0.add(Activation('relu'))
    model0.add(Conv2D(64, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(MaxPooling2D(pool_size=(2, 2)))

    model0.add(Conv2D(128, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(Conv2D(128, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(MaxPooling2D(pool_size=(2, 2)))

    model0.add(Flatten())
    model0.add(Dense(256))
    model0.add(Activation('relu'))
    model0.add(Dense(256))
    model0.add(Activation('relu'))
    model0.add(Dense(nb_classes))

    # load lower weights
    model0.load_weights(lower_path)

    # lower output
    output0 = model0.layers[13].output

    # upper model
    top_model = Sequential()
    top_model.add(Dense(256, input_shape=(256,)))
    top_model.add(Activation('relu'))
    top_model.add(Dense(nb_classes))

    # load upper model weights
    top_model.load_weights(upper_path)

    big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
        output0))

    return big_model


def generate_hsw_v4_fix(nb_low, nb_up,lower_folder='../Model/CIFAR_regular_30/', upper_folder='../Model/CIFAR_uppers_v4/'):
    lower_path = lower_folder + 'cifar_cnn_ 0.h5'
    upper_path = upper_folder + '%d/' % nb_up + 'upper0.h5'

    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    model0 = Sequential()

    model0.add(Input)
    model0.add(Conv2D(64, (3, 3),
                      input_shape=(img_rows, img_cols, channels)))
    model0.add(Activation('relu'))
    model0.add(Conv2D(64, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(MaxPooling2D(pool_size=(2, 2)))

    model0.add(Conv2D(128, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(Conv2D(128, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(MaxPooling2D(pool_size=(2, 2)))

    model0.add(Flatten())
    model0.add(Dense(256))
    model0.add(Activation('relu'))
    model0.add(Dense(256))
    model0.add(Activation('relu'))
    model0.add(Dense(nb_classes))

    # load lower weights
    model0.load_weights(lower_path)

    # lower output
    output0 = model0.layers[12].output

    # upper model
    top_model = Sequential()
    top_model.add(Dense(256, input_shape=(256,)))
    top_model.add(Activation('relu'))
    top_model.add(Dense(nb_classes))

    # load upper model weights
    top_model.load_weights(upper_path)

    big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
        output0))

    return big_model


def generate_hrs3(n_channels=(5,5,5), b0_weights='../Model/CIFAR_regular_30/', b1_weights='../Model/CIFAR_hrs3/b1/5/',
                  b2_weights='../Model/CIFAR_hrs3/b2/5_5/'):
    n_b0 = n_channels[0]
    n_b1 = n_channels[1]
    n_b2 = n_channels[2]
    b0_weights_path = b0_weights
    b0_file_prefix = 'cifar_cnn_%2d.h5'
    b1_weights_path = b1_weights
    b1_file_prefix = 'b1_%d.h5'
    b2_weights_path = b2_weights
    b2_file_prefix = 'b2_%d.h5'

    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # block 0
    b0_outputs = []
    for i in range(n_b0):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(64, (3, 3),
                          input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(256))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.7))
        model0.add(Dense(256))
        model0.add(Activation('relu'))
        model0.add(Dense(nb_classes))

        if b0_weights_path:
            model0.load_weights(b0_weights_path + b0_file_prefix % i)

        output0 = model0.layers[10].output
        b0_outputs.append(output0)
        b0_output_shape = model0.layers[10].output_shape

        # freeze submodel weights
        for layer in model0.layers:
            layer.trainable = False
    # random mask
    b0_c = keras.layers.concatenate(b0_outputs)
    b0_o_reshape = Reshape((n_b0, b0_output_shape[1]))(b0_c)
    b0_o_mask = Mask_Layer(n_b0)(b0_o_reshape)

    # block 1
    b1_outputs = []
    for i in range(n_b1):
        b1 = Sequential()
        b1.add(Dense(256, input_shape=(3200,)))
        b1.add(Activation('relu'))
        b1.add(Dropout(0.7))
        b1.add(Dense(256))
        b1.add(Activation('relu'))
        b1.add(Dense(nb_classes))
        b1.load_weights(b1_weights_path + b1_file_prefix % i)
        # freeze submodel weights
        for layer in b1.layers:
            layer.trainable = False

        bb1 = Sequential()
        bb1.add(b1.layers[0])
        bb1.add(b1.layers[1])
        bb1.add(b1.layers[2])

        # b1.inputs[0] = b0_o_mask
        # oo = b1(b0_o_mask)

        # output1 = b1.layers[2].output
        output1 = bb1(b0_o_mask)
        b1_outputs.append(output1)
        b1_output_shape = b1.layers[2].output_shape
    # random mask
    b1_c = keras.layers.concatenate(b1_outputs)
    b1_o_reshape = Reshape((n_b1, b1_output_shape[1]))(b1_c)
    b1_o_mask = Mask_Layer(n_b1)(b1_o_reshape)

    # block 2
    b2_outputs = []
    for i in range(n_b2):
        b2 = Sequential()
        b2.add(Dense(256, input_shape=(256,)))
        b2.add(Activation('relu'))
        b2.add(Dense(nb_classes))
        # load_weight
        b2.load_weights(b2_weights_path + b2_file_prefix % i)
        # get output
        output2 = b2(b1_o_mask)
        b2_outputs.append(output2)
    b2_c = keras.layers.concatenate(b2_outputs)
    b2_o_reshape = Reshape((n_b2, 10))(b2_c)
    b2_o_mask = Mask_Layer(n_b2)(b2_o_reshape)

    # the switching model
    hrs = keras.models.Model(inputs=Input.input, outputs=b2_o_mask)

    return hrs







def generate_model_switch(n, weights_forlder='../Model/CIFAR_regular_30/'):
    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []

    for i in range(n):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(64, (3, 3),
                          input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(256))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.7))

        model0.add(Dense(256))
        model0.add(Activation('relu'))

        model0.add(Dense(nb_classes))

        if weights_forlder:
            model0.load_weights(weights_forlder + 'cifar_cnn_%2d.h5' % i)

        output0 = model0.layers[16].output

        submodel_outputs.append(output0)

        output_shape = model0.layers[16].output_shape

    # top model
    top_model = Sequential()
    top_model.add(Reshape((n, output_shape[1]), input_shape=(output_shape[1] * n,)))
    top_model.add(Mask_Layer(n))

    # combine model
    big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
        keras.layers.concatenate(submodel_outputs)))

    return big_model


def generate_model_switch_strong_weak(n=2):
    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []

    for i in range(n):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(64, (3, 3),
                          input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(256))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.7))

        model0.add(Dense(256))
        model0.add(Activation('relu'))

        model0.add(Dense(nb_classes))

        if i == 0:
            model0.load_weights('../Model/CIFAR_regular_30/cifar_cnn_ 0.h5')
        elif i == 1:
            model0.load_weights('../Model/adv_train_ICCAD/cifar_noaug_wd_eps8_0')

        output0 = model0.layers[16].output

        submodel_outputs.append(output0)

        output_shape = model0.layers[16].output_shape

    # top model
    top_model = Sequential()
    top_model.add(Reshape((n, output_shape[1]), input_shape=(output_shape[1] * n,)))
    top_model.add(Mask_Layer(n))

    # combine model
    big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
        keras.layers.concatenate(submodel_outputs)))

    return big_model



def generate_model_switch_adv(n, weights_forlder, file_prefix):
    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []

    for i in range(n):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(64, (3, 3),
                          input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(128, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(256))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.7))

        model0.add(Dense(256))
        model0.add(Activation('relu'))

        model0.add(Dense(nb_classes))

        if weights_forlder:
            model0.load_weights(weights_forlder + file_prefix + '_%d' % i)

        output0 = model0.layers[16].output

        submodel_outputs.append(output0)

        output_shape = model0.layers[16].output_shape

    # top model
    top_model = Sequential()
    top_model.add(Reshape((n, output_shape[1]), input_shape=(output_shape[1] * n,)))
    top_model.add(Mask_Layer(n))

    # combine model
    big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
        keras.layers.concatenate(submodel_outputs)))

    return big_model


def choose_defense_model(defense_model_indicator):

    if defense_model_indicator == 'single':
        # generate sigle model
        keras_model = generate_single_model()
        # load weights
        print(os.getcwd())
        keras_model.load_weights('../Model/CIFAR_regular_30/cifar_cnn_ 0.h5')
        keras.backend.set_learning_phase(0)
        return keras_model
    elif defense_model_indicator.startswith('which_single'):
        [_, idx] = defense_model_indicator.split('[')
        idx = idx[:-1]
        idx = int(idx)
        keras.backend.set_learning_phase(0)
        # generate sigle model
        keras_model = generate_single_model()
        # load weights
        keras_model.load_weights('../Model/CIFAR_regular_30/cifar_cnn_ %d.h5' % idx)
        return keras_model
    elif defense_model_indicator == 'sap':
        keras.backend.set_learning_phase(1)
        keras_model = generate_sap_model()
        keras_model.load_weights('../Model/CIFAR_regulars/cifar_cnn_ 0.h5')
        return keras_model
    elif defense_model_indicator == 'sap_all':
        keras.backend.set_learning_phase(1)
        keras_model = generate_sap_all()
        keras_model.load_weights('../Model/CIFAR_regular_30/cifar_cnn_ 0.h5')
        return keras_model
    elif defense_model_indicator.startswith('dropout'):
        keras.backend.set_learning_phase(1)
        [_, dropout_rate] = defense_model_indicator.split('[')
        dropout_rate = float(dropout_rate[:-1])
        keras_model = generate_dropout_model(dropout_rate)
        keras_model.load_weights('../Model/CIFAR_regular_30/cifar_cnn_ 0.h5')
        print('using dropout model with dropout rate of %f' % dropout_rate)
        return keras_model
    elif defense_model_indicator.startswith('dropout_v2'):
        keras.backend.set_learning_phase(1)
        [_, dropout_rate] = defense_model_indicator.split('[')
        dropout_rate = float(dropout_rate[:-1])
        keras_model = generate_dropout_model_v2(dropout_rate)
        keras_model.load_weights('../Model/CIFAR_regular_30/cifar_cnn_ 0.h5')
        print('using dropout model with dropout rate of %f' % dropout_rate)
        return keras_model
    elif defense_model_indicator.startswith('traindropout'):
        keras.backend.set_learning_phase(1)
        [_, train_rate, test_rate] = defense_model_indicator.split('[')
        train_rate = float(train_rate[:-1])
        test_rate = float(test_rate[:-1])
        keras_model = generate_dropout_model(test_rate)
        keras_model.load_weights('../Model/CIFAR_regular_train_dropout/CIFAR_train_%.2f.h5' % train_rate)
        print('using dropout model with train rate of %.2f and test rate of %.2f' % (train_rate, test_rate))
        return keras_model
    elif defense_model_indicator.startswith('sw'):
        keras.backend.set_learning_phase(0)
        [_, nb_channels] = defense_model_indicator.split('[')
        nb_channels = int(nb_channels[:-1])
        keras_model = generate_switch_model(nb_channels)
        keras_model.load_weights('../Model/CIFAR_switch_30/switch_model_%d' % nb_channels)
        print('using sw model with %d channels' % nb_channels)
        return keras_model
    elif defense_model_indicator.startswith('hsw'):
        keras.backend.set_learning_phase(0)
        # check if has enough parameters
        [_, low, up] = defense_model_indicator.split('[')
        low = int(low[:-1])
        up = int(up[:-1])
        keras_model = generate_h_switch_model_v4(low, up, '../Model/CIFAR_regular_30/', '../Model/CIFAR_uppers_v4/%d/' % low)
        print('using hsw model with %d lowers and %d uppers' % (low, up))
        return keras_model
    elif defense_model_indicator.startswith('3hrs'):
        keras.backend.set_learning_phase(0)
        # check if has enough parameters
        [_, b0_c, b1_c, b2_c] = defense_model_indicator.split('[')
        b0_c = int(b0_c[:-1])
        b1_c = int(b1_c[:-1])
        b2_c = int(b2_c[:-1])
        keras_model = generate_hrs3((b0_c, b1_c, b2_c))
        print('using 3-block hrs with (%d, %d, %d) channels' % (b0_c, b1_c, b2_c))
        return keras_model
    elif defense_model_indicator.startswith('IJCAI_hsw_data'):
        keras.backend.set_learning_phase(0)
        [_, percentage] = defense_model_indicator.split('[')
        percentage = float(percentage[:-1])
        if percentage == 0.5:
            keras_model = generate_h_switch_model_v4(5, 5, '../Model/CIFAR_halfdata_5/',
                                                 '../Model/CIFAR_partial_data/CIFAR_uppers_50/%d/' % 5)
        elif percentage == 0.2:
            keras_model = generate_h_switch_model_v4(5, 5, '../Model/CIFAR_halfdata_2/',
                                                 '../Model/CIFAR_partial_data/CIFAR_uppers_20/%d/' % 5)
        print('using hsw model with %d lowers and %d uppers' % (5, 5))
        return keras_model

    elif defense_model_indicator.startswith('fix_hsw'):
        keras.backend.set_learning_phase(0)
        # check if has enough parameters
        [_, low, up] = defense_model_indicator.split('[')
        low = int(low[:-1])
        up = int(up[:-1])
        keras_model = generate_hsw_v4_fix(low, up)
        print('using fix hsw model with %d lowers and %d uppers' % (low, up))
        return keras_model
    # elif defense_model_indicator.startswith('ndhsw'):
    #     keras.backend.set_learning_phase(0)
    #     # check if has enough parameters
    #     [_, low, up] = defense_model_indicator.split('[')
    #     low = int(low[:-1])
    #     up = int(up[:-1])
    #     keras_model = generate_h_switch_model_v3_no_dropout(low, up, '../Model/CIFAR_regular_30/', '../Model/CIFAR_uppers_v3/%d/' % low)
    #     print('using hsw no dropout model with %d lowers and %d uppers' % (low, up))
    #     return keras_model
    elif defense_model_indicator.startswith('avg'):
        # input eg: 'avg[10]'
        keras.backend.set_learning_phase(0)
        [_, channels] = defense_model_indicator.split('[')
        channels = int(channels[:-1])
        keras_model = generate_avg_model(channels)
        print('using avg model with %d channels' % channels)
        return keras_model
    elif defense_model_indicator.startswith('gaussian'):
        # input eg: 'gaussian'
        [_, init, inner] = defense_model_indicator.split('[')
        init = float(init[:-1])
        inner = float(inner[:-1])
        keras.backend.set_learning_phase(1)
        keras_model = generate_gaussian_model(init=init, inner=inner)
        keras_model.load_weights('../Model/CIFAR_gaussian/gaussian[%.4f][%.4f]' % (init, inner))
        print('using gaussian model ')
        return keras_model
    elif defense_model_indicator.startswith('fix_gaussian'):
        # input eg: 'gaussian'
        [_, init, inner] = defense_model_indicator.split('[')
        init = float(init[:-1])
        inner = float(inner[:-1])
        keras.backend.set_learning_phase(0)
        keras_model = generate_gaussian_model(init=init, inner=inner)
        keras_model.load_weights('../Model/CIFAR_gaussian/gaussian[%.4f][%.4f]' % (init, inner))
        print('using fix gaussian model ')
        return keras_model
    elif defense_model_indicator.startswith('model_switch'):
        keras.backend.set_learning_phase(0)
        [_, n] = defense_model_indicator.split('[')
        n = int(n[:-1])
        keras_model = generate_model_switch(n)
        print('using model switching with %d models' % n)
        return keras_model
    elif defense_model_indicator.startswith('adv_model_switch'):
        # e.g. adv_model_switch[epsilon][n]
        # note epsilon 8 means 8/255
        keras.backend.set_learning_phase(0)
        [_, eps, n] = defense_model_indicator.split('[')
        eps = int(eps[:-1])
        n = int(n[:-1])
        weights_forlder = '../Model/adv_train_ICCAD/'
        file_prefix = 'cifar_noaug_wd_eps%d' % eps
        keras_model = generate_model_switch_adv(n, weights_forlder, file_prefix)
        print('using adv model switching. Epsilon = %d , n = %d' % (eps, n))
        return keras_model


    elif defense_model_indicator == 'resizing_model':
        keras.backend.set_learning_phase(0)
        keras_model = generate_resizing_model()
        keras_model.load_weights('../Model/CIFAR_regulars/cifar_cnn_ 0.h5')
        return keras_model
    elif defense_model_indicator.startswith('resize'):
        [_, r] = defense_model_indicator.split('[')
        r = float(r[:-1])
        keras_model = generate_resizing_model(scale=r)
        keras_model.load_weights('../Model/CIFAR_regulars/cifar_cnn_ 0.h5')
        keras.backend.set_learning_phase(0)
        return keras_model
    elif defense_model_indicator.startswith('weights_noise'):
        [_, dev] = defense_model_indicator.split('[')
        dev = float(dev[:-1])
        keras.backend.set_learning_phase(0)
        keras_model = generate_dense_conv_noise_model(dev)
        keras_model.load_weights('../Model/CIFAR_regulars/cifar_cnn_ 0.h5')
        return keras_model
    elif defense_model_indicator.startswith('adv_train'):
        [_, dir] = defense_model_indicator.split('[')
        dir = dir[:-1]
        # generate sigle model
        keras_model = generate_single_model()
        # laod weights
        keras_model.load_weights(dir)
        keras.backend.set_learning_phase(0)
        return keras_model
    elif defense_model_indicator.startswith('ICCAD_adv_train'):
        [_, eps] = defense_model_indicator.split('[')
        eps = int(eps[:-1])
        keras.backend.set_learning_phase(0)
        # generate sigle model
        keras_model = generate_single_model()
        # laod weights
        weights_path = '../Model/adv_train_ICCAD/cifar_noaug_wd_eps%d_0' % eps
        keras_model.load_weights(weights_path)
        print('Using ICCAD adv train model. Epsilon = %d' % eps)
        return keras_model
    elif defense_model_indicator == 'nips_ms_special':
        keras.backend.set_learning_phase(0)
        keras_model = generate_model_switch_strong_weak()
        print('using nips special model switch model ...')
        return keras_model
    else:
        print('Model indicator unrecognized.')

    # elif defense_model_indicator.startswith('transform_model'):
    #     keras.backend.set_learning_phase(0)
    #     keras_model = generate_trans_switch_model()
    #     keras_model.load_weights('../Model/CIFAR_regulars/cifar_cnn_ 0.h5')
    #     return keras_model


def choose_generate_model(defense_model_indicator):
    if defense_model_indicator in ('single', 'sap') or defense_model_indicator.startswith('dropout'):
        return choose_defense_model('single')
    elif defense_model_indicator.startswith('gaussian'):
        return choose_defense_model('fix_' + defense_model_indicator)
    elif defense_model_indicator.startswith('hsw'):
        return choose_defense_model('fix_' + defense_model_indicator)


def choose_fixed_randomness_model(defense_model_indicator):
    if defense_model_indicator == 'single':
        # generate sigle model
        keras_model = generate_single_model()
        # load weights
        print(os.getcwd())
        keras_model.load_weights('../Model/CIFAR_regular_30/cifar_cnn_ 0.h5')
        keras.backend.set_learning_phase(0)
        return keras_model
    elif defense_model_indicator.startswith('which_single'):
        [_, idx] = defense_model_indicator.split('[')
        idx = idx[:-1]
        idx = int(idx)
        keras.backend.set_learning_phase(0)
        # generate sigle model
        keras_model = generate_single_model()
        # load weights
        keras_model.load_weights('../Model/CIFAR_regular_30/cifar_cnn_ %d.h5' % idx)
        return keras_model
    elif defense_model_indicator.startswith('model_switch'):
        keras.backend.set_learning_phase(0)
        [_, n] = defense_model_indicator.split('[')
        n = int(n[:-1])
        # generate a random number in n
        idx = np.random.randint(n)
        keras_model = generate_single_model()
        keras_model.load_weights('../Model/CIFAR_regular_30/cifar_cnn_ %d.h5' % idx)

        print('FIXED RANDOMNESS!!! : using model switching with %d models' % n)
        print('the %d th single model is sampled ...' % idx)
        return keras_model





















