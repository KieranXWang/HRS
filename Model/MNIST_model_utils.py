# all keras code
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, Reshape, GaussianNoise
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
import tensorflow as tf
import keras
from keras.engine.topology import Layer
from keras.layers.convolutional import _Conv
import numpy as np

# model general parameters
nb_classes = 10
img_rows = 28
img_cols = 28
channels = 1


# define Conv2D + Gaussian noise
class _Conv_noise(_Conv):

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

        self.noise = tf.random_normal(shape=kernel_shape, mean=0.0, stddev=0.01, dtype=tf.float32, seed=None,
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

class Conv2D_noise(_Conv_noise):
    def __init__(self, filters,
                 kernel_size,
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


    def get_config(self):
        config = super(Conv2D_noise, self).get_config()
        config.pop('rank')
        return config

# define Dense layer + Gaussian noise

class Dense_noise(Dense):

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

        self.noise = tf.random_normal(shape=(input_dim, self.units), mean=0.0, stddev=0.01, dtype=tf.float32,
                                      seed=None, name=None)

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
        output = keras.backend.dot(x, self.kernel + self.noise)
        if self.use_bias:
            output = keras.backend.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

# define random resizing and padding layer CIFAR

class Resizing_Layer_MNIST(Layer):
    def __init__(self, pad_value=0.5, **kwargs):
        self.pad_value = pad_value
        super(Resizing_Layer_MNIST, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Resizing_Layer_MNIST, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # self.rescaled = tf.image.crop_and_resize(x, [[0, 0, 1, 1]], [0], [self.rnd, self.rnd])
        # x = self.rescaled(x)
        rnd = tf.random_uniform((), 25, 28, dtype=tf.int32)
        rescaled = tf.image.crop_and_resize(x, [[0, 0, 1, 1]], [0], [rnd, rnd])
        h_rem = 28 - rnd
        w_rem = 28 - rnd
        pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                        constant_values=self.pad_value)
        padded.set_shape((None, 28, 28, 1))
        return padded

class Resize_general_MNIST(Layer):
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
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                        constant_values=self.pad_value)
        padded.set_shape((None, shape[1], shape[2], shape[3]))
        return padded



# define mask layer
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
        p = tf.abs(r) / tf.reduce_sum(tf.abs(r), axis=1, keep_dims=True)
        N = sum(p.get_shape().as_list()[1:]) * 2
        p_keep = 1 - tf.exp(-N * p)
        rand = tf.random_uniform(tf.shape(p_keep))
        keep = rand < p_keep
        r = tf.cast(keep, tf.float32) * r / (p_keep + 1e-8)
        r = tf.check_numerics(r, "OH NO")
        return r


def load_mnist_data(one_hot=True, scale1=True):
    # the defualt is 0-255, not one hot coding
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # reshape
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))


    if one_hot:
        Y_train = np_utils.to_categorical(Y_train, 10)
        Y_test = np_utils.to_categorical(Y_test, 10)

    if scale1:
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255

    return X_train, X_test, Y_train, Y_test




def generate_dense_conv_noise_model():
    model = Sequential()

    model.add(Conv2D_noise(32, (3, 3), input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))
    model.add(Conv2D_noise(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D_noise(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D_noise(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense_noise(200))
    model.add(Activation('relu'))
    # model.add(Dropout(0.7))
    model.add(Dense_noise(200))
    model.add(Activation('relu'))
    model.add(Dense_noise(nb_classes))

    return model



def generate_resizing_model():

    model = Sequential()
    model.add(Resizing_Layer_MNIST(input_shape=(img_rows, img_cols, channels)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10))

    return model



def generate_single_model():
    # build model
    model = Sequential()

    model.add(Conv2D(32, (3, 3),
                     input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10))

    return model


def generate_sap_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3),
                     input_shape=(img_rows, img_cols, channels)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(sap_dense())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))

    return model


def generate_dropout_model(drop_rate):
    # build model
    model = Sequential()

    model.add(Conv2D(32, (3, 3),
                     input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(drop_rate))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10))

    return model



def generate_switch_model(nb_submodels, load_submodels_weights_path=None):

    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []

    for i in range(nb_submodels):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(32, (3, 3),
                         input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(32, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.3))
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dense(nb_classes))

        if load_submodels_weights_path:
            model0.load_weights(load_submodels_weights_path + 'MNIST_cnn_%d.h5' % i)

        output0 = model0.layers[15].output

        submodel_outputs.append(output0)

        output_shape = model0.layers[15].output_shape

    # top model
    top_model = Sequential()
    top_model.add(Reshape((nb_submodels, output_shape[1]), input_shape=(output_shape[1] * nb_submodels,)))
    top_model.add(Mask_Layer(nb_submodels))
    top_model.add(Dense(200))
    top_model.add(Activation('relu'))
    top_model.add(Dense(nb_classes))

    # combine model
    big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
        keras.layers.concatenate(submodel_outputs)))

    return big_model


def generate_h_switch_model(nb_low, nb_up, load_submodels_weights_path=None, load_upper_weights_path=None):

    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []
    upper_outputs = []

    for i in range(nb_low):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(32, (3, 3),
                         input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(32, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.3))
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dense(nb_classes))

        if load_submodels_weights_path:
            model0.load_weights(load_submodels_weights_path + 'MNIST_cnn_%2d.h5' % i)

        output0 = model0.layers[15].output

        submodel_outputs.append(output0)

        output_shape = model0.layers[15].output_shape

    for i in range(nb_up):
        # top model
        top_model = Sequential()
        top_model.add(Reshape((nb_low, output_shape[1]), input_shape=(output_shape[1] * nb_low,)))
        top_model.add(Mask_Layer(nb_low))
        top_model.add(Dense(200))
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
    out_model.add(Reshape((nb_up, 10), input_shape=(10*nb_up,)))
    out_model.add(Mask_Layer(nb_up))

    total_model = keras.models.Model(inputs=Input.input, outputs=out_model(keras.layers.concatenate(upper_outputs)))


    return total_model


def generate_h_switch_model_v3(nb_low, nb_up, load_submodels_weights_path=None, load_upper_weights_path=None):

    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []
    upper_outputs = []

    for i in range(nb_low):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(32, (3, 3),
                         input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(32, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.3))
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dense(nb_classes))

        if load_submodels_weights_path:
            model0.load_weights(load_submodels_weights_path + 'MNIST_cnn_%2d.h5' % i)

        output0 = model0.layers[14].output

        submodel_outputs.append(output0)

        output_shape = model0.layers[14].output_shape

    for i in range(nb_up):
        # top model
        top_model = Sequential()
        top_model.add(Reshape((nb_low, output_shape[1]), input_shape=(output_shape[1] * nb_low,)))
        top_model.add(Mask_Layer(nb_low))
        top_model.add(Dense(200))
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
    out_model.add(Reshape((nb_up, 10), input_shape=(10*nb_up,)))
    out_model.add(Mask_Layer(nb_up))

    total_model = keras.models.Model(inputs=Input.input, outputs=out_model(keras.layers.concatenate(upper_outputs)))


    return total_model


def generate_h_switch_model_v4(nb_low, nb_up, load_submodels_weights_path=None, load_upper_weights_path=None):

    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []
    upper_outputs = []

    for i in range(nb_low):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(32, (3, 3),
                         input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(32, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.3))
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dense(nb_classes))

        if load_submodels_weights_path:
            model0.load_weights(load_submodels_weights_path + 'MNIST_cnn_%2d.h5' % i)

        output0 = model0.layers[13].output

        submodel_outputs.append(output0)

        output_shape = model0.layers[13].output_shape

    for i in range(nb_up):
        # top model
        top_model = Sequential()
        top_model.add(Reshape((nb_low, output_shape[1]), input_shape=(output_shape[1] * nb_low,)))
        top_model.add(Mask_Layer(nb_low))
        top_model.add(Dense(200))
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
    out_model.add(Reshape((nb_up, 10), input_shape=(10*nb_up,)))
    out_model.add(Mask_Layer(nb_up))

    total_model = keras.models.Model(inputs=Input.input, outputs=out_model(keras.layers.concatenate(upper_outputs)))


    return total_model


def generate_avg_model(nb_submodels, merge_at='logit', sub_model_weights_dir='../Model/MNIST_regular_30/'):
    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []

    for i in range(nb_submodels):

        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(32, (3, 3),
                         input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(32, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        # model0.add(Dropout(0.3))
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dense(nb_classes))
        # model0.add(Activation('softmax'))

        model0.load_weights(sub_model_weights_dir + 'MNIST_cnn_%2d.h5' % i)

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



def generate_gaussian_model(init=0.2, inner=0.1):
    # build model
    model = Sequential()

    model.add(GaussianNoise(init, input_shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3),))
    model.add(Activation('relu'))
    model.add(GaussianNoise(inner))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GaussianNoise(inner))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(GaussianNoise(inner))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dense(10))

    return model


def generate_hsw_v3_fix(nb_low, nb_up, lower_folder='../Model/MNIST_regular_30/',
                        upper_folder='../Model/MNIST_uppers_v3/'):
    lower_path = lower_folder + 'MNIST_cnn_ 0.h5'
    upper_path = upper_folder + '%d/' % nb_up + 'upper0.h5'

    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    model0 = Sequential()

    model0.add(Input)
    model0.add(Conv2D(32, (3, 3),
                      input_shape=(img_rows, img_cols, channels)))
    model0.add(Activation('relu'))
    model0.add(Conv2D(32, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(MaxPooling2D(pool_size=(2, 2)))

    model0.add(Conv2D(64, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(Conv2D(64, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(MaxPooling2D(pool_size=(2, 2)))

    model0.add(Flatten())
    model0.add(Dense(200))
    model0.add(Activation('relu'))
    model0.add(Dense(200))
    model0.add(Activation('relu'))
    model0.add(Dense(nb_classes))

    # load lower weights
    model0.load_weights(lower_path)

    # lower output
    output0 = model0.layers[13].output

    # upper model
    top_model = Sequential()
    top_model.add(Dense(200, input_shape=(200,)))
    top_model.add(Activation('relu'))
    top_model.add(Dense(nb_classes))

    # load upper model weights
    top_model.load_weights(upper_path)

    big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
        output0))

    return big_model


def generate_hsw_v4_fix(nb_low, nb_up, lower_folder='../Model/MNIST_regular_30/',
                        upper_folder='../Model/MNIST_uppers_v4/'):
    lower_path = lower_folder + 'MNIST_cnn_ 0.h5'
    upper_path = upper_folder + '%d/' % nb_up + 'upper0.h5'

    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    model0 = Sequential()

    model0.add(Input)
    model0.add(Conv2D(32, (3, 3),
                      input_shape=(img_rows, img_cols, channels)))
    model0.add(Activation('relu'))
    model0.add(Conv2D(32, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(MaxPooling2D(pool_size=(2, 2)))

    model0.add(Conv2D(64, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(Conv2D(64, (3, 3)))
    model0.add(Activation('relu'))
    model0.add(MaxPooling2D(pool_size=(2, 2)))

    model0.add(Flatten())
    model0.add(Dense(200))
    model0.add(Activation('relu'))
    model0.add(Dense(200))
    model0.add(Activation('relu'))
    model0.add(Dense(nb_classes))

    # load lower weights
    model0.load_weights(lower_path)

    # lower output
    output0 = model0.layers[12].output

    # upper model
    top_model = Sequential()
    top_model.add(Dense(200, input_shape=(200,)))
    top_model.add(Activation('relu'))
    top_model.add(Dense(nb_classes))

    # load upper model weights
    top_model.load_weights(upper_path)

    big_model = keras.models.Model(inputs=Input.input, outputs=top_model(
        output0))

    return big_model

def generate_model_switch(n, weights_forlder='../Model/MNIST_regular_30/'):
    # common input
    Input = InputLayer(input_shape=(img_rows, img_cols, channels))

    # generate submodels, loop over submodels, get the list of outputs
    submodel_outputs = []

    for i in range(n):
        model0 = Sequential()

        model0.add(Input)
        model0.add(Conv2D(32, (3, 3),
                          input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(32, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.3))

        model0.add(Dense(200))
        model0.add(Activation('relu'))

        model0.add(Dense(nb_classes))

        if weights_forlder:
            model0.load_weights(weights_forlder + 'MNIST_cnn_%2d.h5' % i)

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
        model0.add(Conv2D(32, (3, 3),
                          input_shape=(img_rows, img_cols, channels)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(32, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(Conv2D(64, (3, 3)))
        model0.add(Activation('relu'))
        model0.add(MaxPooling2D(pool_size=(2, 2)))

        model0.add(Flatten())
        model0.add(Dense(200))
        model0.add(Activation('relu'))
        model0.add(Dropout(0.3))

        model0.add(Dense(200))
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
        keras.backend.set_learning_phase(0)
        keras_model = generate_single_model()
        keras_model.load_weights('../Model/MNIST_regular_30/MNIST_cnn_ 0.h5')
        print('using regular model ...')
        return keras_model
    elif defense_model_indicator == 'sap':
        keras.backend.set_learning_phase(0)
        keras_model = generate_sap_model()
        keras_model.load_weights('../Model/MNIST_regulars/MNIST_cnn_ 0.h5')
        print('using sap model ...')
        return keras_model
    elif defense_model_indicator.startswith('dropout'):
        keras.backend.set_learning_phase(1)
        [_, dropout_rate] = defense_model_indicator.split('[')
        dropout_rate = float(dropout_rate[:-1])
        keras_model = generate_dropout_model(dropout_rate)
        keras_model.load_weights('../Model/MNIST_regulars/MNIST_cnn_ 0.h5')
        print('using dropout model with dropout rate of %f' % dropout_rate)
        return keras_model
    elif defense_model_indicator.startswith('traindropout'):
        keras.backend.set_learning_phase(1)
        [_, train_rate, test_rate] = defense_model_indicator.split('[')
        train_rate = float(train_rate[:-1])
        test_rate = float(test_rate[:-1])
        keras_model = generate_dropout_model(test_rate)
        keras_model.load_weights('../Model/MNIST_regular_train_dropout/MNIST_train_%.2f.h5' % train_rate)
        print('using dropout model with train rate of %.2f and test rate of %.2f' % (train_rate, test_rate))
        return keras_model

    elif defense_model_indicator.startswith('sw'):
        keras.backend.set_learning_phase(0)
        [_, nb_channels] = defense_model_indicator.split('[')
        nb_channels = int(nb_channels[:-1])
        keras_model = generate_switch_model(nb_channels)
        keras_model.load_weights('../Model/MNIST_switch_30/switch_model_%d' % nb_channels)
        print('using sw model with %d channels' % nb_channels)
        return keras_model
    elif defense_model_indicator.startswith('hsw'):
        keras.backend.set_learning_phase(0)
        # check if has enough parameters
        [_, low, up] = defense_model_indicator.split('[')
        low = int(low[:-1])
        up = int(up[:-1])
        keras_model = generate_h_switch_model_v4(low, up, '../Model/MNIST_regular_30/', '../Model/MNIST_uppers_v4/%d/' % low)
        print('using hsw model with % d lowers and % d uppers' % (low, up))
        return keras_model
    elif defense_model_indicator.startswith('fix_hsw'):
        keras.backend.set_learning_phase(0)
        # check if has enough parameters
        [_, low, up] = defense_model_indicator.split('[')
        low = int(low[:-1])
        up = int(up[:-1])
        keras_model = generate_hsw_v4_fix(low, up)
        print('using fix hsw model with % d lowers and % d uppers' % (low, up))
        return keras_model
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
        [_ ,init, inner] = defense_model_indicator.split('[')
        init = float(init[:-1])
        inner = float(inner[:-1])
        keras.backend.set_learning_phase(1)
        keras_model = generate_gaussian_model(init=init, inner=inner)
        keras_model.load_weights('../Model/MNIST_gaussian/gaussian[%.4f][%.4f]' % (init, inner))
        print('using gaussian model ')
        return keras_model
    elif defense_model_indicator.startswith('fix_gaussian'):
        # input eg: 'gaussian'
        [_ ,init, inner] = defense_model_indicator.split('[')
        init = float(init[:-1])
        inner = float(inner[:-1])
        keras.backend.set_learning_phase(0)
        keras_model = generate_gaussian_model(init=init, inner=inner)
        keras_model.load_weights('../Model/MNIST_gaussian/gaussian[%.4f][%.4f]' % (init, inner))
        print('using fix gaussian model ')
        return keras_model
    elif defense_model_indicator == 'resizing_model':
        keras.backend.set_learning_phase(0)
        keras_model = generate_resizing_model()
        keras_model.load_weights('../Model/MNIST_regulars/MNIST_cnn_ 0.h5')
        return keras_model
    elif defense_model_indicator == 'weights_noise':
        keras.backend.set_learning_phase(0)
        keras_model = generate_dense_conv_noise_model()
        keras_model.load_weights('../Model/MNIST_regulars/MNIST_cnn_ 0.h5')
        return keras_model
    elif defense_model_indicator.startswith('adv_model_switch'):
        # e.g. adv_model_switch[epsilon][n]
        # note epsilon 8 means 8/255
        keras.backend.set_learning_phase(0)
        [_, eps, n] = defense_model_indicator.split('[')
        eps = float(eps[:-1])
        n = int(n[:-1])
        weights_forlder = '../Model/adv_train_ICCAD/'
        file_prefix = 'mnist_noaug_nd_eps%.2f' % eps
        keras_model = generate_model_switch_adv(n, weights_forlder, file_prefix)
        print('using adv model switching. Epsilon = %.2f , n = %d' % (eps, n))
        return keras_model
    elif defense_model_indicator.startswith('ICCAD_adv_train'):
        [_, eps] = defense_model_indicator.split('[')
        eps = float(eps[:-1])
        keras.backend.set_learning_phase(0)
        # generate sigle model
        keras_model = generate_single_model()
        # laod weights
        weights_path = '../Model/adv_train_ICCAD/mnist_noaug_nd_eps%.2f_0' % eps
        keras_model.load_weights(weights_path)
        print('Using ICCAD adv train model. Epsilon = %.2f' % eps)
        return keras_model
    elif defense_model_indicator.startswith('model_switch'):
        keras.backend.set_learning_phase(0)
        [_, n] = defense_model_indicator.split('[')
        n = int(n[:-1])
        keras_model = generate_model_switch(n)
        print('using model switching with %d models' % n)
        return keras_model

def choose_generate_model(defense_model_indicator):
    if defense_model_indicator in ('single', 'sap') or defense_model_indicator.startswith('dropout'):
        return choose_defense_model('single')
    elif defense_model_indicator.startswith('gaussian'):
        return choose_defense_model('fix_' + defense_model_indicator)
    elif defense_model_indicator.startswith('hsw'):
        return choose_defense_model('fix_' + defense_model_indicator)













