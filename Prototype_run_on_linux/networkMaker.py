import numpy as np
import theano
import theano.tensor as T
import lasagne


# Helper functions for CNN network creation

###############################################################################
###############################################################################
# function receives: input_var - input data
# function returns: # the output layer of a a single cnn layer network
#
# assuming: input size 14x14 images
def build_triple_cnn14_big_fc(input_var=None):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 14, 14),
                                        input_var=input_var)

    # First convolutional layer parameters
    filt1_size = (5, 5)
    filters_1 = 64
    stride_1 = (1, 1)
    padding_1 = 1

    # First convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_1, filter_size=filt1_size, stride=stride_1, pad=padding_1,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # max pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # second convolutional layer parameters
    filt2_size = (3, 3)
    filters_2 = 32
    stride_2 = (1, 1)
    padding_2 = 1

    # Second convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_2, filter_size=filt2_size, stride=stride_2, pad=padding_2,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # third convolutional layer parameters
    filt3_size = (3, 3)
    filters_3 = 32
    stride_3 = (1, 1)
    padding_3 = 0

    # third convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_3, filter_size=filt3_size, stride=stride_3, pad=padding_3,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # max pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # fully connected layer
    full_size = 512
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.2),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # fully connected layer
    full_size = 512
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.2),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # output layer
    out_size = 3  # 3 class options
    network = lasagne.layers.DenseLayer(
        network,
        num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)  # softmax on non-linearity

    return network


###############################################################################
###############################################################################
# function receives: input_var - input data
# function returns: # the output layer of a a single cnn layer network
#
# assuming: input size 14x14 images
def build_double_cnn14_onemaxpool(input_var=None):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 14, 14),
                                        input_var=input_var)

    # First convolutional layer parameters
    filt1_size = (5, 5)
    filters_1 = 64
    stride_1 = (1, 1)
    padding_1 = 1

    # First convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_1, filter_size=filt1_size, stride=stride_1, pad=padding_1,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # max pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # First convolutional layer parameters
    filt2_size = (3, 3)
    filters_2 = 64
    stride_2 = (1, 1)
    padding_2 = 0

    # Second convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_2, filter_size=filt2_size, stride=stride_2, pad=padding_2,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # fully connected layer
    full_size = 256
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # fully connected layer
    full_size = 256
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # output layer
    out_size = 3  # 3 class options
    network = lasagne.layers.DenseLayer(
        network,
        num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)  # softmax on non-linearity

    return network


###############################################################################
###############################################################################
# function receives: input_var - input data
# function returns: # the output layer of a a single cnn layer network
#
# assuming: input size 14x14 images
def build_double_cnn14(input_var=None):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 14, 14),
                                        input_var=input_var)

    # First convolutional layer parameters
    filt1_size = (5, 5)
    filters_1 = 32
    stride_1 = (1, 1)
    padding_1 = 1

    # First convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_1, filter_size=filt1_size, stride=stride_1, pad=padding_1,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # max pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # First convolutional layer parameters
    filt2_size = (3, 3)
    filters_2 = 32
    stride_2 = (1, 1)
    padding_2 = 0

    # Second convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_2, filter_size=filt2_size, stride=stride_2, pad=padding_2,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # max pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # fully connected layer
    full_size = 256
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # fully connected layer
    full_size = 256
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # output layer
    out_size = 3  # 3 class options
    network = lasagne.layers.DenseLayer(
        network,
        num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)  # softmax on non-linearity

    return network


###############################################################################
###############################################################################
# function receives: input_var - input data
# function returns: # the output layer of a a single cnn layer network
#
# assuming: input size 28x28
def build_double_cnn28(input_var=None):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # First convolutional layer parameters
    filt1_size = (11, 11)
    filters_1 = 64
    stride_1 = (1, 1)
    padding_1 = 0

    # First convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_1, filter_size=filt1_size, stride=stride_1, pad=padding_1,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # max pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # First convolutional layer parameters
    filt1_size = (3, 3)
    filters_1 = 32
    stride_1 = (1, 1)
    padding_1 = 0

    # First convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_1, filter_size=filt1_size, stride=stride_1, pad=padding_1,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # max pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # fully connected layer
    full_size = 256
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=0.2),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # output layer
    out_size = 3  # 3 class options
    network = lasagne.layers.DenseLayer(
        network,
        num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)  # softmax on non-linearity

    return network


###############################################################################
###############################################################################
# function receives: input_var - input data
# function returns: # the output layer of a a single cnn layer network
#
# assuming: input size 16x16 images
def build_simple_cnn14(input_var=None):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 14, 14),
                                        input_var=input_var)

    # First convolutional layer parameters
    filt1_size = (5, 5)
    filters_1 = 32
    stride_1 = (1, 1)
    padding_1 = 1

    # First convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_1, filter_size=filt1_size, stride=stride_1, pad=padding_1,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # max pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # fully connected layer
    full_size = 256
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # output layer
    out_size = 3  # 3 class options
    network = lasagne.layers.DenseLayer(
        network,
        num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)  # softmax on non-linearity

    return network


###############################################################################
###############################################################################
# function receives: input_var - input data
# function returns: # the output layer of a a single cnn layer network
#
# assuming: input size 16x16 images
def build_simple_cnn16(input_var=None):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 16, 16),
                                        input_var=input_var)

    # First convolutional layer parameters
    filt1_size = (5, 5)
    filters_1 = 32
    stride_1 = (1, 1)
    padding_1 = 0

    # First convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_1, filter_size=filt1_size, stride=stride_1, pad=padding_1,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # max pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # fully connected layer
    full_size = 256
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # output layer
    out_size = 3  # 3 class options
    network = lasagne.layers.DenseLayer(
        network,
        num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)  # softmax on non-linearity

    return network


###############################################################################
###############################################################################
# function receives: input_var - input data
# function returns:  the output layer of a an alexnet variation neural network
#
# assuming: input size 28x28 images
def build_simple_cnn28(input_var=None):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # First convolutional layer parameters
    filt1_size = (5, 5)
    filters_1 = 32
    stride_1 = (1, 1)
    padding_1 = 0

    # First convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_1, filter_size=filt1_size, stride=stride_1, pad=padding_1,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # max pooling layer
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # fully connected layer
    full_size = 256
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # output layer
    out_size = 3  # 3 class options
    network = lasagne.layers.DenseLayer(
        network,
        num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)  # softmax on non-linearity

    return network


###############################################################################
###############################################################################
# function receives: input_var - input data
# function returns: # the output layer of a an alexnet variation neural network
#
# assuming: input size 28x28 images
def build_AlexNet28(input_var=None):
    # note: in all layers non linearty neuron function is RelU excepot for out which is softmax
    # hence for all middle layers set: nonlinearity=lasagne.nonlinearities.rectify
    # and for out layer set nonlinearty = lasagne.nonlinearities.softmax

    # input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)

    # First convolutional layer parameters
    filt1_size = (5, 5)
    filters_1 = 32
    stride_1 = (1, 1)
    padding_1 = 0

    # First convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_1, filter_size=filt1_size, stride=stride_1, pad=padding_1,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # next do max pooling
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # second and third convolutional layer params
    filt23_size = (3, 3)
    filters_23 = 48
    stride_23 = (1, 1)
    padding_23 = 1

    # Second convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_23, filter_size=filt23_size, stride=stride_23, pad=padding_23,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # Third convolutional network
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_23, filter_size=filt23_size, stride=stride_23, pad=padding_23,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # fourth convolutional layer params
    filt4_size = (3, 3)
    filters_4 = 32
    stride_4 = (1, 1)
    padding_4 = 1

    # fourth convolutional layer
    network = lasagne.layers.Conv2DLayer(
        network, num_filters=filters_4, filter_size=filt4_size, stride=stride_4, pad=padding_4,
        W=lasagne.init.GlorotUniform(),  # default init of weights
        nonlinearity=lasagne.nonlinearities.rectify)  # RelU non linearity

    # perform max_pooling
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # fully connected layers
    # perform dropout on first two fully connected layers
    # first fully connected layer
    full_size = 256
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # second fully connected layer
    full_size = 256
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=full_size,
        nonlinearity=lasagne.nonlinearities.rectify)

    # out layer with 3 outputs - liver, kidney, nothing
    out_size = 3  # 3 class options
    network = lasagne.layers.DenseLayer(
        network,
        num_units=out_size,
        nonlinearity=lasagne.nonlinearities.softmax)  # softmax on non-linearity

    return network




