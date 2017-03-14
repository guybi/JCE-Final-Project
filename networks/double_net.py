
import tensorflow as tf
from networks.layers import conv, pool

input_size = 14
dropout = 0.2


def build_double_cnn14(x, weights, biases):

    # input layer (shape=[batch, in_height, in_width, in_channels]
    input = tf.reshape(x, shape=[-1, input_size, input_size, 1]) # in_channels=64?

    # first convolution layer parameters
    filter_1 = [5, 5, 1, 1]
    stride_1 = [1,1,1,1]
    padding_1 = 'VALID'

    # first convolution layer
    conv1 = conv.conv2d(x=input, W=weights['wc1'], b=biases['bc1'], filter=filter_1, strides=stride_1, padding=padding_1, name='conv1')

    # max pooling (down-sampling)
    conv1 = pool.maxpool2d(conv1, k=2, name='maxpool1')

    filter_2 = [3, 3, 1, 1]
    stride_2 = [1, 1, 1, 1]
    padding_2 = 'VALID'

    # second convolution layer
    conv2 = conv.conv2d(x=conv1, W=weights['wc2'], b=biases['bc2'], filter=filter_2, strides=stride_2, padding=padding_2, name='conv2')

    # max pooling (down-sampling)
    conv2 = pool.maxpool2d(conv2, k=2, name='maxpool2')

    # fully connected layer
    full_size = 256
    fc1 = tf.reshape(conv2, shape=[-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output
    output = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return output
