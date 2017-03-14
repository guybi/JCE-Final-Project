
import tensorflow as tf
from networks.layers import conv, pool

input_size = 14
dropout = 0.2


def build_simple_cnn14(x, weights, biases):

    # input layer (shape=[batch, in_height, in_width, in_channels]
    input = tf.reshape(x, shape=[-1, input_size, input_size, 1])

    # first convolution layer parameters
    filter = [5, 5, 1, 1]
    stride_1 = [1, 1, 1, 1]
    padding_1 = 'VALID'

    # first convolution layer
    conv1 = conv.conv2d(x=input, W=weights['wc1'], b=biases['bc1'], filter=filter, strides=stride_1, padding=padding_1, name='conv1')

    # max pooling (down-sampling)
    conv1 = pool.maxpool2d(conv1, k=2, name='maxpool1')

    # fully connected layer
    fc1 = tf.reshape(conv1, shape=[-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output
    output = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return output