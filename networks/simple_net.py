
import tensorflow as tf
from networks.layers import conv, pool

input_size = 14
dropout = 0.2


def build_simple_cnn14(x, weights, biases):

    # input layer (shape=[batch, in_height, in_width, in_channels]
    input = tf.reshape(x, shape=[-1, input_size, input_size, 1])

    # first convolution layer parameters
    stride_1 = [1, 1, 1, 1]
    padding_1 = 'SAME'

    # first convolution layer
    conv1 = conv.conv2d(x=input, W=weights['wc1'], b=biases['bc1'], strides=stride_1, padding=padding_1, name='conv1')

    variable_summaries(weights['wc1'], "wc1")
    variable_summaries(biases['bc1'], "bc1")
    # max pooling (down-sampling)
    # conv1 = pool.maxpool2d(conv1, k=2, name='maxpool1')

    # fully connected layer
    fc1 = tf.reshape(conv1, shape=[-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    variable_summaries(weights['wd1'], "weights")
    variable_summaries(biases['bd1'], "bd1")

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    variable_summaries(fc1, "activation")

    # Output
    output = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    variable_summaries(weights['out'], "weights")
    variable_summaries(biases['out'], "out")

    return output


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram(name, var)
