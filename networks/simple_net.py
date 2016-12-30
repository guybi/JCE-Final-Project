
import tensorflow as tf
from networks.layers import conv, pool

input_size = 14
dropout = 0.5
out_size = 3  # 3 class options
learning_rate = 0.00001

def build_simple_cnn14(x, weights, biases):

    # input layer (shape=[batch, in_height, in_width, in_channels]
    input = tf.reshape(x, shape=[-1, input_size, input_size, 1]) # channels=?

    # filter = [filter_height, filter_width, in_channels, out_channels]
    # filter = [5, 5, 64, 64] # channels=?
    # stride_1 = (1, 1)
    # padding_1 = 1

    filter = tf.Variable([5, 5, 1, 1], name='filter')
    filter_size = 5
    stride_1 = [1,1,1,1]
    padding_1 = 'VALID'

    # First Convolution layer parameters
    conv1 = conv.conv2d(x=x, W=weights['wc1'], b=biases['bc1'], filter=filter, strides=stride_1, padding=padding_1)

    # Max Pooling (down-sampling)
    conv1 = pool.maxpool2d(conv1, k=2)

    # Fully Connected Layer
    full_size = tf.Variable(256, name='full_size')
    fc1 = tf.reshape(conv1, shape=[-1, full_size, full_size, 1])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1'], biases['bd1']))
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output
    output = tf.add(tf.matmul(fc1, weights['out'], biases['out']))

    return output
