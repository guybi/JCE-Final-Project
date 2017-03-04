
import tensorflow as tf
from networks.layers import conv, pool

input_size = 14
dropout = 0.5
out_size = 3  # 3 class options
learning_rate = 0.00001

def build_triple_cnn14(x, weights, biases):

    # input layer (shape=[batch, in_height, in_width, in_channels]
    input = tf.reshape(x, shape=[-1, input_size, input_size, 1]) # in_channels=64?

    # first convolution layer parameters
    filter_1 = tf.Variable([5, 5, 1, 1], dtype='float32', name='filter_1')
    stride_1 = [1,1,1,1]
    padding_1 = 'VALID'

    # first convolution layer
    conv1 = conv.conv2d(x=input, W=weights['wc1'], b=biases['bc1'], filter=filter_1, strides=stride_1, padding=padding_1)

    # max pooling (down-sampling)
    conv1 = pool.maxpool2d(conv1, k=2)

    # ReLU on conv1
    conv1 = tf.nn.relu(conv1)

    filter_2 = tf.Variable([3, 3, 1, 1], dtype='float32', name='filter_2')
    stride_2 = [1, 1, 1, 1]
    padding_2 = 'VALID'

    # second convolution layer
    conv2 = conv.conv2d(x=conv1, W=weights['wc2'], b=biases['bc2'], filter=filter_2, strides=stride_2, padding=padding_2)

    # max pooling (down-sampling)
    conv2 = pool.maxpool2d(conv2, k=2)

    # RelU on conv2
    conv2 = tf.nn.relu(conv2)

    filter_3 = tf.Variable([3, 3, 1, 1], dtype='float32', name='filter_3')
    stride_3 = [1, 1, 1, 1]
    padding_3 = 'SAME'

    # third convolution layer
    conv3 = conv.conv2d(x=conv2, W=weights['wc3'], b=biases['bc3'], filter=filter_3, strides=stride_3, padding=padding_3)

    # max pooling (down-sampling)
    conv3 = pool.maxpool2d(conv3, k=2)

    # ReLU on conv3
    conv3 = tf.nn.relu(conv3)


    # fully connected layer
    fc1 = tf.reshape(conv3, shape=[-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.reshape(fc1, shape=[-1, weights['wd2'].get_shape().as_list()[0]])
    fc2 = tf.add(tf.matmul(fc2, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)

    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    # Output
    output = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

    return output
