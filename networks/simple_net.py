
import tensorflow as tf

input_size = 14
dropout = 0.5
out_size = 3  # 3 class options
learning_rate = 0.00001



def build_simple_cnn14(x, weights, biases, dropout):

    # input layer (shape=[batch, in_height, in_width, in_channels]
    input = tf.reshape(x, shape=[-1, input_size, input_size, 64]) # channels=?

    # filter = [filter_height, filter_width, in_channels, out_channels]
    filter = [5, 5, 64, 64] # channels=?
    stride_1 = (1, 1)
    padding_1 = 1

    # First Convolution layer parameters
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], filter, stride_1, padding_1)

    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Fully Connected Layer
    full_size = 256
    fc1 = tf.reshape(conv1, shape=[-1, full_size, full_size, 1])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1'], biases['bd1']))
    fc1 = tf.nn.relu(fc1)

    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)



    # Output
    tf.add(tf.matmul(fc1, weights['out'], biases['out']))


def conv2d(x, W, b, filter_size, filters_amount, strides, padding):
    x = tf.nn.conv2d(x, W, strides, padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")