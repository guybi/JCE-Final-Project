
import tensorflow as tf

def conv2d(x, W, b, filter_size, filters_amount, strides, padding):
    x = tf.nn.conv2d(x, W, strides, padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)