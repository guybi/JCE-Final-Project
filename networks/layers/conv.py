
import tensorflow as tf

def conv2d(x, W, b, filter, strides, padding):
    x = tf.nn.conv2d(x, W, strides, padding, filter=filter)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)