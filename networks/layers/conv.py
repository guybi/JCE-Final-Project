
import tensorflow as tf


def conv2d(x, W, b, strides=[0,0,0,0], padding='SAME', name=''):
    x = tf.nn.conv2d(x, W, strides, padding, name=name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)