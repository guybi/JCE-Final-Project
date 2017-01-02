
import tensorflow as tf

def conv2d(x, W, b, filter, strides=[0,0,0,0], padding='SAME'):
    x = tf.nn.conv2d(x, W, strides, padding)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)