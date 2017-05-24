
import tensorflow as tf

n_classes = 3


def get_weights_and_biases(network_type):
    if network_type == 'simple':
        weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name="wc1"),
            # fully connected, 32 inputs, 512 outputs
            'wd1': tf.Variable(tf.truncated_normal([1568, 256], stddev=0.1), name="wd1"),  # 32
            # 256 inputs, 3 outputs (class prediction)
            'out': tf.Variable(tf.truncated_normal([256, n_classes], stddev=0.1), name="wout")
        }

        biases = {
            'bc1': tf.Variable(tf.truncated_normal([32], stddev=0.1), name="bc1"),
            'bd1': tf.Variable(tf.truncated_normal([256], stddev=0.1), name="bd1"),
            'out': tf.Variable(tf.truncated_normal([n_classes], stddev=0.1), name="bout")
        }

    # double net weight and biases
    if network_type == 'double':
        weights = {
            # 5x5 conv, 1 input, 64 outputs
            'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 64], stddev=0.01), name="wc1"),
            # 5x5 conv, 64 inputs, 64 outputs
            'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.01), name="wc2"),
            # fully connected, 64 inputs, 256 outputs
            'wd1': tf.Variable(tf.truncated_normal([64, 256], stddev=0.01), name="wd1"),
            # fully connected, 256 inputs, 256 outputs
            'wd2': tf.Variable(tf.truncated_normal([256, 256], stddev=0.01), name="wd2"),
            # 256 inputs, 3 outputs (class prediction)
            'out': tf.Variable(tf.truncated_normal([256, n_classes], stddev=0.01), name="wout")
        }

        biases = {
            'bc1': tf.Variable(tf.truncated_normal([64], stddev=0.01), name="bc1"),
            'bc2': tf.Variable(tf.truncated_normal([64], stddev=0.01), name="bc2"),
            'bd1': tf.Variable(tf.truncated_normal([256], stddev=0.01), name="bd1"),
            'bd2': tf.Variable(tf.truncated_normal([256], stddev=0.01), name="bd2"),
            'out': tf.Variable(tf.truncated_normal([n_classes], stddev=0.01), name="bout")
        }

    # triple net weight and biases
    if network_type == 'triple':
        weights = {
            # 5x5 conv, 1 input, 64 outputs
            'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 64], stddev=0.5), name="wc1"),
            # 5x5 conv, 64 inputs, 32 outputs
            'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 32], stddev=0.5), name="wc2"),
            # 5x5 conv, 32 inputs, 32 outputs
            'wc3': tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.5), name="wc3"),
            # fully connected, 32 inputs, 512 outputs
            'wd1': tf.Variable(tf.truncated_normal([32, 512], stddev=0.5), name="wd1"),
            # fully connected, 512 inputs, 512 outputs
            'wd2': tf.Variable(tf.truncated_normal([512, 512], stddev=0.5), name="wd2"),
            # 512 inputs, 3 outputs (class prediction)
            'out': tf.Variable(tf.truncated_normal([512, n_classes], stddev=0.5), name="wout")
        }

        biases = {
            'bc1': tf.Variable(tf.truncated_normal([64], stddev=0.5), name="bc1"),
            'bc2': tf.Variable(tf.truncated_normal([32], stddev=0.5), name="bc2"),
            'bc3': tf.Variable(tf.truncated_normal([32], stddev=0.5), name="bc3"),
            'bd1': tf.Variable(tf.truncated_normal([512], stddev=0.5), name="bd1"),
            'bd2': tf.Variable(tf.truncated_normal([512], stddev=0.5), name="bd2"),
            'out': tf.Variable(tf.truncated_normal([n_classes], stddev=0.5), name="bout")
        }
    return weights, biases
