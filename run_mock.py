
import os
import tensorflow as tf
import numpy as np
from random import shuffle
from helpers import data_prep
from networks.simple_net import build_simple_cnn14
from networks.double_net import build_double_cnn14
from networks.triple_net import build_triple_cnn14
import matplotlib.pyplot as plt

# def randomize_file_list(file_list):
#     tmp = list(file_list) #copy list object
#     shuffle(tmp)
#     return tmp

env = 'mock'
network = 'simple'

learning_rate = 0.0001
# learning_rate = 0.0000001
momentum = 0.9
# momentum = 0.2
batch_size = 1024
# training_iters = 200000
training_iters = 5
display_step = 5
validation_files_ind = [18,19]
n_classes = 3

x = tf.placeholder(tf.float32, [None, None], name='x')
y = tf.placeholder(tf.float32, [None, None], name='y')

if(env == 'test'):
    vol_src_path = "C:\\CT\\Test\\Volumes"
    seg_src_path = "C:\\CT\\Test\\Segmentations"
    vol_dest_path = "C:\\CT\\Test\\Train\\Volumes"
    seg_dest_path = "C:\\CT\\Test\\Train\\Class"
    train_vol_path = "C:\\CT\\Test\\Train\\Volumes"
    train_class_path = "C:\\CT\\Test\\Train\\Class"
    val_vol_path = "C:\\CT\\Test\\Val\\Volumes"
    val_class_path = "C:\\CT\\Test\\Val\\Class"
elif(env == 'prod'):
    vol_src_path = "C:\\CT\\Volumes"
    seg_src_path = "C:\\CT\\Segmentations"
    vol_dest_path = "C:\\CT\\Train\\Volumes"
    seg_dest_path = "C:\\CT\\Train\\Class"
    train_vol_path = "C:\\CT\\Train\\Volumes"
    train_class_path = "C:\\CT\\Train\\Class"
    val_vol_path = "C:\\CT\\Val\\Volumes"
    val_class_path = "C:\\CT\\Val\\Class"
elif(env == 'mock'):
    vol_src_path = "C:\\CT\\mocks\\volumes"
    seg_src_path = "C:\\CT\\mocks\\segmentations"
    vol_dest_path = "C:\\CT\\mocks\\Train\\volumes"
    seg_dest_path = "C:\\CT\\mocks\\Train\\segmentations"
    train_vol_path = "C:\\CT\\mocks\\Train\\volumes"
    train_class_path = "C:\\CT\\mocks\\Train\\segmentations"

train_vol_list = os.listdir(train_vol_path)
train_class_list = os.listdir(train_class_path)

# simple net weight and biases
if network == 'simple':
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name="wc1"),
        # fully connected, 32 inputs, 512 outputs
        'wd1': tf.Variable(tf.random_normal([32, 256]), name="wd1"),
        # 256 inputs, 3 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([256, 1]), name="wout")
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32]), name="bc1"),
        'bd1': tf.Variable(tf.random_normal([256]), name="bd1"),
        'out': tf.Variable(tf.random_normal([1]), name="bout")
    }

# double net weight and biases
if network == 'double':
    weights = {
        # 5x5 conv, 1 input, 64 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64]), name="wc1"),
        # 5x5 conv, 64 inputs, 64 outputs
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64]), name="wc2"),
        # fully connected, 64 inputs, 256 outputs
        'wd1': tf.Variable(tf.random_normal([64, 256]), name="wd1"),
        # fully connected, 256 inputs, 256 outputs
        'wd2': tf.Variable(tf.random_normal([256, 256]), name="wd2"),
        # 256 inputs, 3 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([256, n_classes]), name="wout")
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([64]), name="bc1"),
        'bc2': tf.Variable(tf.random_normal([64]), name="bc2"),
        'bd1': tf.Variable(tf.random_normal([256]), name="bd1"),
        'bd2': tf.Variable(tf.random_normal([256]), name="bd2"),
        'out': tf.Variable(tf.random_normal([n_classes]), name="bout")
    }

# triple net weight and biases
if network == 'triple':
    weights = {
        # 5x5 conv, 1 input, 64 outputs
        'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64]), name="wc1"),
        # 5x5 conv, 64 inputs, 32 outputs
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 32]), name="wc2"),
        # 5x5 conv, 32 inputs, 32 outputs
        'wc3': tf.Variable(tf.random_normal([3, 3, 32, 32]), name="wc3"),
        # fully connected, 32 inputs, 512 outputs
        'wd1': tf.Variable(tf.random_normal([32, 512]), name="wd1"),
        # fully connected, 512 inputs, 512 outputs
        'wd2': tf.Variable(tf.random_normal([512, 512]), name="wd2"),
        # 512 inputs, 3 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([512, n_classes]), name="wout")
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([64]), name="bc1"),
        'bc2': tf.Variable(tf.random_normal([32]), name="bc2"),
        'bc3': tf.Variable(tf.random_normal([32]), name="bc3"),
        'bd1': tf.Variable(tf.random_normal([512]), name="bd1"),
        'bd2': tf.Variable(tf.random_normal([512]), name="bd2"),
        'out': tf.Variable(tf.random_normal([n_classes]), name="bout")
    }

if network == 'simple':
    pred = build_simple_cnn14(x, weights, biases)
if network == 'double':
    pred = build_double_cnn14(x, weights, biases)
if network == 'triple':
    pred = build_triple_cnn14(x, weights, biases)

# Define loss and optimizer
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
tf.summary.scalar("cost", cost)

with tf.name_scope('train'):
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,
    #                                               name='gradient_descent').minimize(cost)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=momentum,
                                           use_nesterov=True,
                                           use_locking=True,
                                           name='momentum').minimize(cost)

# Evaluate model
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar("accuracy", accuracy)

merged_summary = tf.summary.merge_all()

# Initializing the variables
init = tf.global_variables_initializer()

print('start tensorflow session...')

with tf.Session() as sess:
    sess.run(init)
    step = 1

    writer = tf.summary.FileWriter("log", sess.graph)

    # Keep training until reach max iterations
    while step < training_iters:
        for vol_f in train_vol_list:
            print('training on', vol_f)
            class_f = data_prep.ret_class_file(vol_f, train_class_list)
            x_data = np.load(train_vol_path + "\\" + vol_f)
            y_data = np.load(train_class_path + "\\" + class_f)

            # plt.imshow(x_data, cmap='gray')
            # plt.show()
            # plt.imshow(y_data, cmap='gray')
            # plt.show()

            y_data = np.reshape(y_data, (196,1))

            sess.run(optimizer, feed_dict={x: x_data, y: y_data})
            # if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: x_data, y: y_data})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss = " + \
                  "{:.6f}".format(loss) + ", Training Accuracy = " + \
                  "{:.5f}".format(acc))

            # s = sess.run(merged_summary, feed_dict={x: x_data, y: y_data})
            # writer.add_summary(s, step)
            step += 1

    print("Optimization Finished!")

    sess.close()
