import os
import tensorflow as tf
import numpy as np
from random import shuffle
from helpers import data_prep
from networks.simple_net import build_simple_cnn14
from networks.double_net import build_double_cnn14
from networks.triple_net import build_triple_cnn14
import matplotlib.pyplot as plt

def randomize_file_list(file_list):
    tmp = list(file_list) #copy list object
    shuffle(tmp)
    return tmp

env = 'test'
network = 'simple'

klr = 3  # in percentage
seg_ratio = 0.75
# learning_rate = 0.0001
learning_rate = 0.0001
momentum = 0.9
# momentum = 0.2
batch_size = 1000
# training_iters = 200000
training_iters = 5
display_step = 10
validation_files_ind = [18,19]
n_classes = 3
user = 'guy'

x = tf.placeholder(tf.float32, [None, None, None, None], name='x')
y = tf.placeholder(tf.float32, [None,3], name='y')

# x_data = tf.placeholder(tf.float32, [None, None, None, None], name='x_data')
# y_data = tf.placeholder(tf.float32, [None, None], name='y_data')

#batch_x = tf.placeholder(tf.float32, [None, None, None, None], name='batch_x')
#batch_y = tf.placeholder(tf.float32, [None, 3], name='batch_y')

if (user == 'tal'):
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

if (user == 'guy'):
    if (env == 'test'):
        vol_src_path = "/home/guy/project/CT/Test/Volumes"
        seg_src_path = "/home/guy/project/CT/Test/Segmentations"
        vol_dest_path = "/home/guy/project/CT/Test/Train/Volumes"
        seg_dest_path = "/home/guy/project/CT/Test/Train/Class"
        train_vol_path = "/home/guy/project/CT/Test/Train/Volumes"
        train_class_path = "/home/guy/project/CT/Test/Train/Class"
        val_vol_path = "/home/guy/project/CT/Test/Val/Volumes"
        val_class_path = "/home/guy/project/CT/Test/Val/Class"

    elif (env == 'prod'):
        vol_src_path = "/home/guy/project/CT/Volumes"
        seg_src_path = "/home/guy/project/CT/Segmentations"
        vol_dest_path = "/home/guy/project/CT/Train/Volumes"
        seg_dest_path = "/home/guy/project/CT/Train/Class"
        train_vol_path = "/home/guy/project/CT/Train/Volumes"
        train_class_path = "/home/guy/project/CT/Train/Class"
        val_vol_path = "/home/guy/project/CT/Val/Volumes"
        val_class_path = "/home/guy/project/CT/Val/Class"

data_prep.data_load(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)

train_vol_list = os.listdir(train_vol_path)
train_class_list = os.listdir(train_class_path)

# simple net weight and biases
if network == 'simple':
    weights = {
        # 5x5 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.truncated_normal([ 5, 5,1 ,32], stddev=0.1), name="wc1"),
        # fully connected, 32 inputs, 512 outputs
        'wd1': tf.Variable(tf.truncated_normal([1568, 256], stddev=0.1), name="wd1"), #32
        # 256 inputs, 3 outputs (class prediction)
        'out': tf.Variable(tf.truncated_normal([256, n_classes], stddev=0.1), name="wout")
    }

    biases = {
        'bc1': tf.Variable(tf.truncated_normal([32], stddev=0.1), name="bc1"),
        'bd1': tf.Variable(tf.truncated_normal([256], stddev=0.1), name="bd1"),
        'out': tf.Variable(tf.truncated_normal([n_classes], stddev=0.1), name="bout")
    }

# double net weight and biases
if network == 'double':
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
if network == 'triple':
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

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=momentum,
                                           use_nesterov=True,
                                           use_locking=False, #GUY: Originally True
                                           name='Momentum').minimize(cost)

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
        min_epochs = 100
        for vol_f in randomize_file_list(train_vol_list):
            print('training on', vol_f)
            class_f = data_prep.ret_class_file(vol_f, train_class_list)
            x_data = np.load(train_vol_path + "/" + vol_f)
            y_data = np.load(train_class_path + "/" + class_f)

            x_data, label_data = data_prep.norm_data_rand(x_data, y_data)

            n=np.size(label_data,0)
            y_data= np.zeros([n,3])
            y_data[range(n),label_data]=1

            tf.summary.image('ex_input', x_data)
            tf.summary.image('ex_output', y_data)

            batch_x, batch_y = tf.train.batch([x_data, y_data],
                                     batch_size=[batch_size],
                                     num_threads=1,
                                     enqueue_many=True,
                                     capacity=50000)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            epochs = 100
            while not coord.should_stop():
                try:
                    batch_x_eval, batch_y_eval = sess.run([batch_x, batch_y])
                    # Run training
                    sess.run(optimizer, feed_dict={x: batch_x_eval, y: batch_y_eval})
                    # tf.summary.image('ex_output', batch_y_eval)
                    if step % display_step == 0:
                        # Calculate batch loss and accuracy
                        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x_eval, y: batch_y_eval})
                        # print(cp)
                        print("Iter " + str(step * batch_size) + ", Minibatch Loss = " + \
                              "{:.6f}".format(loss/batch_size) + ", Training Accuracy = " + \
                              "{:.5f}".format(acc))

                        # break condition
                        if epochs > min_epochs and acc > 0.95:
                            break

                    step += 1
                    epochs += 1

                except tf.errors.OutOfRangeError:
                    print('Done training -- epoch limit reached')
                finally:
                    # When done, ask the threads to stop.
                    # coord.request_stop()
                    pass
    print("Optimization Finished!")

    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: x_data,
    #                                   y: y_data}))

    sess.close()