###############################################################################
# run code
###############################################################################
from random import shuffle
import prep_data as pd
import numpy as np
import dataHelper as dh
import helper_functions as helper_f
import os as os
import tensorflow as tf

# params

vol_src_path = "/home/guy/project/Project_new/DATA_SET/Test/Volumes"
seg_src_path = "/home/guy/project/Project_new/DATA_SET/Test/Segmentations"
train_vol_path = vol_dest_path = "/home/guy/project/Project_new/DATA_SET/Test/Train/Volumes"
train_segmentations_path = seg_dest_path = "/home/guy/project/Project_new/DATA_SET/Test/Train/Segmentations"

klr = 3  # in percentage
dxy = 4
start = 300
end = 600
patch_size = 14
downsample = True
SegRatio = 0.75

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 3 #
dropout = 0.75 # Dropout, probability to keep units
learning_rate = 0.0000001

# Parameters
training_iters = 200000
batch_size = 1024
display_step = 1

def randomize_file_list(file_list):
    tmp = list(file_list) #copy list object
    shuffle(tmp)
    return tmp


resTrainIsExsiting = helper_f.isTrainClassExisiting(vol_dest_path,seg_dest_path)

if (resTrainIsExsiting == False):
    pd.prepare_data(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, klr, SegRatio, dxy=dxy, ind_start=start, ind_end=end, patch_size=patch_size, downsample=downsample)

train_vol_list = os.listdir(train_vol_path)
train_class_list = os.listdir(train_segmentations_path)

# tf Graph input
x = tf.placeholder(tf.float32, [None, None], name='x')
y = tf.placeholder(tf.float32, [None, n_classes], name='y')
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name="wc1"),
    # fully connected, 32 inputs, 512 outputs
    'wd1': tf.Variable(tf.random_normal([32, 256]), name="wd1"),
    # 256 inputs, 3 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([256, n_classes]), name="wout")
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name="bc1"),
    'bd1': tf.Variable(tf.random_normal([256]), name="bd1"),
    'out': tf.Variable(tf.random_normal([n_classes]), name="bout")
}

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out


# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


########################################################################################################################

# Initializing the variables
merged_summary = tf.summary.merge_all()
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1

    while step < training_iters:

        for vol_f in randomize_file_list(train_vol_list):
            print ('training on', vol_f)
            class_f = dh.ret_class_file(vol_f,train_class_list)
            x_data = np.load(train_vol_path + '/' + vol_f)
            y_data = np.load(train_segmentations_path + '/' + class_f)
            x_data, y_data = helper_f.norm_data_rand(x_data, y_data)

            batch_x = tf.train.batch([x_data],
                                        batch_size=[batch_size],
                                        num_threads=1,
                                        enqueue_many=True,
                                        capacity=50000)

            batch_y = tf.train.batch([y_data],
                                     batch_size=[batch_size * 25 * 3],
                                     num_threads=1,
                                     enqueue_many=True,
                                     capacity=50000)

            batch_y = tf.reshape(batch_y, shape=(25600, 3))



        # Run optimization op (backprop)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        epochs = 1
        while not coord.should_stop():
            try:
                # test_batch_x, test_batch_y = tf.train.shuffle_batch(
                #     [x_data, y_data], batch_size=128,
                #     capacity=2000,
                #     min_after_dequeue=1000)

                batch_x_eval, batch_y_eval = sess.run([batch_x, batch_y])

                # Run training
                sess.run(optimizer, feed_dict={x: batch_x_eval, y: batch_y_eval})
                tf.summary.image('ex_output', batch_y_eval)
                if step % display_step == 0:
                    # Calculate batch loss and accuracy
                    # loss, acc, cp = sess.run([cost, accuracy, correct_pred], feed_dict={x: batch_x_eval,
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x_eval, y: batch_y_eval})
                    # print(cp)
                    print("Iter " + str(step * batch_size) + ", Minibatch Loss = " + \
                          "{:.6f}".format(loss / batch_size) + ", Training Accuracy = " + \
                          "{:.5f}".format(acc))

                    # checkpoint visualization
                    # saver.save(sess, "log/model.ckpt")

                    # break condition

                s = sess.run(merged_summary, feed_dict={x: batch_x_eval, y: batch_y_eval})
                step += 1
                epochs += 1

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                # coord.request_stop()
                pass
