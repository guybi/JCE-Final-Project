
import os
import tensorflow as tf
import numpy as np
from helpers import data_prep
from networks.simple_net import build_simple_cnn14
from networks.triple_net import build_triple_cnn14


env = 'prod'

seg_ratio = 0.75
klr = 3  # in percentage
learning_rate = 0.00001
batch_size = 1000
# training_iters = 200000
training_iters = 200
display_step = 1
validation_files_ind = [18,19]

x = tf.placeholder(tf.float32, [None, None, None, None], name='x')
y = tf.placeholder(tf.float32, [None, 3], name='y')

# x_data = tf.placeholder(tf.float32, [batch_size, None, None, None], name='x_data')
# y_data = tf.placeholder(tf.float32, [batch_size], name='y_data')

batch_x = tf.placeholder(tf.float32, [None, None, None, None], name='batch_x')
batch_y = tf.placeholder(tf.float32, [None, 3], name='batch_y')

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

data_prep.data_load(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)
# data_prep.prepare_val_train_data(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, validation_files_ind)

train_vol_list = os.listdir(train_vol_path)
train_class_list = os.listdir(train_class_path)
# val_vol_path = os.listdir(val_vol_path)
# val_class_list = os.listdir(val_class_path)

weights = {
    # 5x5 conv, 1 input, 64 outputs
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 64])),
    # 5x5 conv, 64 inputs, 32 outputs
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 32])),
    # 5x5 conv, 32 inputs, 32 outputs
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 32, 512])),
    # fully connected, 5*5*32 inputs, 512 outputs
    'wd1': tf.Variable(tf.truncated_normal([512, 512])),
    # fully connected, 5*5*32 inputs, 512 outputs
    'wd2': tf.Variable(tf.truncated_normal([512, 512])),
    # 512 inputs, 3 outputs (class prediction)
    'out': tf.Variable(tf.truncated_normal([512, 3]))
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([64])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([512])),
    'bd1': tf.Variable(tf.truncated_normal([512])),
    'bd2': tf.Variable(tf.truncated_normal([512])),
    'out': tf.Variable(tf.truncated_normal([3]))
}

pred = build_triple_cnn14(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                       momentum=0.9,
                                       use_nesterov=True,
                                       name='momentum').minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

print('start tensorflow session...')

with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Keep training until reach max iterations
    while step < training_iters:
        for vol_f in train_vol_list:
            print('training on', vol_f)
            class_f = data_prep.ret_class_file(vol_f, train_class_list)
            x_data = np.load(train_vol_path + "\\" + vol_f)
            y_data = np.load(train_class_path + "\\" + class_f)
            x_data, y_data = data_prep.norm_data_rand(x_data, y_data)

            # batch_x, batch_y = tf.train.batch([x_data, y_data],
            #                                   batch_size=[batch_size],
            #                                   num_threads=4,
            #                                   enqueue_many=True,
            #                                   capacity=50000)

            batch_x = tf.train.batch([x_data],
                                      batch_size=[batch_size],
                                      num_threads=4,
                                      enqueue_many=True,
                                      capacity=50000)

            batch_y = tf.train.batch([y_data],
                                      batch_size=[batch_size*9*3],
                                      num_threads=4,
                                      enqueue_many=True,
                                      capacity=50000)

            batch_y = tf.reshape(batch_y, shape=(9000,3))

            # batch_x, batch_y = tf.train.shuffle_batch([x_data, y_data],
            #                                           batch_size=batch_size,
            #                                           enqueue_many=True,
            #                                           capacity=50000,
            #                                           min_after_dequeue=1)


            # cost = tf.reduce_mean(tf.cast(tf.nn.softmax_cross_entropy_with_logits(tf.transpose(pred), batch_y), tf.float32))
            # cost = tf.reduce_mean(tf.cast(tf.nn.softmax_cross_entropy_with_logits(tf.transpose(pred), tf.ones_like(batch_y)), tf.float32))
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

            # test_batch_x, test_batch_y = tf.train.shuffle_batch(
            #     [x_data, y_data], batch_size=128,
            #     capacity=2000,
            #     min_after_dequeue=1000)

            # Evaluate model
            # correct_pred = tf.equal(tf.argmax(pred, axis=0), tf.argmax(batch_y, axis=0))
            # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            batch_x_eval, batch_y_eval = sess.run([batch_x, batch_y])

            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x_eval, y: batch_y_eval})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc, cp = sess.run([cost, accuracy, correct_pred], feed_dict={x: batch_x_eval,
                                                                                    y: batch_y_eval})
                print(cp)
                print("Iter " + str(step*batch_size) + ", Minibatch Loss = " + \
                      "{:.6f}".format(loss) + ", Training Accuracy = " + \
                      "{:.5f}".format(acc))

                # print("pred: ")
                # print(sess.run(pred, feed_dict={x: batch_x_eval, y: batch_y_eval}))
                #
                # print("batch_y_eval: ")
                # print(batch_y_eval)
            step += 1
    print("Optimization Finished!")

    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: x_data,
    #                                   y: y_data}))

    coord.request_stop()
    coord.join(threads)
    sess.close()