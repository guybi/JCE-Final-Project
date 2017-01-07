
import os
import tensorflow as tf
import numpy as np
from helpers import data_prep
from networks.simple_net import build_simple_cnn14


env = 'test'

seg_ratio = 0.75
klr = 3  # in percentage
learning_rate = 0.00001
batch_size = 200
# training_iters = 200000
training_iters = 2
display_step = 1

x = tf.placeholder(tf.float32, [batch_size, None, None, None])
y = tf.placeholder(tf.float32, [batch_size])

if env == 'test':
    vol_src_path = "/home/tal/CT/Test/Volumes"
    seg_src_path = "/home/tal/CT/Test/Segmentations"
    vol_dest_path = "/home/tal/CT/Test/Train/Volumes"
    seg_dest_path = "/home/tal/CT/Test/Train/Class"
    train_vol_path = "/home/tal/CT/Test/Train/Volumes"
    train_class_path = "/home/tal/CT/Test/Train/Class"
    val_vol_path = "/home/tal/CT/Test/Val/Volumes"
    val_class_path = "/home/tal/CT/Test/Val/Class"
else:
    vol_src_path = "/home/tal/CT/Volumes"
    seg_src_path = "/home/tal/CT/Segmentations"
    vol_dest_path = "/home/tal/CT/Train/Volumes"
    seg_dest_path = "/home/tal/CT/Train/Class"
    train_vol_path = "/home/tal/CT/Train/Volumes"
    train_class_path = "/home/tal/CT/Train/Class"
    val_vol_path = "/home/tal/CT/Val/Volumes"
    val_class_path = "/home/tal/CT/Val/Class"

data_prep.data_load(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)

train_vol_list = os.listdir(train_vol_path)
train_class_list = os.listdir(train_class_path)
val_vol_list = os.listdir(val_vol_path)
val_class_list = os.listdir(val_class_path)

weights = {
    # 5x5 conv, 1 input, 64 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 64])),
    # 5x5 conv, 32 inputs, 64 outputs
    # 'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 5*5*64 inputs, 256 outputs
    'wd1': tf.Variable(tf.random_normal([5*5*64, 256])),
    # 256 inputs, 1 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([256, 1]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([64])),
    # 'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([256])),
    'out': tf.Variable(tf.random_normal([1]))
}

pred = build_simple_cnn14(x, weights, biases)



# Define loss and optimizer
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, np.array(y_data).reshape(y_data.shape[0], 1)))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.transpose(pred), y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

print 'start tensorflow session...'

with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Keep training until reach max iterations
    while step < training_iters:
        batch_x, batch_y = tf.train.batch([x_data, y_data], batch_size=batch_size, enqueue_many=True, capacity=32)
        test_batch_x, test_batch_y = tf.train.shuffle_batch(
            [x_data, y_data], batch_size=128,
            capacity=2000,
            min_after_dequeue=1000)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        batch_x, batch_y = sess.run([batch_x, batch_y])

        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: batch_x,
    #                                   y: batch_y}))

    coord.request_stop()
    coord.join(threads)
    sess.close()