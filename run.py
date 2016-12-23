
import tensorflow as tf
from helpers import data_prep
from networks.simple_net import build_simple_cnn14

env = 'test'

seg_ratio = 0.75
klr = 3  # in percentage
learning_rate = 0.00001

if env == 'test':
    vol_src_path = "c:\\CT\\Test\\Volumes"
    seg_src_path = "c:\\CT\\Test\\Segmentations"
    vol_dest_path = "c:\\CT\\Test\\Train\\Volumes"
    seg_dest_path = "c:\\CT\\Test\\Train\\Class"
else:
    vol_src_path = "c:\\CT\\Volumes"
    seg_src_path = "c:\\CT\\Segmentations"
    vol_dest_path = "c:\\CT\\Train\\Volumes"
    seg_dest_path = "c:\\CT\\Train\\Class"

x, y = data_prep.prep_data(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)

weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, klr]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([klr]))
}

pred = build_simple_cnn14(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

batch_size = 128
training_iters = 200000
display_step = 10

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step < training_iters:
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: x, y: y})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: x,
                                                              y: y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: x,
                                      y: y}))