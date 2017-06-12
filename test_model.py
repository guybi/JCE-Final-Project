import os
import tensorflow as tf
import numpy as np
from helpers import data_prep
from networks.build_network import build_network
from networks.weights.liver_kidney_weights import get_weights_and_biases
import SimpleITK as stk
import nibabel as nib

env = 'test'
network = 'simple'
user = 'tal'

klr = 3  # in percentage
seg_ratio = 0.75
batch_size = 1338
display_step = 10
min_epochs = 1000
n_classes = 3

x = tf.placeholder(tf.float32, [None, None, None, None], name='x')
y = tf.placeholder(tf.float32, [None,3], name='y')

test_vol_src_path, test_seg_src_path, test_vol_dest_path,\
test_seg_dest_path, weights_dir,\
predict_segmentations_dir = data_prep.get_test_folders_dir(user, env)

data_prep.data_load(test_vol_src_path, test_seg_src_path, test_vol_dest_path, test_seg_dest_path, seg_ratio, klr)

train_vol_list = os.listdir(test_vol_dest_path)
train_class_list = os.listdir(test_seg_dest_path)

weights, biases = get_weights_and_biases(network)

pred = build_network(network, x, weights, biases)

# Evaluate model
with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_pred = tf.equal(tf.argmax(pred, axis=1), tf.argmax(y, axis=1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

print('start tensorflow session...')
with tf.Session() as sess:
    sess.run(init)

    if tf.train.latest_checkpoint(weights_dir):
        saver.restore(sess, tf.train.latest_checkpoint(weights_dir))
        weights = sess.run(tf.trainable_variables())
        print('Model restored from latest checkpoint.')

    else:
        print('no model found. exiting...')
        exit(-1)

    for vol_f in train_vol_list:
        class_f = data_prep.ret_class_file(vol_f, train_class_list)
        x_data = np.load(test_vol_dest_path + "/" + vol_f)
        label_data = np.load(test_seg_dest_path + "/" + class_f)

        n = np.size(label_data, 0)
        y_data = np.zeros([n, 3])
        y_data[range(n), label_data] = 1

        tmp_img = stk.ReadImage(test_vol_src_path + '\\10000100_1_CTce_ThAb.nii')
        img_dims = tmp_img.GetSize()
        img_size = img_dims[0] * img_dims[1] * img_dims[2]

        seg_tmp = stk.ReadImage('C:\\CT\\Test\\Segmentations\\10000100_1_CTce_ThAb_58_6.nii')
        print("set_tmp - 58_6.nii")

        batch_x, batch_y = tf.train.batch([x_data, y_data],
                                          batch_size=[batch_size],
                                          num_threads=4,
                                          enqueue_many=True,
                                          capacity=32)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        epochs = 1
        z_index = 0
        count_iters = 0
        x_index, y_index = 0, 0
        while not coord.should_stop():
            try:
                batch_x_eval, batch_y_eval = sess.run([batch_x, batch_y])

                # predicted_seg = np.zeros(x_data.size, dtype=np.uint8)
                # predicted_seg = np.zeros(img_size, dtype=np.uint8)
                acctual_prediction = tf.argmax(pred, axis=1)
                ap_eval = sess.run([acctual_prediction], feed_dict={x: batch_x_eval, y: batch_y_eval})

                for val in ap_eval[0]:
                    count_iters += 1
                    print(count_iters)
                    for p_height in range(14):
                        for p_width in range(14):
                            value = 1 if val >= 1 else 0
                            #print('x: ' + str(i), ' y: ', str(j), ' z: ', str(z_index))
                            #print('patch height: ', str(p_height), ' patch_width: ', str(p_width))
                            #print('value: ', str(value))
                            seg_tmp.SetPixel((x_index*14)+p_width, (y_index*14)+p_height, z_index, value)
                            x_index += 1
                        x_index = 0
                        y_index += 1
                    x_index = 0
                    y_index = 0
                stk.WriteImage(seg_tmp, 'Predicted_Segmentations\\' + class_f + '_pseg.nii.gz')
                print('Predicted segmentation has saved.')
                z_index += 1

                if epochs % 10 == 0:
                    loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x_eval, y: batch_y_eval})
                    print("Iter " + str(epochs * batch_size) + ", Minibatch Test Loss = " + \
                          "{:.6f}".format(loss / batch_size) + ", Test Accuracy = " + \
                          "{:.5f}".format(acc))

                    # create predicted segmentation
                    # z_index, i, j = 0, 0, 0
                    # i_patch, j_patch = 0, 0
                    # for k, v in enumerate(ap_eval[0]):
                    #         if i == 13:
                    #             i = 0
                    #             j += 1
                    #         # if v == 0:
                    #         #     continue
                    #         # predicted_seg[z_index, (i+k)+(i_patch*14):((i+k)+(i_patch*14))+14, (j*14)+(j_patch*14):(j*14)+(j_patch*14)+14] = 1 if v >= 1 else 0
                    #         #seg_tmp.SetPixel(z_index, (i+k)+(i_patch*14):((i+k)+(i_patch*14))+14, (j*14)+(j_patch*14):(j*14)+(j_patch*14)+14, 1 if v >= 1 else 0)
                    #         #print('z: ', z_index)
                    #         #print('x: ', (i+k)+(i_patch*14), ' ', ((i+k)+(i_patch*14))+14)
                    #         #print('y: ', (j*14)+(j_patch*14), ' ', (j*14)+(j_patch*14)+14)
                    #         if k == img_dims[0]-1:
                    #             j_patch += 1
                    #         if i_patch == 37:
                    #             i_patch = 0
                    #             j_patch += 1
                    #         if i_patch == img_dims[0]-1 and j_patch == img_dims[1]-1:
                    #             i_patch, j_patch = 0
                    #             z_index += 1
                    #         i_patch += 1

                    # predicted_seg = predicted_seg.reshape(predicted_seg.shape[0], 1)
                    # predicted_seg = predicted_seg.reshape(img_dims[2],img_dims[0],img_dims[1])
                    # stk_img = stk.GetImageFromArray(predicted_seg, isVector=False)
                    # stk_img = stk.Cast(stk_img, stk.sitkUInt8)
                    # stk_img = stk.Shrink(stk_img, [1, 2, 1])
                    # stk.WriteImage(seg_tmp, 'Predicted_Segmentations\\' + class_f + '_pseg.nii.gz')
                    # print('Predicted segmentation has saved.')

                epochs += 1

            except tf.errors.OutOfRangeError:
                print('Done testing -- epoch limit reached')
                coord.request_stop()
            finally:
                pass