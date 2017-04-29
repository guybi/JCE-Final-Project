import os
import time
from random import shuffle
import numpy as np
import theano
import theano.tensor as T
import lasagne
from networkMaker import build_simple_cnn14_big_fc
import dataHelper as dh
import tensorflow as tf

#function receives: file_list - list of files names
#         returns: file list in random order
def randomize_file_list(file_list):
    tmp = list(file_list) #copy list object
    shuffle(tmp)
    return tmp


# function receives: vol_fn - volume file name
#                   class_list - list of classification files
#        returns: file name of classification file which classifies vol_fn file
def ret_class_file(vol_fn, class_list):
    # start of each
    vol_header = "Volume_patchsize_"
    class_header = "Classification_patchsize_"

    # copy class list
    tmp_class_list = list(class_list)

    # remove header from file name
    tmp_vol_fn = vol_fn[len(vol_header):len(vol_fn)]

    # remove header from each file name in class list
    k = 0
    for f in class_list:
        tmp_class_list[k] = f[len(class_header):len(f)]
        k += 1

    # search for tmp_vol_fn in tmp_class_list and get index
    ind = tmp_class_list.index(tmp_vol_fn)
    return class_list[ind]


# function
# receives: inputs - numpy
# input
# array
#                   targets - numpy classification array
#                   batchsize - the size of the batches to divide the data into
#                   shuffle - True = shuffle data order, FALSE - do not shuffle
#         returns:
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

    # deal with last batch
    if (len(inputs) % batchsize != 0):
        b_count = len(inputs) / batchsize
        if shuffle:
            excerpt = indices[batchsize * b_count:len(inputs)]
        else:
            excerpt = slice(batchsize * b_count, len(inputs))
        yield inputs[excerpt], targets[excerpt]


def norm_data_rand(vol, cat):
    # get indexes of each type of patch
    liver_ind = np.where(cat == dh.liver)
    kidney_ind = np.where(cat == dh.kidney)
    nothing_ind = np.where(cat == dh.nothing)

    # get all liver patches
    livers = vol[liver_ind[0], :, :, :]
    # get nothings patches in random order
    tmp = np.array(vol[nothing_ind[0], :, :, :], dtype=np.float32)
    np.random.shuffle(tmp)
    nothings = tmp
    # create extended kidneys data
    if (len(kidney_ind[0] != 0)):
        r = len(liver_ind[0]) / len(kidney_ind[0])
        tmp = vol[kidney_ind[0]]
        d = tmp.shape
        kidneys = np.zeros((r * d[0], d[1], d[2], d[3]), dtype=np.float32)
        for k in range(r):
            kidneys[k * d[0]:(k + 1) * d[0], :, :, :] = vol[kidney_ind[0]]

    # create new classification array
    if (len(kidney_ind[0] != 0)):
        new_class = np.zeros(len(livers) + len(kidneys) + len(livers), dtype=np.int32)
        new_class[:len(livers)] = dh.liver
        new_class[len(livers):len(livers) + len(kidneys)] = dh.kidney
        new_class[len(livers) + len(kidneys):] = dh.nothing
    else:
        new_class = np.zeros(2 * len(livers), dtype=np.int32)
        new_class[:len(livers)] = dh.liver
        new_class[len(livers):] = dh.nothing

    # create a new and smaller numpy volumes array
    if (len(kidney_ind[0] != 0)):
        new_vol = np.zeros((len(livers) + len(kidneys) + len(livers), d[1], d[2], d[3]), dtype=np.float32)
        new_vol[:len(livers), :, :, :] = livers
        new_vol[len(livers):len(livers) + len(kidneys), :, :, :] = kidneys
        new_vol[len(livers) + len(kidneys):, :, :, :] = nothings[0:len(livers)]
    else:
        new_vol = np.zeros((2 * len(livers), d[1], d[2], d[3]), dtype=np.float32)
        new_vol[:len(livers), :, :, :] = livers
        new_vol[len(livers):, :, :, :] = nothings[0:len(livers)]

    return new_vol, new_class

#main function that handles the training
#function recieves: train_vol_path - path of training volumes
#                   train_class_path - path of classification vectors for each volume
#                   vol_val_path - validation volumes path
#                   val_class_path - validation classification vectors path
#                   network_params_path - path of network parameters to be loaded
#                   res_path - path in which to save results vectors and network params
#                   epochs - number of epochs tu run. default is 2000
#                   load_params - whether to load params from network_params_path. default is False.\
#                   stop_sat - if True function will stop training when train_loss is in saturation.
#
#function runs the training and saves at end of run time:
#               1) weights of the neural network
#               2) training loss function vector - loss function value at each epoch
#               3) validation loss function vector - loss function value at each epoch
#               4) accuracy vector - accuracy of validation set at each epoch

def train_cnn(train_vol_path, train_class_path, vol_val_path, val_class_path, network_params_path,
              train_loss_path, valid_loss_path, valid_acc_path, res_path, epochs=200,
              load_params=False, stop_sat=False):
    train_vol_list = os.listdir(train_vol_path)
    train_class_list = os.listdir(train_class_path)
    # val_vol_list = os.listdir(val_vol_path)
    # val_class_list = os.listdir(val_class_path)

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # create neural network
    network = build_simple_cnn14_big_fc(input_var)

    if (load_params):
        with np.load(network_params_path) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)

    # create a loss epxression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # create update expression for training
    mu = 0.9
    rate = 0.00001

    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=rate, momentum=mu)

    # loss expression for validation purposes
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    # expression for test accuracy
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    # update function
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # validation loss function and test_accuracy
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # print ret_class_file("Volume_patchsize_28_18_xshift0_yshift0.npy", val_class_list)

    train_loss_arr = list()
    val_loss_arr = list()
    val_acc_arr = list()
    # testing each individual patch type accuracy -  to get the feel of it :]
    val_kid_acc_arr = list()
    val_liver_acc_arr = list()
    val_nothing_acc_arr = list()
    begin_itr = 0

    if (load_params):  # if loding network params, continue running from last saved point
        train_loss_arr = np.load(train_loss_path).tolist()
        val_loss_arr = np.load(valid_loss_path).tolist()
        val_acc_arr = np.load(valid_acc_path).tolist()
        begin_itr = len(train_loss_arr)

    print "starting from itr = " + str(begin_itr)

    # Training...
    for itr in range(begin_itr, epochs):
        # reset vars
        train_err = 0.0
        train_batches = 0.0
        start_time = time.time()
        train_batch_size = 1000

        # full pass over the training data
        for vol_f in randomize_file_list(train_vol_list):
            # get classification file name
            class_f = ret_class_file(vol_f, train_class_list)
            # read from disk training volume and classification files
            vol_train_raw = np.load(train_vol_path + "/" + vol_f)
            class_train_raw = np.load(train_class_path + "/" + class_f)
            # create normalized data from the loaded file in which there is an equal
            # amount of kidneys, livers and nothings, by taking all liver patches
            # multipying kindey patches, and taking random nothings patches
            vol_train, class_train = norm_data_rand(vol_train_raw, class_train_raw)

            for batch in iterate_minibatches(vol_train, class_train, train_batch_size, shuffle=True):
                inputs, targets = batch
                r = len(inputs) / float(train_batch_size)
                train_err += r * train_fn(inputs, targets)
                train_batches += r

            # free numpy array memory
            del vol_train_raw
            del class_train_raw
            del vol_train
            del class_train

        # Validation...
        val_err = 0.0
        val_acc = 0.0
        val_k_acc = 0.0
        val_l_acc = 0.0
        val_n_acc = 0.0
        val_batches = 0.0
        val_k_batches = 0.0
        val_l_batches = 0.0
        val_n_batches = 0.0

        val_batch_size = 500
        # for vol_f in val_vol_list:
        #     # get validation file name
        #     class_f = ret_class_file(vol_f, val_class_list)
        #     # read from disk validation volume and classification files
        #     vol_val = np.load(val_vol_path + "\\" + vol_f)
        #     class_val = np.load(val_class_path + "\\" + class_f)
        #
        #     for batch in iterate_minibatches(vol_val, class_val, val_batch_size, shuffle=False):
        #         inputs, targets = batch
        #         r = len(inputs) / float(val_batch_size)
        #         err, acc = val_fn(inputs, targets)
        #         val_err += r * err
        #         val_acc += r * acc
        #         val_batches += r
        #
        #     # get kidney validation patches
        #     d = len(vol_val) / 3
        #     kidneys = vol_val[0:d]
        #     kidneys_class = class_val[0:d]
        #     # get liver validation patches
        #     livers = vol_val[d:2 * d]
        #     livers_class = class_val[d:2 * d]
        #     # get nothing validation patches
        #     nothings = vol_val[2 * d:3 * d]
        #     nothings_class = class_val[2 * d:3 * d]
        #
        #     for batch in iterate_minibatches(kidneys, kidneys_class, val_batch_size, shuffle=False):
        #         inputs, targets = batch
        #         r = len(inputs) / float(val_batch_size)
        #         err, acc = val_fn(inputs, targets)
        #         val_k_acc += r * acc
        #         val_k_batches += r
        #
        #     for batch in iterate_minibatches(livers, livers_class, val_batch_size, shuffle=False):
        #         inputs, targets = batch
        #         r = len(inputs) / float(val_batch_size)
        #         err, acc = val_fn(inputs, targets)
        #         val_l_acc += r * acc
        #         val_l_batches += r
        #
        #     for batch in iterate_minibatches(nothings, nothings_class, val_batch_size, shuffle=False):
        #         inputs, targets = batch
        #         r = len(inputs) / float(val_batch_size)
        #         err, acc = val_fn(inputs, targets)
        #         val_n_acc += r * acc
        #         val_n_batches += r
        #
        #         # free numpy array memory
        #     del vol_val
        #     del class_val
        #
        # deal with divison by zero
        # if (val_batches == 0):
        #     val_batches = 1
        # if (train_batches == 0):
        #     train_batches = 1

    #     # print training results for current epoch and save loss function to array
    #     print "finished epoch num = %d training with the following results:" % itr
    #     print "Training took t = %f seconds" % (time.time() - start_time)
    #     print "Training loss = %f" % (train_err / train_batches)
    #     print "Validation loss = %f" % (val_err / val_batches)
    #     print "Validation Accuracy = %f" % (val_acc / val_batches * 100)
    #     print "Kidneys validation Accuracy = %f" % (val_k_acc / val_batches * 100)
    #     print "Livers validation Accuracy = %f" % (val_l_acc / val_batches * 100)
    #     print "Nothings validation Accuracy = %f" % (val_n_acc / val_batches * 100)
    #
    #     # save results to an array
    #     train_loss_arr.append(train_err / train_batches)
    #     val_loss_arr.append(val_err / val_batches)
    #     val_acc_arr.append(val_acc / val_batches * 100)
    #     val_kid_acc_arr.append(val_k_acc / val_batches * 100)
    #     val_liver_acc_arr.append(val_l_acc / val_batches * 100)
    #     val_nothing_acc_arr.append(val_n_acc / val_batches * 100)
    #
    #     # check if dest path for results exsits if not create new path
    #     if not (os.path.exists(res_path)):
    #         os.makedirs(res_path)
    #
    #         # for each 5 epochs save network params and data so far to disk
    #     if (itr % 5 == 0):
    #         print "saving params after iter = " + str(itr) + " ..."
    #         # 1. save the cnn params to disk
    #         np.savez(res_path + "\\" + "model_itr" + str(itr) + ".npz", *lasagne.layers.get_all_param_values(network))
    #         # 2. save loss function value array
    #         np.save(res_path + "\\" + "train_loss_function_itr" + str(itr) + ".np", np.array(train_loss_arr))
    #         # 3. save loss function for validation set
    #         np.save(res_path + "\\" + "validation_loss_function_itr" + str(itr) + ".np", np.array(val_loss_arr))
    #         # 4. save accuracy array for validation set
    #         np.save(res_path + "\\" + "validation_accuracy_itr" + str(itr) + ".np", np.array(val_acc_arr))
    #         # 5. save accuracy for kidneys validation
    #         np.save(res_path + "\\" + "validation_accuracy_k_itr" + str(itr) + ".np", np.array(val_kid_acc_arr))
    #         # 6. save accuracy for kidneys validation
    #         np.save(res_path + "\\" + "validation_accuracy_l_itr" + str(itr) + ".np", np.array(val_liver_acc_arr))
    #         # 7. save accuracy for kidneys validatio
    #         np.save(res_path + "\\" + "validation_accuracy_n_itr" + str(itr) + ".np", np.array(val_nothing_acc_arr))
    #
    #     # check if reached saturation if yes stop
    #
    #     if (stop_sat):
    #         # start checking after 5 iterrations
    #         if (itr >= 5):
    #             mean_t_loss = np.mean(np.array(train_loss_arr[itr - 5:itr]))
    #             last_t_loss = train_err / train_batches
    #             print mean_t_loss
    #             print last_t_loss
    #             # if last train loss is larger than moving avg train loss break
    #             if (last_t_loss >= mean_t_loss):
    #                 print "reached training loss saturation. stopping training..."
    #                 break
    #
    # # finished running all the epochs now save to disk the following params:
    # print "Done training.Saving params of last epoch, itr = " + str(itr)
    # # 1. save the cnn params to disk
    # np.savez(res_path + "\\" + "model_itr" + str(itr) + ".npz", *lasagne.layers.get_all_param_values(network))
    # # 2. save loss function value array
    # np.save(res_path + "\\" + "train_loss_function_itr" + str(itr) + ".np", np.array(train_loss_arr))
    # # 3. save loss function for validation set
    # np.save(res_path + "\\" + "validation_loss_function_itr" + str(itr) + ".np", np.array(val_loss_arr))
    # # 4. save accuracy array for validation set
    # np.save(res_path + "\\" + "validation_accuracy_itr" + str(itr) + ".np", np.array(val_acc_arr))
    # # 5. save accuracy for kidneys validation
    # np.save(res_path + "\\" + "validation_accuracy_k_itr" + str(itr) + ".np", np.array(val_kid_acc_arr))
    # # 6. save accuracy for kidneys validation
    # np.save(res_path + "\\" + "validation_accuracy_l_itr" + str(itr) + ".np", np.array(val_liver_acc_arr))
    # # 7. save accuracy for kidneys validatio
    # np.save(res_path + "\\" + "validation_accuracy_n_itr" + str(itr) + ".np", np.array(val_nothing_acc_arr))


###############################################################################
#                                                                             #
#                                Running Training                             #
#                                                                             #
###############################################################################;

# paths of training volumes and classification files
vol_src_path = "/home/guy/project/research_tensor_Integration_lasagna/DATA_SET/Test/Volumes"
seg_src_path = "/home/guy/project/research_tensor_Integration_lasagna/DATA_SET/Test/Segmentations"
vol_dest_path = "/home/guy/project/research_tensor_Integration_lasagna/DATA_SET/Test/Train/Volumes"
seg_dest_path = "/home/guy/project/research_tensor_Integration_lasagna/DATA_SET/Test/Train/Segmentations"


train_vol_path = "/home/guy/project/research_tensor_Integration_lasagna/DATA_SET/Test/Train/Volumes"
train_class_path = "/home/guy/project/research_tensor_Integration_lasagna/DATA_SET/Test/Train/Segmentations"

# paths of validation volumes and classification files
# val_vol_path = "Val\Volumes"
# val_class_path = "Val\Class"
# results path
res_path = "results"

# # network params file path
# network_params_path = "results\\model_itr99.npz"
# # trainig loss path
# train_loss_path = "results\\train_loss_function_itr99.np.npy"
# # trainig loss path
# valid_loss_path = "results\\validation_loss_function_itr99.np.npy"
# # trainig loss path
# valid_acc_path = "results\\validation_accuracy_itr99.np.npy"

# print "running main..."
train_cnn(train_vol_path, train_class_path, "", "", "",
          "", "", "", res_path, 100, False, False)
print "done"









