import numpy as np
import os
import dataHelper as dh


# function receives:
#       vol_src_path - path of input CT volume files
#       seg_src_path - path of input CT segmentation files
#       val_vol_dest_path - destination path of validation volume files
#       val_seg_dest_path - destination path of validation segmentation files
#       val_num_ind - list of file indexes. assuming files with such indexes exist.
#
#       function creates validation files from files with indexes in val_num_ind list
#       and saves them to disk. Validation files are such that there is an equal amount of
#       kidney, liver and nothing patches
#
def prepare_val_train_data(vol_src_path, seg_src_path, val_vol_dest_path, val_seg_dest_path, val_num_ind):
    vol_list = os.listdir(vol_src_path)  # get list of validation volumes
    class_list = os.listdir(seg_src_path)  # get list of validation classification arrays

    # check if dest path exsits if not create new path
    if not (os.path.exists(val_vol_dest_path)):
        os.makedirs(val_vol_dest_path)

    if not (os.path.exists(val_seg_dest_path)):
        os.makedirs(val_seg_dest_path)

    # create sub list with validation file names
    val_vol_list = list()
    for ind in val_num_ind:
        for f in vol_list:
            split_fn = f.split("_")
            if (split_fn[3] == str(ind)):
                val_vol_list.append(f)

    for f in val_vol_list:
        vol = np.load(vol_src_path + "/" + f)  # load volume data
        class_f = dh.ret_class_file(f, class_list)  # get class file name
        cat = np.load(seg_src_path + "/" + class_f)  # load class data

        # get indexes of each type of patch
        liver_ind = np.where(cat == dh.liver)
        kidney_ind = np.where(cat == dh.kidney)
        nothing_ind = np.where(cat == dh.nothing)

        kidneys = vol[kidney_ind[0], :, :, :]  # getkidney patches
        # get liver patches in random order
        tmp = np.array(vol[liver_ind[0], :, :, :], dtype=np.float32)
        np.random.shuffle(tmp)
        livers = tmp
        # get nothing patches in random order
        tmp = np.array(vol[nothing_ind[0], :, :, :], dtype=np.float32)
        np.random.shuffle(tmp)
        nothings = tmp

        # create a new classification array with equal amount of liver, kidney and nothing patches
        # first elements shall be kidneys, then livers and then nothing patches
        # assuming: there's less kidney patches than liver and nothing
        d = kidneys.shape
        new_class = np.zeros(3 * d[0], dtype=np.int32)
        new_class[0:d[0]] = dh.kidney
        new_class[d[0]:2 * d[0]] = dh.liver
        new_class[2 * d[0]:] = dh.nothing

        # create a new and smaller numpy volumes array
        new_vol = np.zeros((3 * d[0], d[1], d[2], d[3]), dtype=np.float32)
        new_vol[0:d[0]] = kidneys
        new_vol[d[0]:2 * d[0]] = livers[0:d[0]]
        new_vol[2 * d[0]:] = nothings[0:d[0]]

        # save files to path
        np.save(val_vol_dest_path + "/" + f, new_vol)
        np.save(val_seg_dest_path + "/" + class_f, new_class)

        # delete original files
        os.remove(vol_src_path + "/" + f)
        os.remove(seg_src_path + "/" + class_f)


###############################################################################
# run code
###############################################################################

# to be run after prep_data
vol_src_path = "/home/guy/project/Project_new/DATA_SET/Test/Train/Volumes"
seg_src_path = "/home/guy/project/Project_new/DATA_SET/Test/Train/Segmentations"

vol_dest_path = "/home/guy/project/Project_new/DATA_SET/Test/Val/Volumes"
seg_dest_path = "/home/guy/project/Project_new/DATA_SET/Test/Val/Class"


validation_files_ind = [1, 19]

prepare_val_train_data(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, validation_files_ind)
print "done."










