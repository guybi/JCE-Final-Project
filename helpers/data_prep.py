import numpy as np
import SimpleITK as stk
import os, sys
import random
from PIL import Image
from random import shuffle

# classification index value of nothing, liver and kidney
nothing = 0
liver = 1
kidney = 2
spleen = 3
aorta = 4


def get_folders_dir(user, env):
    if (user == 'tal'):
        if (env == 'test'):
            vol_src_path = "C:\\CT\\Test\\Volumes"
            seg_src_path = "C:\\CT\\Test\\Segmentations"
            vol_dest_path = "C:\\CT\\Test\\Train\\Volumes"
            seg_dest_path = "C:\\CT\\Test\\Train\\Class"
            train_vol_path = "C:\\CT\\Test\\Train\\Volumes"
            train_class_path = "C:\\CT\\Test\\Train\\Class"
            val_vol_path = "C:\\CT\\Test\\Val\\Volumes"
            val_class_path = "C:\\CT\\Test\\Val\\Class"
            weights_dir = "C:\\CT\\Test\\Weights"
            predict_segmentations_dir = "C:\\CT\\Test\\Train\\PredictSegmentations"
        elif (env == 'prod'):
            vol_src_path = "C:\\CT\\Volumes"
            seg_src_path = "C:\\CT\\Segmentations"
            vol_dest_path = "C:\\CT\\Train\\Volumes"
            seg_dest_path = "C:\\CT\\Train\\Class"
            train_vol_path = "C:\\CT\\Train\\Volumes"
            train_class_path = "C:\\CT\\Train\\Class"
            val_vol_path = "C:\\CT\\Val\\Volumes"
            val_class_path = "C:\\CT\\Val\\Class"
            weights_dir = "C:\\CT\\Weights"
            predict_segmentations_dir = "C:\\CT\\Train\\PredictSegmentations"

    elif (user == 'guy'):
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

    elif (user == 'assaf'):
        if (env == 'test'):
            vol_src_path = ""
            seg_src_path = ""
            vol_dest_path = ""
            seg_dest_path = ""
            train_vol_path = ""
            train_class_path = ""
            val_vol_path = ""
            val_class_path = ""

        elif (env == 'prod'):
            vol_src_path = ""
            seg_src_path = ""
            vol_dest_path = ""
            seg_dest_path = ""
            train_vol_path = ""
            train_class_path = ""
            val_vol_path = ""
            val_class_path = ""

    return vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, train_vol_path,\
           train_class_path, train_class_path, val_vol_path, val_class_path, weights_dir, predict_segmentations_dir


def get_test_folders_dir(user, env):
    if (user == 'tal'):
        if (env == 'test'):
            test_vol_src_path = "C:\\CT\\Test\\Volumes"
            test_seg_src_path = "C:\\CT\\Test\\Segmentations"
            test_vol_dest_path = "C:\\CT\\Test\\Train\\Volumes"
            test_seg_dest_path = "C:\\CT\\Test\\Train\\Class"
            weights_dir = "C:\\CT\\Test\\Weights"
            predict_segmentations_dir = "C:\\CT\\Test\\Train\\PredictSegmentations"
        elif (env == 'prod'):
            test_vol_src_path = "C:\\CT\\Volumes"
            test_seg_src_path = "C:\\CT\\Segmentations"
            test_vol_dest_path = "C:\\CT\\Test\\Train\\Volumes"
            test_seg_dest_path = "C:\\CT\\Test\\Train\\Class"
            weights_dir = "C:\\CT\\Weights"
            predict_segmentations_dir = "C:\\CT\\Train\\PredictSegmentations"

    elif (user == 'guy'):
        if (env == 'test'):
            test_vol_src_path = "/home/guy/project/CT/Test/Volumes"
            test_seg_src_path = "/home/guy/project/CT/Test/Segmentations"
            test_vol_dest_path = "C:\\CT\\Test\\Train\\Volumes"
            test_seg_dest_path = "C:\\CT\\Test\\Train\\Class"
            weights_dir = "/home/guy/project/CT/Test/Weights"
            predict_segmentations_dir = "/home/guy/project/CT/Test/PredictSegmentations"

        elif (env == 'prod'):
            test_vol_src_path = "/home/guy/project/CT/Volumes"
            test_seg_src_path = "/home/guy/project/CT/Segmentations"
            test_vol_dest_path = "C:\\CT\\Test\\Train\\Volumes"
            test_seg_dest_path = "C:\\CT\\Test\\Train\\Class"
            weights_dir = "/home/guy/project/CT/Weights"
            predict_segmentations_dir = "/home/guy/project/CT/PredictSegmentations"

    elif (user == 'assaf'):
        if (env == 'test'):
            test_vol_src_path = ""
            test_seg_src_path = ""
            test_vol_dest_path = ""
            test_seg_dest_path = ""
            weights_dir = ""
            predict_segmentations_dir = ""

        elif (env == 'prod'):
            test_vol_src_path = ""
            test_seg_src_path = ""
            test_vol_dest_path = ""
            test_seg_dest_path = ""
            weights_dir = ""
            predict_segmentations_dir = ""

    return test_vol_src_path, test_seg_src_path, test_vol_dest_path, test_seg_dest_path, weights_dir, predict_segmentations_dir


def randomize_file_list(file_list):
    tmp = list(file_list) #copy list object
    shuffle(tmp) # TODO use tf shuffle
    return tmp


def getSegFileName(path, file_name,index):
    seg_list = os.listdir(path)
    sub_name = file_name[0:len(file_name)-7] + "_" + str(index)
    for f in seg_list:
        if (f.startswith(sub_name)):
            return f
    return "Fail"


def vol_size_reduce(input_im, r, start_im, end_im):
    d = input_im.shape
    res_list = list()

    for i in range(start_im, end_im):
        im = Image.fromarray(input_im[i, :, :])
        im = im.resize((int(d[1] / r), int(d[2] / r)), Image.BICUBIC)  # reduce image size
        tmp = np.array(im)  # convert back to numpy array
        res_list.append(tmp)

    return np.array(res_list)


def seg_size_reduce(input_im, r, start_im, end_im):
    d = input_im.shape
    res_list = list()

    for i in range(start_im, end_im):
        im = Image.fromarray(input_im[i, :, :])
        im = im.resize((int(d[1] / r), int(d[2] / r)), Image.NEAREST)  # reduce image size
        tmp = np.array(im)  # convert back to numpy array
        res_list.append(tmp)

    return np.array(res_list)


def Im2Blks(Im, BlkSz, x_shift=0, y_shift=0, is_vol=True):
    # assuming image size is symmetric nxn and BlkSz < 512 and BlkSz % 2 ==0

    # check dimension of Image
    d = Im.shape
    if (len(d) == 2):  # if 2d image
        L = [1, d[0], d[1]]  # if 2d image
    else:
        L = d  # if 3d image

    # shift Im in x axis and y axis by x_shift and y_shift pixels
    if (is_vol):
        shiftedIm = -1000 * np.ones([L[0], L[1], L[2]], dtype=np.float32)
    else:
        shiftedIm = np.zeros([L[0], L[1], L[2]], dtype=np.float32)

    if (len(d) == 2):
        shiftedIm[:, :L[1] - x_shift, :L[2] - y_shift] = Im[x_shift:L[1], y_shift:L[2]]
    else:
        shiftedIm[:, :L[1] - x_shift, :L[2] - y_shift] = Im[:, x_shift:L[1], y_shift:L[2]]

    # enlarge image with negative -1000 values such that Im.size%BlkSz == 0
    if (L[1] % BlkSz != 0):
        row_col_add = BlkSz - L[1] % BlkSz
        if (is_vol):
            tmpIm = -1000 * np.ones([L[0], L[1] + row_col_add, L[2] + row_col_add], dtype=np.float32)
        else:
            tmpIm = np.zeros([L[0], L[1] + row_col_add, L[2] + row_col_add], dtype=np.float32)
        L = tmpIm.shape
        tmpIm[:, row_col_add / 2:L[1] - row_col_add / 2, row_col_add / 2:L[2] - row_col_add / 2] = shiftedIm
    else:
        tmpIm = shiftedIm

    # reshape Image to a collection of patches
    shape = [L[0], L[1] / BlkSz, L[2] / BlkSz, BlkSz, BlkSz]
    strides = [4 * L[1] * L[2], 4 * BlkSz * L[2], 4 * BlkSz, 4 * L[2], 4]

    Blk = np.lib.stride_tricks.as_strided(tmpIm, shape=shape, strides=strides)
    Blk = np.reshape(Blk, [L[0] * L[1] * L[2] / BlkSz ** 2, 1, BlkSz, BlkSz])
    return Blk

#function receives: vol_fn - volume file name
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


def prep_data_liver_kidneys(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr):
    """
    prepare data for liver and kindeys detection
    """
    start = 245
    end = 600
    downsample = 1
    dxy = 4
    patch_size = 14

    liverId = 58
    kindeyLId = 29662
    kidneyRId = 29663


    vol_list = os.listdir(vol_src_path)
    print("List of input files:")
    print(vol_list)

    if not (os.path.exists(vol_dest_path)):
        os.makedirs(vol_dest_path)

    if not (os.path.exists(seg_dest_path)):
        os.makedirs(seg_dest_path)

    k = 0

    for f in vol_list:

        print('###############################################################################')

        # read volume
        print('read file...', f)
        tmp_img = stk.ReadImage(vol_src_path + '/' + f)
        input_vol = stk.GetArrayFromImage(tmp_img)

        # reduce volume to relevant size
        print('reduce image size...')
        d = input_vol.shape
        if (start > d[0] or end > d[0]):
            start = 0
            end = d[0]
        input_vol = vol_size_reduce(input_vol.astype(np.float32), downsample, start, end)

        # read liver segmentation
        print('read liver segmentation...')
        file_name = getSegFileName(seg_src_path, f, liverId)
        if file_name == 'Fail':
            print('segmentation not found. exiting...')
            sys.exit(0)
        tmp = stk.ReadImage(seg_src_path + "/" + file_name)  # read liver segmentation
        liver_seg = stk.GetArrayFromImage(tmp)
        liver_seg = seg_size_reduce(liver_seg.astype(np.float32), downsample, start, end)  # reduce size of liver_seg image

        # read left kidney segmentation
        print('read left kidney segmentation...')
        file_name = getSegFileName(seg_src_path, f, kindeyLId)
        if file_name == 'Fail':
            print('segmentation not found. exiting...')
            sys.exit(0)
        tmp = stk.ReadImage(seg_src_path + "/" + file_name)
        left_kidney_seg = stk.GetArrayFromImage(tmp)
        left_kidney_seg = seg_size_reduce(left_kidney_seg.astype(np.float32), downsample, start,
                                             end)  # reduce size of left_kid_seg image

        # read right kidney segmentation
        print('read right kidney segmentation...')
        file_name = getSegFileName(seg_src_path, f, kidneyRId)
        if file_name == 'Fail':
            print('segmentation not found. exiting...')
            sys.exit(0)
        tmp = stk.ReadImage(seg_src_path + "/" + file_name)  # read left kidney seg
        right_kidney_seg = stk.GetArrayFromImage(tmp)
        right_kidney_seg = seg_size_reduce(right_kidney_seg.astype(np.float32), downsample, start,
                                              end)  # reduce size of right_kid_seg image

        # make one kidney segmentation file
        print('merge left and right kidney segmentation...')
        kidney_seg = np.add(right_kidney_seg, left_kidney_seg)

        # turn segmentation matrices into 1.0 or 0.0 values
        kidney_seg[np.where(kidney_seg > 1)] = 1
        liver_seg[np.where(liver_seg > 1)] = 1

        # get initial random shift number
        g_r_count = 0  # counter for good iters where #kidneys/#livers >= klr
        r_count = 0  # counter for total number of iterations
        shifts_list = list()

        while (g_r_count < dxy and r_count < 20):
            f = False
            dx, dy = random.randint(0, 15), random.randint(0, 15)

            # check if dx dy was previously used
            for dxdy in shifts_list:
                if (dxdy[0] == dx and dxdy[1] == dy):
                    f = True
                    break

            if f:  # if used before go to next iteration with advancing counters
                continue
            else:  # if not used

                # create shifted images
                shifted_input_vol = Im2Blks(input_vol, patch_size, dx, dy)
                shifted_liver_seg = Im2Blks(liver_seg, patch_size, dx, dy, False)
                shifted_kidney_seg = Im2Blks(kidney_seg, patch_size, dx, dy, False)

                # remove irrelevant patches
                # if all values in patch are less than 10 hounsfeld  -> remove patch
                # if all values in ptach above 400 hounsfeld -> remove patch
                # if patch is not uniformly segmented  -> remove patch

                s = shifted_input_vol.shape  # size changed because array was reshaped
                ind_del = list()  # indexes of patches to delete
                res_array = list()  # result array nothing = 0, liver = 1, kidney = 2

                for i in range(s[0]):
                    if (((shifted_input_vol[i, :, :] < 10).all()) or ((shifted_input_vol[i, :, :] > 400).all()) or \
                                (np.sum(shifted_liver_seg[i, :, :]) > 0 and np.sum(
                                    shifted_liver_seg[i, :, :]) < seg_ratio * patch_size ** 2) or \
                                (np.sum(shifted_kidney_seg[i, :, :]) > 0 and np.sum(
                                    shifted_kidney_seg[i, :, :]) < seg_ratio * patch_size ** 2)):
                        ind_del.append(i)

                    # build result_seg_array
                    if (np.sum(shifted_liver_seg[i, 0, :, :]) >= seg_ratio * patch_size ** 2):
                        res_array.append(liver)
                    elif (np.sum(shifted_kidney_seg[i, 0, :, :]) >= seg_ratio * patch_size ** 2):
                        res_array.append(kidney)
                    else:
                        res_array.append(nothing)

                # delete irrelevant patches
                shifted_input_vol = np.delete(shifted_input_vol, ind_del, 0)
                y_res = np.delete(np.array(res_array), ind_del, 0)

                # calc and print stats for each volume
                kidney_count = len(y_res[np.where(y_res == 2)])
                liver_count = len(y_res[np.where(y_res == 1)])
                nothing_count = len(y_res) - kidney_count - liver_count

                # check if kidney to liver ration better than klr
                if kidney_count / float(liver_count) < klr / float(100):
                    r_count += 1
                else:
                    # save numpy input_vol array to disk
                    # save numpy y_res vector to disk
                    print("shifts are: dx = " + str(dx) + " dy = " + str(dy))
                    print(str(kidney_count) + " kidney patches, " + str(liver_count) + " liver patches and " + str(
                        nothing_count) + \
                          " the rest")
                    np.save(vol_dest_path + "/Volume_patchsize_" + str(patch_size) + "_" + str(k) + "_xshift" + str(
                        dx) + "_yshift" + str(dy), shifted_input_vol)
                    np.save(seg_dest_path + "/Classification_patchsize_" + str(patch_size) + "_" + str(
                        k) + "_xshift" + str(dx) + "_yshift" + str(dy), y_res)
                    r_count += 1
                    g_r_count += 1

            shifts_list.append([dx, dy])  # mark used random shifts pair

        print("saved " + str(g_r_count) + " random shifted files out of " + str(dxy))
        k += 1

    print('###############################################################################')

"""
prepare data for aorta and apleen detection
"""
def prep_data_aorta_spleen(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr):

    start = 245
    end = 385
    downsample = 1
    dxy = 4
    patch_size = 14

    spleenId = 86
    aortaId = 480

    vol_list = os.listdir(vol_src_path)
    print("List of input files:")
    print(vol_list)

    if not (os.path.exists(vol_dest_path)):
        os.makedirs(vol_dest_path)

    if not (os.path.exists(seg_dest_path)):
        os.makedirs(seg_dest_path)

    k = 0

    for f in vol_list:

        print('###############################################################################')

        # read volume
        print('read file...', f)
        tmp_img = stk.ReadImage(vol_src_path + '/' + f)
        input_vol = stk.GetArrayFromImage(tmp_img)

        # reduce volume to relevant size
        print('reduce image size...')
        d = input_vol.shape
        if (start > d[0] or end > d[0]):
            start = 0
            end = d[0]
        input_vol = vol_size_reduce(input_vol.astype(np.float32), downsample, start, end)

        # read spleen segmentation
        print('read spleen segmentation...')
        file_name = getSegFileName(seg_src_path, f, spleenId)
        if file_name == 'Fail':
            print('segmentation not found. exiting...')
            sys.exit(0)
        tmp = stk.ReadImage(seg_src_path + "/" + file_name)  # read spleen segmentation
        spleen_seg = stk.GetArrayFromImage(tmp)
        spleen_seg = seg_size_reduce(spleen_seg.astype(np.float32), downsample, start, end)  # reduce size of spleen_seg image

        #read aorta segmentation
        print('read aorta segmentation...')
        file_name = getSegFileName(seg_src_path, f, aortaId)
        if file_name == 'Fail':
            print('segmentation not found. exiting...')
            sys.exit(0)
        tmp = stk.ReadImage(seg_src_path + "/" + file_name) # read aorta segmentation
        aorta_seg = stk.GetArrayFromImage(tmp)
        aorta_seg = seg_size_reduce(aorta_seg.astype(np.float32), downsample, start, end) # reduce size of aorta_seg image

        # turn segmentation matrices into 1.0 or 0.0 values
        spleen_seg[np.where(spleen_seg > 1)] = 1
        aorta_seg[np.where(aorta_seg > 1)] = 1

        # get initial random shift number
        g_r_count = 0  # counter for good iters where #kidneys/#livers >= klr
        r_count = 0  # counter for total number of iterations
        shifts_list = list()

        while (g_r_count < dxy and r_count < 20):
            f = False
            dx, dy = random.randint(0, 15), random.randint(0, 15)

            # check if dx dy was previously used
            for dxdy in shifts_list:
                if (dxdy[0] == dx and dxdy[1] == dy):
                    f = True
                    break

            if f:  # if used before go to next iteration with advancing counters
                continue
            else:  # if not used

                # create shifted images
                shifted_input_vol = Im2Blks(input_vol, patch_size, dx, dy)
                shifted_spleen_seg = Im2Blks(spleen_seg, patch_size, dx, dy, False)
                shifted_aorta_seg = Im2Blks(aorta_seg, patch_size, dx, dy, False)

                # remove irrelevant patches
                # if all values in patch are less than 10 hounsfeld  -> remove patch
                # if all values in ptach above 400 hounsfeld -> remove patch
                # if patch is not uniformly segmented  -> remove patch

                s = shifted_input_vol.shape  # size changed because array was reshaped
                ind_del = list()  # indexes of patches to delete
                res_array = list()  # result array nothing = 0, liver = 1, kidney = 2

                for i in range(s[0]):
                    if (((shifted_input_vol[i, :, :] < 10).all()) or (
                    (shifted_input_vol[i, :, :] > 400).all()) or \
                                (np.sum(shifted_spleen_seg[i, :, :]) > 0 and np.sum(
                                    shifted_spleen_seg[i, :, :]) < seg_ratio * patch_size ** 2) or \
                                (np.sum(shifted_aorta_seg[i, :, :]) > 0 and np.sum(
                                    shifted_aorta_seg[i, :, :]) < seg_ratio * patch_size ** 2)):
                        ind_del.append(i)

                    # build result_seg_array
                    if (np.sum(shifted_spleen_seg[i, 0, :, :]) >= seg_ratio * patch_size ** 2):
                        res_array.append(spleen)
                    elif (np.sum(shifted_aorta_seg[i, 0, :, :]) >= seg_ratio * patch_size ** 2):
                        res_array.append(aorta)
                    else:
                        res_array.append(nothing)

                # delete irrelevant patches
                shifted_input_vol = np.delete(shifted_input_vol, ind_del, 0)
                y_res = np.delete(np.array(res_array), ind_del, 0)

                # calc and print stats for each volume
                spleen_count = len(y_res[np.where(y_res == 3)])
                aorta_count = len(y_res[np.where(y_res == 4)])
                nothing_count = len(y_res) - spleen_count - aorta_count

                # check if aorta to spleen ration better than klr
                if spleen_count == 0 or aorta_count == 0:
                    continue
                if spleen_count / float(aorta_count) < klr / float(100):
                    r_count += 1
                else:
                    # save numpy input_vol array to disk
                    # save numpy y_res vector to disk
                    print("shifts are: dx = " + str(dx) + " dy = " + str(dy))
                    print(str(spleen_count) + " spleen patches, " + str(
                        aorta_count) + " aorta patches and " + str(
                        nothing_count) + \
                          " the rest")
                    np.save(
                        vol_dest_path + "/Volume_patchsize_" + str(patch_size) + "_" + str(k) + "_xshift" + str(
                            dx) + "_yshift" + str(dy), shifted_input_vol)
                    np.save(seg_dest_path + "/Classification_patchsize_" + str(patch_size) + "_" + str(
                        k) + "_xshift" + str(dx) + "_yshift" + str(dy), y_res)
                    r_count += 1
                    g_r_count += 1

            shifts_list.append([dx, dy])  # mark used random shifts pair

        print("saved " + str(g_r_count) + " random shifted files out of " + str(dxy))
        k += 1

        print('###############################################################################')

"""
prepare data for liver, kidneys, aorta and apleen detection
"""
def prep_data(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr):

    start = 245
    end = 600
    downsample = 1
    dxy = 4
    patch_size = 14

    liverId = 58
    spleenId = 86
    aortaId = 480
    kindeyLId = 29662
    kidneyRId = 29663

    vol_list = os.listdir(vol_src_path)
    print("List of input files:")
    print(vol_list)

    if not (os.path.exists(vol_dest_path)):
        os.makedirs(vol_dest_path)

    if not (os.path.exists(seg_dest_path)):
        os.makedirs(seg_dest_path)

    k = 0

    for f in vol_list:

        print('###############################################################################')

        # read volume
        print('read file...', f)
        tmp_img = stk.ReadImage(vol_src_path + '/' + f)
        input_vol = stk.GetArrayFromImage(tmp_img)

        # reduce volume to relevant size
        print('reduce image size...')
        d = input_vol.shape
        if (start > d[0] or end > d[0]):
            start = 0
            end = d[0]
        input_vol = vol_size_reduce(input_vol.astype(np.float32), downsample, start, end)

        # read liver segmentation
        print('read liver segmentation...')
        file_name = getSegFileName(seg_src_path, f, liverId)
        if file_name == 'Fail':
            print('segmentation not found. exiting...')
            sys.exit(0)
        tmp = stk.ReadImage(seg_src_path + "/" + file_name)  # read liver segmentation
        liver_seg = stk.GetArrayFromImage(tmp)
        liver_seg = seg_size_reduce(liver_seg.astype(np.float32), downsample, start, end)  # reduce size of liver_seg image

        # read spleen segmentation
        print('read spleen segmentation...')
        file_name = getSegFileName(seg_src_path, f, spleenId)
        if file_name == 'Fail':
            print('segmentation not found. exiting...')
            sys.exit(0)
        tmp = stk.ReadImage(seg_src_path + "/" + file_name)  # read spleen segmentation
        spleen_seg = stk.GetArrayFromImage(tmp)
        spleen_seg = seg_size_reduce(spleen_seg.astype(np.float32), downsample, start,
                                     end)  # reduce size of spleen_seg image

        # read aorta segmentation
        print('read aorta segmentation...')
        file_name = getSegFileName(seg_src_path, f, aortaId)
        if file_name == 'Fail':
            print('segmentation not found. exiting...')
            sys.exit(0)
        tmp = stk.ReadImage(seg_src_path + "/" + file_name)  # read aorta segmentation
        aorta_seg = stk.GetArrayFromImage(tmp)
        aorta_seg = seg_size_reduce(aorta_seg.astype(np.float32), downsample, start,
                                    end)  # reduce size of aorta_seg image

        # read left kidney segmentation
        print('read left kidney segmentation...')
        file_name = getSegFileName(seg_src_path, f, kindeyLId)
        if file_name == 'Fail':
            print('segmentation not found. exiting...')
            sys.exit(0)
        tmp = stk.ReadImage(seg_src_path + "/" + file_name)
        left_kidney_seg = stk.GetArrayFromImage(tmp)
        left_kidney_seg = seg_size_reduce(left_kidney_seg.astype(np.float32), downsample, start,
                                             end)  # reduce size of left_kid_seg image

        # read right kidney segmentation
        print('read right kidney segmentation...')
        file_name = getSegFileName(seg_src_path, f, kidneyRId)
        if file_name == 'Fail':
            print('segmentation not found. exiting...')
            sys.exit(0)
        tmp = stk.ReadImage(seg_src_path + "/" + file_name)  # read left kidney seg
        right_kidney_seg = stk.GetArrayFromImage(tmp)
        right_kidney_seg = seg_size_reduce(right_kidney_seg.astype(np.float32), downsample, start,
                                              end)  # reduce size of right_kid_seg image

        # make one kidney segmentation file
        print('merge left and right kidney segmentations...')
        kidney_seg = np.add(right_kidney_seg, left_kidney_seg)

        # turn segmentation matrices into 1.0 or 0.0 values
        kidney_seg[np.where(kidney_seg > 1)] = 1
        spleen_seg[np.where(spleen_seg > 1)] = 1
        aorta_seg[np.where(aorta_seg > 1)] = 1
        liver_seg[np.where(liver_seg > 1)] = 1

        # get initial random shift number
        g_r_count = 0  # counter for good iters where #kidneys/#livers >= klr
        r_count = 0  # counter for total number of iterations
        shifts_list = list()

        while (g_r_count < dxy and r_count < 20):
            f = False
            dx, dy = random.randint(0, 15), random.randint(0, 15)

            # check if dx dy was previously used
            for dxdy in shifts_list:
                if (dxdy[0] == dx and dxdy[1] == dy):
                    f = True
                    break

            if f:  # if used before go to next iteration with advancing counters
                continue
            else:  # if not used

                # create shifted images
                shifted_input_vol = Im2Blks(input_vol, patch_size, dx, dy)
                shifted_liver_seg = Im2Blks(liver_seg, patch_size, dx, dy, False)
                shifted_kidney_seg = Im2Blks(kidney_seg, patch_size, dx, dy, False)
                shifted_spleen_seg = Im2Blks(spleen_seg, patch_size, dx, dy, False)
                shifted_aorta_seg = Im2Blks(aorta_seg, patch_size, dx, dy, False)

                # remove irrelevant patches
                # if all values in patch are less than 10 hounsfeld  -> remove patch
                # if all values in ptach above 400 hounsfeld -> remove patch
                # if patch is not uniformly segmented  -> remove patch

                s = shifted_input_vol.shape  # size changed because array was reshaped
                ind_del = list()  # indexes of patches to delete
                res_array = list()  # result array nothing = 0, liver = 1, kidney = 2

                for i in range(s[0]):
                    if (((shifted_input_vol[i, :, :] < 10).all()) or ((shifted_input_vol[i, :, :] > 400).all()) or \
                                (np.sum(shifted_liver_seg[i, :, :]) > 0 and np.sum(
                                    shifted_liver_seg[i, :, :]) < seg_ratio * patch_size ** 2) or \
                                (np.sum(shifted_kidney_seg[i, :, :]) > 0 and np.sum(
                                    shifted_kidney_seg[i, :, :]) < seg_ratio * patch_size ** 2) or \
                                (np.sum(shifted_spleen_seg[i, :, :]) > 0 and np.sum(
                                    shifted_spleen_seg[i, :, :]) < seg_ratio * patch_size ** 2) or \
                                (np.sum(shifted_aorta_seg[i, :, :]) > 0 and np.sum(
                                    shifted_aorta_seg[i, :, :]) < seg_ratio * patch_size ** 2)):
                        ind_del.append(i)

                    # build result_seg_array
                    if (np.sum(shifted_liver_seg[i, 0, :, :]) >= seg_ratio * patch_size ** 2):
                        res_array.append(liver)
                    elif (np.sum(shifted_kidney_seg[i, 0, :, :]) >= seg_ratio * patch_size ** 2):
                        res_array.append(kidney)
                    elif (np.sum(shifted_spleen_seg[i, 0, :, :]) >= seg_ratio * patch_size ** 2):
                        res_array.append(spleen)
                    elif (np.sum(shifted_aorta_seg[i, 0, :, :]) >= seg_ratio * patch_size ** 2):
                        res_array.append(aorta)
                    else:
                        res_array.append(nothing)

                # delete irrelevant patches
                shifted_input_vol = np.delete(shifted_input_vol, ind_del, 0)
                y_res = np.delete(np.array(res_array), ind_del, 0)

                # calc and print stats for each volume
                kidney_count = len(y_res[np.where(y_res == 2)])
                liver_count = len(y_res[np.where(y_res == 1)])
                spleen_count = len(y_res[np.where(y_res == 3)])
                aorta_count = len(y_res[np.where(y_res == 4)])
                nothing_count = len(y_res) - kidney_count - liver_count - spleen_count - aorta_count

                # check if kidney to liver ration better than klr
                if kidney_count / float(liver_count) < klr / float(100):
                    r_count += 1
                else:
                    # save numpy input_vol array to disk
                    # save numpy y_res vector to disk
                    print("shifts are: dx = " + str(dx) + " dy = " + str(dy))
                    print(str(kidney_count) + " kidney patches, " + str(liver_count) + " liver patches and " +
                          str(spleen_count) + " spleen patches, " + str(aorta_count) + " aorta patches and " +
                          str(nothing_count) + " the rest")
                    np.save(vol_dest_path + "/Volume_patchsize_" + str(patch_size) + "_" + str(k) + "_xshift" + str(
                        dx) + "_yshift" + str(dy), shifted_input_vol)
                    np.save(seg_dest_path + "/Classification_patchsize_" + str(patch_size) + "_" + str(
                        k) + "_xshift" + str(dx) + "_yshift" + str(dy), y_res)
                    r_count += 1
                    g_r_count += 1

            shifts_list.append([dx, dy])  # mark used random shifts pair

        print("saved " + str(g_r_count) + " random shifted files out of " + str(dxy))
        k += 1

    print('###############################################################################')


def norm_data_liver_kidneys(vol_src_path, seg_src_path):

    vol_list = os.listdir(vol_src_path)  # get list of validation volumes
    class_list = os.listdir(seg_src_path)  # get list of validation classification arrays

    # create sub list of file names
    val_vol_list = list()
    for ind in range(0, 18):
        for f in vol_list:
            split_fn = f.split("_")
            if (split_fn[3] == str(ind)):
                val_vol_list.append(f)

    for f in val_vol_list:
        print("normalizing data file: " + f)
        vol = np.load(vol_src_path + "/" + f)  # load volume data
        class_f = ret_class_file(f, class_list)  # get class file name
        cat = np.load(seg_src_path + "/" + class_f)  # load class data

        # get indexes of each type of patch
        liver_ind = np.where(cat == liver)
        kidney_ind = np.where(cat == kidney)
        nothing_ind = np.where(cat == nothing)

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
            for k in range(int(r)):
                kidneys[k * d[0]:(k + 1) * d[0], :, :, :] = vol[kidney_ind[0]]

        # create new classification array
        if (len(kidney_ind[0] != 0)):
            new_class = np.zeros(len(livers) + len(kidneys) + len(livers), dtype=np.int32)
            new_class[:len(livers)] = liver
            new_class[len(livers):len(livers) + len(kidneys)] = kidney
            new_class[len(livers) + len(kidneys):] = nothing
        else:
            new_class = np.zeros(2 * len(livers), dtype=np.int32)
            new_class[:len(livers)] = liver
            new_class[len(livers):] = nothing

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

        # overwrite original files
        np.save(vol_src_path + "\\" + f, new_vol)
        np.save(seg_src_path + "\\" + class_f, new_class)


def norm_data_aorta_spleen(vol_src_path, seg_src_path):

    vol_list = os.listdir(vol_src_path)  # get list of validation volumes
    class_list = os.listdir(seg_src_path)  # get list of validation classification arrays

    # create sub list of file names
    val_vol_list = list()
    for ind in range(0, 18):
        for f in vol_list:
            split_fn = f.split("_")
            if (split_fn[3] == str(ind)):
                val_vol_list.append(f)

    for f in val_vol_list:
        print("normalizing data file: " + f)
        vol = np.load(vol_src_path + "/" + f)  # load volume data
        class_f = ret_class_file(f, class_list)  # get class file name
        cat = np.load(seg_src_path + "/" + class_f)  # load class data

        # get indexes of each type of patch
        spleen_ind = np.where(cat == spleen)
        aorta_ind = np.where(cat == aorta)
        nothing_ind = np.where(cat == nothing)

        # get all spleen patches
        spleens = vol[spleen_ind[0], :, :, :]
        # get nothings patches in random order
        tmp = np.array(vol[nothing_ind[0], :, :, :], dtype=np.float32)
        np.random.shuffle(tmp)
        nothings = tmp
        # create extended aorta data
        if (len(aorta_ind[0] != 0)):
            r = len(spleen_ind[0]) / len(aorta_ind[0])
            tmp = vol[aorta_ind[0]]
            d = tmp.shape
            aortas = np.zeros((r * d[0], d[1], d[2], d[3]), dtype=np.float32)
            for k in range(int(r)):
                aortas[k * d[0]:(k + 1) * d[0], :, :, :] = vol[aorta_ind[0]]

        # create new classification array
        if (len(aorta_ind[0] != 0)):
            new_class = np.zeros(len(spleens) + len(aortas) + len(spleens), dtype=np.int32)
            new_class[:len(spleens)] = spleen
            new_class[len(spleens):len(spleens) + len(aortas)] = aorta
            new_class[len(spleens) + len(aortas):] = nothing
        else:
            new_class = np.zeros(2 * len(spleens), dtype=np.int32)
            new_class[:len(spleens)] = spleen
            new_class[len(spleens):] = nothing

        # create a new and smaller numpy volumes array
        if (len(aorta_ind[0] != 0)):
            new_vol = np.zeros((len(spleens) + len(aortas) + len(spleens), d[1], d[2], d[3]), dtype=np.float32)
            new_vol[:len(spleens), :, :, :] = spleens
            new_vol[len(spleens):len(spleens) + len(aortas), :, :, :] = aortas
            new_vol[len(spleens) + len(aortas):, :, :, :] = nothings[0:len(spleens)]
        else:
            new_vol = np.zeros((2 * len(spleens), d[1], d[2], d[3]), dtype=np.float32)
            new_vol[:len(spleens), :, :, :] = spleens
            new_vol[len(spleens):, :, :, :] = nothings[0:len(spleens)]

        # overwrite original files
        np.save(vol_src_path + "\\" + f, new_vol)
        np.save(seg_src_path + "\\" + class_f, new_class)


def data_load(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr):
    if os.path.exists(vol_dest_path) and os.path.exists(seg_dest_path):
        vol_list = os.listdir(vol_dest_path)
        seg_list = os.listdir(seg_dest_path)
        if len(vol_list) != 0 or len(seg_list) != 0:
            print('Data found.')
            return 1
        else:
            print('Data directory is empty. Prepare data...')
            # if organs == 'liver_kidneys':
            #     prep_data_liver_kidneys(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)
            #     print('Normalize data...')
            #     norm_data_liver_kidneys(vol_dest_path, seg_dest_path)
            # elif organs == 'spleen_aorta':
            #     prep_data_aorta_spleen(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)
            # else:
            #     return -1
            prep_data_liver_kidneys(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)
            print('Normalize data...')
            norm_data(vol_dest_path, seg_dest_path)

    else:
        print('No data found. Prepare data...')
        # if organs == 'liver_kidneys':
        #     prep_data_liver_kidneys(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)
        #     print('Normalize data...')
        #     norm_data_liver_kidneys(vol_dest_path, seg_dest_path)
        # elif organs == 'spleen_aorta':
        #     prep_data_aorta_spleen(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)
        # else:
        #     return -1
        prep_data_liver_kidneys(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)
        print('Normalize data...')
        norm_data(vol_dest_path, seg_dest_path)


def norm_data_rand_liver_kindeys(vol, cat):
    # get indexes of each type of patch
    liver_ind = np.where(cat == liver)
    kidney_ind = np.where(cat == kidney)
    nothing_ind = np.where(cat == nothing)

    # get all liver patches
    livers = vol[liver_ind[0], :, :, :]
    # get nothings patches in random order
    tmp = np.array(vol[nothing_ind[0], :, :, :], dtype=np.float32)
    np.random.shuffle(tmp)
    nothings = tmp
    # create extended kidneys data
    if len(kidney_ind[0] != 0):
        r = int(len(liver_ind[0]) / len(kidney_ind[0]))
        tmp = vol[kidney_ind[0]]
        d = tmp.shape
        kidneys = np.zeros((r * d[0], d[1], d[2], d[3]), dtype=np.float32)
        for k in range(r):
            kidneys[k * d[0]:(k + 1) * d[0], :, :, :] = vol[kidney_ind[0]]

    # create new classification array
    if len(kidney_ind[0] != 0):
        new_class = np.zeros(len(livers) + len(kidneys) + len(livers), dtype=np.int32)
        new_class[:len(livers)] = liver
        new_class[len(livers):len(livers) + len(kidneys)] = kidney
        new_class[len(livers) + len(kidneys):] = nothing
    else:
        new_class = np.zeros(2 * len(livers), dtype=np.int32)
        new_class[:len(livers)] = liver
        new_class[len(livers):] = nothing

    # create a new and smaller numpy volumes array
    if len(kidney_ind[0] != 0):
        new_vol = np.zeros((len(livers) + len(kidneys) + len(livers), d[1], d[2], d[3]), dtype=np.float32)
        new_vol[:len(livers), :, :, :] = livers
        new_vol[len(livers):len(livers) + len(kidneys), :, :, :] = kidneys
        new_vol[len(livers) + len(kidneys):, :, :, :] = nothings[0:len(livers)]
    else:
        new_vol = np.zeros((2 * len(livers), d[1], d[2], d[3]), dtype=np.float32)
        new_vol[:len(livers), :, :, :] = livers
        new_vol[len(livers):, :, :, :] = nothings[0:len(livers)]

    return new_vol, new_class


def norm_data_rand_spleen_aorta(vol, cat):
    # get indexes of each type of patch
    spleen_ind = np.where(cat == spleen)
    aorta_ind = np.where(cat == aorta)
    nothing_ind = np.where(cat == nothing)

    # get all liver patches
    spleens = vol[spleen_ind[0], :, :, :]
    # get nothings patches in random order
    tmp = np.array(vol[nothing_ind[0], :, :, :], dtype=np.float32)
    np.random.shuffle(tmp)
    nothings = tmp
    # create extended kidneys data
    if len(aorta_ind[0] != 0):
        r = int(len(spleen_ind[0]) / len(aorta_ind[0]))
        tmp = vol[aorta_ind[0]]
        d = tmp.shape
        aortas = np.zeros((r * d[0], d[1], d[2], d[3]), dtype=np.float32)
        for k in range(r):
            aortas[k * d[0]:(k + 1) * d[0], :, :, :] = vol[aorta_ind[0]]

    # create new classification array
    if len(aorta_ind[0] != 0):
        new_class = np.zeros(len(spleens) + len(aortas) + len(spleens), dtype=np.int32)
        new_class[:len(spleens)] = spleen
        new_class[len(spleens):len(spleens) + len(aortas)] = aorta
        new_class[len(spleens) + len(aortas):] = nothing
    else:
        new_class = np.zeros(2 * len(spleens), dtype=np.int32)
        new_class[:len(spleens)] = spleen
        new_class[len(spleens):] = nothing

    # create a new and smaller numpy volumes array
    if len(aorta_ind[0] != 0):
        new_vol = np.zeros((len(spleens) + len(aortas) + len(spleens), d[1], d[2], d[3]), dtype=np.float32)
        new_vol[:len(spleens), :, :, :] = spleens
        new_vol[len(spleens):len(spleens) + len(aortas), :, :, :] = aortas
        new_vol[len(spleens) + len(aortas):, :, :, :] = nothings[0:len(spleens)]
    else:
        new_vol = np.zeros((2 * len(spleens), d[1], d[2], d[3]), dtype=np.float32)
        new_vol[:len(spleens), :, :, :] = spleens
        new_vol[len(spleens):, :, :, :] = nothings[0:len(spleens)]

    return new_vol, new_class


def norm_data(vol, cat):
    # get indexes of each type of patch
    liver_ind = np.where(cat == liver)
    kidney_ind = np.where(cat == kidney)
    # spleen_ind = np.where(cat == spleen)
    # aorta_ind = np.where(cat == aorta)
    nothing_ind = np.where(cat == nothing)

    # get all liver patches
    livers = vol[liver_ind[0], :, :, :]
    # get nothings patches in random order
    tmp = np.array(vol[nothing_ind[0], :, :, :], dtype=np.float32)
    np.random.shuffle(tmp)
    nothings = tmp
    # create extended kidneys data
    if len(kidney_ind[0] != 0):
        r = int(len(liver_ind[0]) / len(kidney_ind[0]))
        tmp = vol[kidney_ind[0]]
        d = tmp.shape
        kidneys = np.zeros((r * d[0], d[1], d[2], d[3]), dtype=np.float32)
        for k in range(r):
            kidneys[k * d[0]:(k + 1) * d[0], :, :, :] = vol[kidney_ind[0]]

    # create new classification array
    if len(kidney_ind[0] != 0):
        new_class = np.zeros(len(livers) + len(kidneys) + len(livers), dtype=np.int32)
        new_class[:len(livers)] = liver
        new_class[len(livers):len(livers) + len(kidneys)] = kidney
        new_class[len(livers) + len(kidneys):] = nothing
    else:
        new_class = np.zeros(2 * len(livers), dtype=np.int32)
        new_class[:len(livers)] = liver
        new_class[len(livers):] = nothing

    # create a new and smaller numpy volumes array
    if len(kidney_ind[0] != 0):
        new_vol = np.zeros((len(livers) + len(kidneys) + len(livers), d[1], d[2], d[3]), dtype=np.float32)
        new_vol[:len(livers), :, :, :] = livers
        new_vol[len(livers):len(livers) + len(kidneys), :, :, :] = kidneys
        new_vol[len(livers) + len(kidneys):, :, :, :] = nothings[0:len(livers)]
    else:
        new_vol = np.zeros((2 * len(livers), d[1], d[2], d[3]), dtype=np.float32)
        new_vol[:len(livers), :, :, :] = livers
        new_vol[len(livers):, :, :, :] = nothings[0:len(livers)]

    return new_vol, new_class


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
        vol = np.load(vol_src_path + "\\" + f)  # load volume data
        class_f = ret_class_file(f, class_list)  # get class file name
        cat = np.load(seg_src_path + "\\" + class_f)  # load class data

        # get indexes of each type of patch
        liver_ind = np.where(cat == liver)
        kidney_ind = np.where(cat == kidney)
        spleen_ind = np.where(cat == spleen)
        aorta_ind = np.where(cat == aorta)
        nothing_ind = np.where(cat == nothing)

        kidneys = vol[kidney_ind[0], :, :, :] # get kidney patches

        # get liver patches in random order
        tmp = np.array(vol[liver_ind[0], :, :, :], dtype=np.float32)
        np.random.shuffle(tmp)
        livers = tmp

        # get spleen patches in random order
        tmp = np.array(vol[spleen_ind[0], :, :, :], dtype=np.float32)
        np.random.shuffle(tmp)
        spleens = tmp

        # get aorta patches in random order
        tmp = np.array(vol[aorta_ind[0], :, :, :], dtype=np.float32)
        np.random.shuffle(tmp)
        aortas = tmp

        # get nothing patches in random order
        tmp = np.array(vol[nothing_ind[0], :, :, :], dtype=np.float32)
        np.random.shuffle(tmp)
        nothings = tmp

        # create a new classification array with equal amount of liver, kidney and nothing patches
        # first elements shall be kidneys, then livers and then nothing patches
        # assuming: there's less kidney patches than liver and nothing
        d = kidneys.shape
        new_class = np.zeros(5 * d[0], dtype=np.int32)
        new_class[0:d[0]] = kidney
        new_class[d[0]:2 * d[0]:3] = liver
        new_class[d[0]:3 * d[0]:4] = spleen
        new_class[d[0]:4 * d[0]:5] = aorta
        new_class[5 * d[0]:] = nothing

        # create a new and smaller numpy volumes array
        new_vol = np.zeros((5 * d[0], d[1], d[2], d[3]), dtype=np.float32)
        new_vol[0:d[0]] = kidneys
        new_vol[d[0]:2 * d[0]:3] = livers[0:d[0]]
        new_vol[d[0]:3 * d[0]:4] = spleens
        new_vol[d[0]:4 * d[0]:5] = aortas
        new_vol[5 * d[0]:] = nothings[0:d[0]]

        # save files to path
        np.save(val_vol_dest_path + "\\" + f, new_vol)
        np.save(val_seg_dest_path + "\\" + class_f, new_class)

        # delete original files
        # os.remove(vol_src_path + "\\" + f)
        # os.remove(seg_src_path + "\\" + class_f)
