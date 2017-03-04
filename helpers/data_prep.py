import numpy as np
import SimpleITK as stk
import os
import random
from PIL import Image
from random import shuffle

# classification index value of nothing, liver and kidney
nothing = 0
liver = 1
kidney = 2


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


def prep_data(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr):

    start = 300
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
        tmp = stk.ReadImage(seg_src_path + "/" + getSegFileName(seg_src_path, f, liverId))  # read liver segmentation
        liver_seg = stk.GetArrayFromImage(tmp)
        liver_seg = seg_size_reduce(liver_seg.astype(np.float32), downsample, start, end)  # reduce size of liver_seg image

        # read left kidney segmentation
        print('read left kidney segmentation...')
        tmp = stk.ReadImage(seg_src_path + "/" + getSegFileName(seg_src_path, f, kindeyLId))
        left_kidney_seg = stk.GetArrayFromImage(tmp)
        left_kidney_seg = seg_size_reduce(left_kidney_seg.astype(np.float32), downsample, start,
                                             end)  # reduce size of left_kid_seg image

        # read right kidney segmentation
        print('read right kidney segmentation...')
        tmp = stk.ReadImage(seg_src_path + "/" + getSegFileName(seg_src_path, f, kidneyRId))  # read left kidney seg
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
                shifted_liver_seg = Im2Blks(liver_seg, patch_size, dx, dy)
                shifted_kidney_seg = Im2Blks(kidney_seg, patch_size, dx, dy)

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


def norm_data(vol_src_path, seg_src_path):

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
        vol = np.load(vol_src_path + "\\" + f)  # load volume data
        class_f = ret_class_file(f, class_list)  # get class file name
        cat = np.load(seg_src_path + "\\" + class_f)  # load class data

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
            for k in range(r):
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


def data_load(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr):
    shifted_input_vol = []
    y_res = []
    if (os.path.exists(vol_dest_path) and os.path.exists(seg_dest_path)):
        vol_list = os.listdir(vol_dest_path)
        seg_list = os.listdir(seg_dest_path)
        if(len(vol_list) != 0 or len(seg_list) != 0):
            print('Shifted data found.')
            return;
        else:
            print('Shifted data directory is empty. Prepare data...')
            prep_data(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio,
                                                 klr)
            print('Normalize data...')
            norm_data(vol_dest_path, seg_dest_path)
    else:
        print('No shifted data found. Prepare data...')
        prep_data(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)
        print('Normalize data...')
        norm_data(vol_dest_path, seg_dest_path)


def norm_data_rand(vol, cat):
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
        r = int(len(liver_ind[0]) / len(kidney_ind[0]))
        tmp = vol[kidney_ind[0]]
        d = tmp.shape
        kidneys = np.zeros((r * d[0], d[1], d[2], d[3]), dtype=np.float32)
        for k in range(r):
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

    return new_vol, new_class
