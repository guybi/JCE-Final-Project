import numpy as np
import SimpleITK as stk
import os
import random
from PIL import Image

# classification index value of nothing, liver and kidney
nothing = 0
liver = 1
kidney = 2

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
        im = im.resize((d[1] / r, d[2] / r), Image.BICUBIC)  # reduce image size
        tmp = np.array(im)  # convert back to numpy array
        res_list.append(tmp)

    return np.array(res_list)


def seg_size_reduce(input_im, r, start_im, end_im):
    d = input_im.shape
    res_list = list()

    for i in range(start_im, end_im):
        im = Image.fromarray(input_im[i, :, :])
        im = im.resize((d[1] / r, d[2] / r), Image.NEAREST)  # reduce image size
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
    print "List of input files:"
    print vol_list

    if not (os.path.exists(vol_dest_path)):
        os.makedirs(vol_dest_path)

    if not (os.path.exists(seg_dest_path)):
        os.makedirs(seg_dest_path)

    k = 0

    for f in vol_list:

        print '######################################'

        # read volume
        print 'read file...', f
        tmp_img = stk.ReadImage(vol_src_path + '\\' + f)
        input_vol = stk.GetArrayFromImage(tmp_img)

        # reduce volume to relevant size
        print 'reduce image size...'
        d = input_vol.shape
        if (start > d[0] or end > d[0]):
            start = 0
            end = d[0]
        input_vol = vol_size_reduce(input_vol.astype(np.float32), downsample, start, end)

        # read liver segmentation
        print 'read liver segmentation...'
        tmp = stk.ReadImage(seg_src_path + "\\" + getSegFileName(seg_src_path, f, liverId))  # read liver segmentation
        liver_seg = stk.GetArrayFromImage(tmp)
        liver_seg = seg_size_reduce(liver_seg.astype(np.float32), downsample, start, end)  # reduce size of liver_seg image

        # read left kidney segmentation
        print 'read left kidney segmentation...'
        tmp = stk.ReadImage(seg_src_path + "\\" + getSegFileName(seg_src_path, f, kindeyLId))
        left_kidney_seg = stk.GetArrayFromImage(tmp)
        left_kidney_seg = seg_size_reduce(left_kidney_seg.astype(np.float32), downsample, start,
                                             end)  # reduce size of left_kid_seg image

        # read right kidney segmentation
        print 'read right kidney segmentation...'
        tmp = stk.ReadImage(seg_src_path + "\\" + getSegFileName(seg_src_path, f, kidneyRId))  # read left kidney seg
        right_kidney_seg = stk.GetArrayFromImage(tmp)
        right_kidney_seg = seg_size_reduce(right_kidney_seg.astype(np.float32), downsample, start,
                                              end)  # reduce size of right_kid_seg image

        # make one kidney segmentation file
        print 'merge left and right kidney segmentation...'
        kidney_seg = np.add(right_kidney_seg, left_kidney_seg)

        # turn segmentation matrices into 1.0 or 0.0 values
        kidney_seg[np.where(kidney_seg > 1)] = 1
        liver_seg[np.where(liver_seg > 1)] = 1

        # get initial random shift number
        g_r_count = 0  # counter for good iters where #kidneys/#livers >= klr
        r_count = 0  # counter for total number of itterations
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
                    print "shifts are: dx = " + str(dx) + " dy = " + str(dy)
                    print str(kidney_count) + " kidney patches, " + str(liver_count) + " liver patches and " + str(
                        nothing_count) + \
                          " the rest"
                    np.save(vol_dest_path + "\\Volume_patchsize_" + str(patch_size) + "_" + str(k) + "_xshift" + str(
                        dx) + "_yshift" + str(dy), shifted_input_vol)
                    np.save(seg_dest_path + "\\Classification_patchsize_" + str(patch_size) + "_" + str(
                        k) + "_xshift" + str(dx) + "_yshift" + str(dy), y_res)
                    r_count += 1
                    g_r_count += 1

            shifts_list.append([dx, dy])  # mark used rabndom shifts pair

        print "saved " + str(g_r_count) + " random shifted files out of " + str(dxy)
        k += 1

    print '######################################'

if __name__ == '__main__':

    env = 'test'

    seg_ratio = 0.75
    klr = 3  # in percentage

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

    prep_data(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, seg_ratio, klr)


