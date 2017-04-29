import os as os
import numpy as np

def isTrainClassExisiting(vol_dest_path,seg_dest_path):

    if (os.path.exists(vol_dest_path) and os.path.exists(seg_dest_path)):
        vol_list = os.listdir(vol_dest_path)
        seg_list = os.listdir(seg_dest_path)
        if(len(vol_list) != 0 or len(seg_list) != 0):
            print('Shifted data found.')
            return True

    return False


def norm_data_rand(vol, cat):

    nothing = 0
    liver = 1
    kidney = 2

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
