import numpy as np
import os
import dataHelper as dh

#function receives:
#       vol_src_path - path of input CT volume files
#       seg_src_path - path of input CT segmentation files
#       val_num_ind - list of file indexes. assuming files with such indexes exist.
#
#       function creates new data files such that the file has an equal number of liver, kidney and nothing patches.
#       It is done by copying kidnye patches several times, copying all liver patches and randomly reducing nothing
#       patches.
#
def norm_data(vol_src_path, seg_src_path, val_num_ind):
    
    vol_list = os.listdir(vol_src_path) #get list of validation volumes
    class_list = os.listdir(seg_src_path) #get list of validation classification arrays
     
    #create sub list of file names
    val_vol_list = list()
    for ind in val_num_ind:
        for f in vol_list:
            split_fn = f.split("_")
            if (split_fn[3] == str(ind)):
                val_vol_list.append(f)

    for f in val_vol_list:
        print "normalizing data file: " + f
        vol = np.load(vol_src_path + "\\" + f) #load volume data
        class_f = dh.ret_class_file(f,class_list) #get class file name
        cat = np.load(seg_src_path + "\\" + class_f) #load class data            
        
        #get indexes of each type of patch
        liver_ind = np.where(cat == dh.liver)
        kidney_ind = np.where(cat == dh.kidney)
        nothing_ind = np.where(cat == dh.nothing)
        
        
        #get all liver patches        
        livers = vol[liver_ind[0],:,:,:]
        #get nothings patches in random order
        tmp = np.array(vol[nothing_ind[0],:,:,:], dtype = np.float32) 
        np.random.shuffle(tmp)
        nothings = tmp
        #create extended kidneys data
        if (len(kidney_ind[0]!=0)):
            r = len(liver_ind[0])/len(kidney_ind[0])
            tmp = vol[kidney_ind[0]]
            d = tmp.shape
            kidneys = np.zeros((r*d[0],d[1],d[2],d[3]),dtype = np.float32)
            for k in range(r):
                kidneys[k*d[0]:(k+1)*d[0],:,:,:] = vol[kidney_ind[0]]
             
        #create new classification array
        if (len(kidney_ind[0]!= 0)):
            new_class = np.zeros(len(livers)+len(kidneys)+len(livers),dtype = np.int32)
            new_class[:len(livers)] = dh.liver
            new_class[len(livers):len(livers)+len(kidneys)] = dh.kidney
            new_class[len(livers)+len(kidneys):] = dh.nothing
        else:
            new_class = np.zeros(2*len(livers),dtype = np.int32)
            new_class[:len(livers)] = dh.liver
            new_class[len(livers):] = dh.nothing
     
        #create a new and smaller numpy volumes array
        if (len(kidney_ind[0]!= 0)):
            new_vol = np.zeros((len(livers)+len(kidneys)+len(livers),d[1],d[2],d[3]),dtype = np.float32)
            new_vol[:len(livers),:,:,:]= livers
            new_vol[len(livers):len(livers)+len(kidneys),:,:,:] = kidneys
            new_vol[len(livers)+len(kidneys):,:,:,:] = nothings[0:len(livers)]
        else:
            new_vol = np.zeros((2*len(livers),d[1],d[2],d[3]),dtype = np.float32)
            new_vol[:len(livers),:,:,:]= livers
            new_vol[len(livers):,:,:,:] = nothings[0:len(livers)]
        
        #overwrite original files
        np.save(vol_src_path + "\\" + f,new_vol)
        np.save(seg_src_path + "\\" + class_f, new_class)
    
###############################################################################
#run code
###############################################################################

#to be run after prep_data
vol_src_path = "Train\\Volumes"
seg_src_path = "Train\\Class"
validation_files_ind = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

norm_data(vol_src_path, seg_src_path, validation_files_ind)
print "done."










