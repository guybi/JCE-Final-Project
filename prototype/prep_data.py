import numpy as np
import SimpleITK as stk
import os
import dataHelper as dh
import random


#function recieves: path, file_name and organ index
#         returns: first file that begins with str 'file_name_index' in path
def getSegFileName(path, file_name,index):
    seg_list = os.listdir(path)
    sub_name = file_name[0:len(file_name)-7] + "_" + str(index)
    for f in seg_list:
        if (f.startswith(sub_name)):
            return f
    return "Fail"

#function receives:
#       vol_path - path of input CT volume files
#       seg_path - path of input CT segmentation files
#       vol_dest_path - destination path for prepared volume files
#       seg_dest_path - destination path for segementation vector - 0 for nothing, 1 for liver and 2 for liver
#       klr - allowed kidney liver ratio. In percentage.
#       dxy - how many random shifts on both axis. Picture is shifted such that a decent number of kidney patches
#       is found. max number of tries to find good ratio of kidneys to liver = 20.
#       can be achieved, meaning #kidnyes/#livers >= klr
#       ind_start, ind_end - start and end of image slice to take into account
#       patch_size - Image is devided into patches of patch_sizexpatch_size
#       downsample - if True the Image is downsampled by two on both axes, else resolution stays the same
#
#       function saves to disk a series of numpy arrays of size (num_of_patches) x 1 x patch_size x patch_size
#                patches that are uniformly lower than -800 hounsfeld ,higher than 1000 hounsfeld are not or are not
#                uniformly segemented are not returned
def prepare_data(vol_path, seg_path, vol_dest_path, seg_dest_path, klr,SegRatio,
                 dxy = 0,ind_start = 0, ind_end = 0, patch_size = 16, downsample = False):
    #liver and kidneys index
    liverId = 58
    kindeyLId = 29662
    kidneyRId = 29663
    
    #read all files in volume directory
    #read all files in volume directory
    vol_list = os.listdir(vol_path)
    print "List of input files:"
    print vol_list
    
    #check if dest path exsits if not create new path
    if not(os.path.exists(vol_dest_path)):
        os.makedirs(vol_dest_path)
    
    if not(os.path.exists(seg_dest_path)):
        os.makedirs(seg_dest_path)
    
    #iterate over all files in volume directory read volume and segmentation for each file
    k=0
    for f in vol_list:
        print "iter: " + str(k)
        print "current file: " + f
        
        #check if to downsample
        r = 1
        if downsample:
            r = 2
          
        #read volume
        tmp = stk.ReadImage(vol_path + "\\" + f)
        input_vol = stk.GetArrayFromImage(tmp)
        #check start_ind and end_ind
        start = ind_start
        end = ind_end
        d = input_vol.shape
        if (ind_start > d[0] or ind_end > d[0]):
            start = 0
            end = d[0]     
        input_vol = dh.Vol_size_reduce(input_vol.astype(np.float32),r,start,end) #reduce size of volume image
        
        #read liver segmentation
        tmp = stk.ReadImage(seg_path + "\\" + getSegFileName(seg_path,f,liverId)) #read liver segmentation
        liver_seg  = stk.GetArrayFromImage(tmp)
        liver_seg = dh.Seg_size_reduce(liver_seg.astype(np.float32),r,start,end) #reduce size of liver_seg image
        
        #read left kidney segmentation
        tmp = stk.ReadImage(seg_path + "\\" + getSegFileName(seg_path,f,kindeyLId)) 
        left_kidney_seg  = stk.GetArrayFromImage(tmp)
        left_kidney_seg = dh.Seg_size_reduce(left_kidney_seg.astype(np.float32),r,start,end) #reduce size of left_kid_seg image
        
        #read right kidney segmentation
        tmp = stk.ReadImage(seg_path + "\\" + getSegFileName(seg_path,f,kidneyRId)) #read left kidney seg
        right_kidney_seg  = stk.GetArrayFromImage(tmp)
        right_kidney_seg = dh.Seg_size_reduce(right_kidney_seg.astype(np.float32),r,start,end) #reduce size of right_kid_seg image
        
        #make one kidney segmentation file
        kidney_seg = np.add(right_kidney_seg,left_kidney_seg)
        
        #turn segmentation matrices into 1.0 or 0.0 values
        kidney_seg[np.where(kidney_seg > 1)] = 1
        liver_seg[np.where(liver_seg > 1)] = 1
        
        #get initial random shift number 
        g_r_count = 0 #counter for good iters where #kidneys/#livers >= klr
        r_count = 0 #counter for toguy number of itterations
        shifts_list = list()
        
        while( g_r_count < dxy and r_count < 20):
            f = False
            dx = random.randint(0,15)
            dy = random.randint(0,15)

            #check if dx dy was previously used
            for dxdy in shifts_list:
                if (dxdy[0] == dx and dxdy[1] == dy):
                    f = True
                    break
            
            if f: #if used before go to next itteration with advancing counters
                continue
            else: #if not used

                #create shifted images
                shifted_input_vol = dh.Im2Blks(input_vol,patch_size,dx,dy)
                shifted_liver_seg = dh.Im2Blks(liver_seg,patch_size,dx,dy)
                shifted_kidney_seg = dh.Im2Blks(kidney_seg,patch_size,dx,dy)

                #remove irrelevant patches
                #if all values in patch are less than 10 hounsfeld  -> remove patch
                #if all values in ptach above 400 hounsfeld -> remove patch
                #if patch is not uniformly segmented  -> remove patch

                s = shifted_input_vol.shape #size changed because array was reshaped
                ind_del = list()   #indexes of patches to delete
                res_array = list() # result array nothing = 0, liver = 1, kidney = 2

                for i in range(s[0]):
                    if ((( shifted_input_vol[i,:,:] < 10).all()) or ((shifted_input_vol[i,:,:] > 400).all()) or \
                    ( np.sum(shifted_liver_seg[i,:,:]) > 0 and np.sum(shifted_liver_seg[i,:,:]) < SegRatio * patch_size ** 2) or \
                    (np.sum(shifted_kidney_seg[i,:,:]) > 0 and  np.sum(shifted_kidney_seg[i,:,:]) < SegRatio * patch_size ** 2)):
                        ind_del.append(i)

                    #build result_seg_array
                    if (np.sum(shifted_liver_seg[i,0,:,:]) >= SegRatio * patch_size ** 2):
                        res_array.append(dh.liver)
                    elif (np.sum(shifted_kidney_seg[i,0,:,:]) >= SegRatio * patch_size ** 2):
                        res_array.append(dh.kidney)
                    else:
                        res_array.append(dh.nothing)

                #delete irrelevant patches
                shifted_input_vol = np.delete(shifted_input_vol,ind_del,0)
                y_res = np.delete(np.array(res_array),ind_del,0)

                #calc and print stats for each volume
                kidney_count = len(y_res[np.where(y_res == 2)])
                liver_count = len(y_res[np.where(y_res == 1)])
                nothing_count = len(y_res)-kidney_count-liver_count

                #check if kidney to liver ration better than klr
                if kidney_count/float(liver_count) < klr/float(100):
                    r_count+=1
                else:
                    #save numpy input_vol array to disk
                    #save numpy y_res vector to disk
                    print "shifts are: dx = " + str (dx) + " dy = " + str(dy)
                    print str(kidney_count) + " kidney patches, " + str(liver_count) + " liver patches and " + str(nothing_count) + \
                    "the rest"
                    np.save(vol_dest_path + "\\Volume_patchsize_" + str(patch_size) + "_" + str(k) +"_xshift" + str(dx) + "_yshift" + str(dy), shifted_input_vol)
                    np.save(seg_dest_path +"\\Classification_patchsize_" + str(patch_size) + "_" + str(k)+"_xshift" + str(dx) + "_yshift" + str(dy), y_res)
                    r_count+=1
                    g_r_count+=1
                
            shifts_list.append([dx, dy]) #mark used rabndom shifts pair
            
        print "saved " + str(g_r_count) + " random shifted files out of " + str(dxy)           
        k+=1 
    
###############################################################################
#run code
###############################################################################

#params

vol_src_path = "c:\\CT\\Volumes"
seg_src_path = "c:\\CT\\Segmentations"
vol_dest_path = "Train\\Volumes"
seg_dest_path = "Train\\Class"
klr = 3 #in percentage
dxy = 4
start = 300
end = 600
patch_size = 14
downsample = True
SegRatio = 0.75

prepare_data(vol_src_path, seg_src_path, vol_dest_path, seg_dest_path, klr, SegRatio, dxy = dxy ,ind_start = start,
             ind_end = end , patch_size = patch_size, downsample = downsample)

print "done!"

    
        
        
    
    
    
    