from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

#Helper functions for CT data manipulation

#classification index value of nothing, liver and kidney
nothing = 0
liver = 1
kidney = 2

###############################################################################
###############################################################################

#function receives: vol_fn - volume file name
#                   class_list - list of classification files
#        returns: file name of classification file which classifies vol_fn file      
def ret_class_file(vol_fn, class_list):
    #start of each 
    vol_header = "Volume_patchsize_"
    class_header = "Classification_patchsize_"
    
    #copy class list
    tmp_class_list = list(class_list)
    
    #remove header from file name
    tmp_vol_fn = vol_fn[len(vol_header):len(vol_fn)]
    
    #remove header from each file name in class list
    k=0
    for f in class_list:
        tmp_class_list[k] = f[len(class_header):len(f)]
        k+=1
        
    #search for tmp_vol_fn in tmp_class_list and get index
    ind = tmp_class_list.index(tmp_vol_fn)
    return class_list[ind]


###############################################################################
###############################################################################

#function recieves: input_im - numpy array size of [h,x,x] whilst x % 2 ==0 which contains a CT volume
#                   r - by how much the image should be reduced. r  == 2^n
#                   start_im - index of sub_image in input_im
#                   end_im - index of sub_image in input_im
#function returns: a numpy array ret which includes images reduced by r, from index
#                   start_im to index end_im
def Vol_size_reduce(input_im,r,start_im,end_im):
    
    d = input_im.shape
    res_list = list()
        
    for i in range(start_im,end_im):
        im = Image.fromarray(input_im[i,:,:])
        im = im.resize((d[1]/r,d[2]/r),Image.BICUBIC) #reduce image size
        tmp = np.array(im) #convert back to numpy array
        res_list.append(tmp)
        
    return np.array(res_list)
    
###############################################################################
###############################################################################
    
#function recieves: input_im - numpy array size of [h,x,x] whilst x % 2 ==0 which contains a CT Segmentation Image
#                   r - by how much the image should be reduced. r  == 2^n
#                   start_im - index of sub_image in input_im
#                   end_im - index of sub_image in input_im
#function returns: a numpy array ret which includes images reduced by r, from index
#                   start_im to index end_im   
def Seg_size_reduce(input_im,r, start_im, end_im):
    
    d = input_im.shape
    res_list = list()
        
    for i in range(start_im,end_im):
        im = Image.fromarray(input_im[i,:,:])
        im = im.resize((d[1]/r,d[2]/r),Image.NEAREST) #reduce image size
        tmp = np.array(im) #convert back to numpy array
        res_list.append(tmp)
        
    return np.array(res_list)
    
###############################################################################
###############################################################################
    
#function recieves: Im  - numpy array of shape [num_of_slices,512,512]
#                   BlkSz - size of patch to divide the image to
#                   x_shift - shift each image in Im by x_pixels on x axis
#                   y_shift - shift each image in Im by y_pixels on y axis
#                   is_vol - if is_vol = True function handles CT volume image, if False segmentation Image
#         returns: numpy array of shape [(512/BlkSz)^2*num_of_slices,1, BlkSz, BlkSz]
#                  such that each patch is a BlkSz x BlkSz from an a square area in a slice
def Im2Blks(Im,BlkSz,x_shift = 0,y_shift = 0, is_vol = True):
    #assuming image size is symmetric nxn and BlkSz < 512 and BlkSz % 2 ==0
    
    #check dimension of Image
    d = Im.shape   
    if (len(d)==2): #if 2d image
        L = [1,d[0], d[1]] #if 2d image
    else:
        L = d #if 3d image

    #shift Im in x axis and y axis by x_shift and y_shift pixels
    if (is_vol):
        shiftedIm = -1000*np.ones([L[0],L[1],L[2]],dtype = np.float32)
    else:
        shiftedIm = np.zeros([L[0],L[1],L[2]],dtype = np.float32)
    
    if (len(d) == 2):
        shiftedIm[:,:L[1]-x_shift,:L[2]-y_shift] = Im[x_shift:L[1],y_shift:L[2]]
    else:
        shiftedIm[:,:L[1]-x_shift,:L[2]-y_shift] = Im[:,x_shift:L[1],y_shift:L[2]]
    
    #enlarge image with negative -1000 values such that Im.size%BlkSz == 0
    if (L[1] % BlkSz != 0 ):
        row_col_add = BlkSz - L[1] % BlkSz
        if (is_vol):
            tmpIm = -1000*np.ones([L[0],L[1]+row_col_add,L[2]+row_col_add],dtype=np.float32)
        else:
            tmpIm = np.zeros([L[0],L[1]+row_col_add,L[2]+row_col_add],dtype=np.float32)
        L = tmpIm.shape
        tmpIm[:,row_col_add/2:L[1]-row_col_add/2,row_col_add/2:L[2]-row_col_add/2] = shiftedIm
    else:
        tmpIm = shiftedIm
             
    #reshape Image to a collection of patches   
    shape = [L[0],L[1]/BlkSz,L[2]/BlkSz,BlkSz,BlkSz]
    strides = [4*L[1]*L[2],4*BlkSz*L[2],4*BlkSz,4*L[2],4]
    
    Blk = np.lib.stride_tricks.as_strided(tmpIm ,shape=shape,strides=strides)
    Blk = np.reshape(Blk,[L[0]*L[1]*L[2]/BlkSz**2,1,BlkSz,BlkSz])
    return Blk