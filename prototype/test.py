import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as stk
from PIL import Image
import theano
import theano.tensor as T
import lasagne
from networkMaker import build_triple_cnn14_big_fc
import dataHelper as dh


##############################################################################
#
# helper functions
#
##############################################################################


#function recieves: I - a numpy 3 dimensional array of CT slices
#                   slice_ind - index of slice to be returned
#                   sub_sample - if True the image returned shall be redcued in size by 2
#function returns: a numpy array which includes slice_ind slice from I

def get_image_slice(I,slice_ind,sub_sample):
    im = Image.fromarray(I[slice_ind,:,:])
    if (sub_sample):
        im = im.resize((256,256),Image.BICUBIC)
    return np.array(im,dtype = np.float32)


###############################################################################
#
# main code
#
###############################################################################

training_loss =  np.load("results\\train_loss_function_itr80.np.npy")
validation_loss = np.load("results\\validation_loss_function_itr80.np.npy")
validation_acc = np.load("results\\validation_accuracy_itr80.np.npy")


#show the training loss function, validation loss function, and accuracy
plt.figure(1)
plt.plot(training_loss,'r', label = "Training Loss")
plt.plot(validation_loss,'b', label = "Validation loss")
plt.plot(1-validation_acc/100,'g', label = "1 - validation accuracy")
legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large')
plt.ylabel("Value")
plt.xlabel("Epoch number")

plt.figure(2)
plt.plot(validation_acc)
plt.ylabel("Validation accuracy")
plt.xlabel("Epoch number")

#show a selected CT slice from a selected file and classification of kidney, liver, other
#using convolutional network
patch_size = 14 #size of patch to be examined
dx = 3 #shift of patch in the image x axis
dy = 3 #shift of patch in the image y axis
sub_sampled = True #should the image be reduced in size by 2
slice_index = 450 #index of slice to be examined
SegRatio = 0.75 #Segmentaion Ratio Threshold
CT_file_path = "CT\\Volumes\\10000006_1_CT_wb.nii.gz" #CT volume path
LiverSeg_file_path = "CT\\Segmentations\\10000006_1_CT_wb_58_8.nii.gz"
Kidney1Seg_file_path = "CT\\Segmentations\\10000006_1_CT_wb_29662_8.nii.gz"
Kidney2Seg_file_path = "CT\\Segmentations\\10000006_1_CT_wb_29663_8.nii.gz"
network_params_path = "results\\model_itr80.npz" #neural network params path

#read image from path
tmp = stk.ReadImage(CT_file_path)
CT_I = stk.GetArrayFromImage(tmp)
tmp = stk.ReadImage(LiverSeg_file_path)
LiverSegI = stk.GetArrayFromImage(tmp)
tmp = stk.ReadImage(Kidney1Seg_file_path)
Kidney1SegI = stk.GetArrayFromImage(tmp)
tmp = stk.ReadImage(Kidney2Seg_file_path)
Kidney2SegI = stk.GetArrayFromImage(tmp)

#get slice from image
I = get_image_slice(CT_I,slice_index, sub_sampled)
LiverSegI = get_image_slice(LiverSegI,slice_index, sub_sampled)
Kidney1SegI = get_image_slice(Kidney1SegI,slice_index, sub_sampled)
Kidney2SegI = get_image_slice(Kidney2SegI,slice_index, sub_sampled)

KidneySegI = Kidney1SegI +  Kidney2SegI
KidneySegI[np.where(KidneySegI > 1)] = 1
LiverSegI[np.where(LiverSegI > 1)] = 1

#build a numpy array from the image without a classification array
vol_patches = dh.Im2Blks(I,patch_size,dx,dy,True)

# Prepare Theano variables for inputs and targets
input_var = T.tensor4('inputs')

#build a neural network and lasagne functions
network = build_triple_cnn14_big_fc(input_var)
test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_prediction_max = T.argmax(test_prediction, axis=1)
cnn_res = theano.function([input_var], [test_prediction_max])

#load neural network params
with np.load(network_params_path) as f:
    param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

#do a forward pass for the selected slice
res = cnn_res(vol_patches)

#show the selected image
plt.figure(3)
plt.imshow(I,cmap='Greys_r')
plt.title("Original Image")

#create a max val patch
max_val = np.ones((patch_size,patch_size), dtype = np.float32)*np.amax(I)

print res[0]

#show on the slected image the liver
L =  I.shape
row_col_add = patch_size - L[0] % patch_size #number of pixels added to original image in Im2Blks
patches_num = (L[0]+row_col_add) / patch_size #how many patches in a row or column
liver_ind = np.where(res[0] == dh.liver)
x_liv = (liver_ind[0]/ patches_num)*patch_size - row_col_add/2 + dx
y_liv = (liver_ind[0] % patches_num)*patch_size - row_col_add/2 + dy


liverI = np.array(I)
for k in range(len(x_liv)):
    #check if patch is a kegal patch to be examined, meaning:
    #all of it's pixels are above 10 and less than 400 hounsfeld
#    if ~((liverI[x_liv[k]:x_liv[k]+patch_size,y_liv[k]:y_liv[k]+patch_size] < 10).all() or (liverI[x_liv[k]:x_liv[k]+patch_size,y_liv[k]:y_liv[k]+patch_size] > 400).all()):
#        liverI[x_liv[k]:x_liv[k]+patch_size,y_liv[k]:y_liv[k]+patch_size]=max_val

    if ~((liverI[x_liv[k]:x_liv[k]+patch_size,y_liv[k]:y_liv[k]+patch_size] < 10).all() or (liverI[x_liv[k]:x_liv[k]+patch_size,y_liv[k]:y_liv[k]+patch_size] > 400).all() or \
        ( ( np.sum(LiverSegI[x_liv[k]:x_liv[k]+patch_size,y_liv[k]:y_liv[k]+patch_size]) > 0 and np.sum(LiverSegI[x_liv[k]:x_liv[k]+patch_size,y_liv[k]:y_liv[k]+patch_size]) < SegRatio * patch_size ** 2) or \
        (np.sum(KidneySegI[x_liv[k]:x_liv[k]+patch_size,y_liv[k]:y_liv[k]+patch_size]) > 0 and  np.sum(KidneySegI[x_liv[k]:x_liv[k]+patch_size,y_liv[k]:y_liv[k]+patch_size]) < SegRatio * patch_size ** 2)))  :
        liverI[x_liv[k]:x_liv[k]+patch_size,y_liv[k]:y_liv[k]+patch_size]=max_val

    

         
                    


plt.figure(4)
plt.imshow(liverI,cmap='Greys_r')
plt.title("Image with marked liver area")


#show on the selected image the kidneys
kidney_ind = np.where(res[0] == dh.kidney)
x_kid = (kidney_ind[0]/ patches_num)*patch_size - row_col_add/2 + dx
y_kid = (kidney_ind[0] % patches_num)*patch_size - row_col_add/2 + dy

kidneyI = np.array(I)
for k in range(len(x_kid)):
    if ~((kidneyI[x_kid[k]:x_kid[k]+patch_size,y_kid[k]:y_kid[k]+patch_size] < 10).all() and (kidneyI[x_kid[k]:x_kid[k]+patch_size,y_kid[k]:y_kid[k]+patch_size] > 400).all()or \
        ( ( np.sum(LiverSegI[x_kid[k]:x_kid[k]+patch_size,y_kid[k]:y_kid[k]+patch_size]) > 0 and np.sum(LiverSegI[x_kid[k]:x_kid[k]+patch_size,y_kid[k]:y_kid[k]+patch_size]) < SegRatio * patch_size ** 2) or \
        (np.sum(KidneySegI[x_kid[k]:x_kid[k]+patch_size,y_kid[k]:y_kid[k]+patch_size]) > 0 and  np.sum(KidneySegI[x_kid[k]:x_kid[k]+patch_size,y_kid[k]:y_kid[k]+patch_size]) < SegRatio * patch_size ** 2)))  :
        kidneyI[x_kid[k]:x_kid[k]+patch_size,y_kid[k]:y_kid[k]+patch_size]=max_val

plt.figure(5)
plt.imshow(kidneyI,cmap='Greys_r')
plt.title("Image with marked kidney area")


plt.show()