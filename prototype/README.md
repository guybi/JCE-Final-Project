# README #

### What is this repository for? ###

Run training and validation for kindney and liver recognition on CT images 
Version V0

### How do I get set up? ###

Currently set to run on ITK file format. Volume and segmentation files should be divided into seperate folders, and should 
have matching files name such as '10000100_1_CTce_ThAb.nii.gz' and '10000100_1_CTce_ThAb_58_6.nii.gz' etc.

Order to run files in:
prep_data
prep_valid_train
train_cnn

later on will add a single run.py file

###To do next###

###Implementation:
1. add a stop rule for training. Stop if training loss is saturated.
2. take into account patches which are non uniformly segemented into an organ.
3. build test file showing a graphical representation of network results over a CT slice
4. next network to implement - two conv, two max pooling and single fully connected.

###To test/run next:
for simple_cnn_16:
1. currently using only anatomy 3 set. Maybe use silver corpus to add more kidney patches for training.
2. change to lower size patch size.