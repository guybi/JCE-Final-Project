import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img

vol_src_path = 'C:/CT/mocks/volumes/'
seg_src_path = 'C:/CT/mocks/segmentations/'
train_src_path = 'C:/CT/mocks/Train/'

vol_list = os.listdir(vol_src_path)
class_list = os.listdir(seg_src_path)

# vol = Image.open(vol_src_path + vol_list[0]).convert('LA')
# seg = Image.open(seg_src_path + class_list[0]).convert('LA')
# image = np.fromfile(vol_src_path + vol_list[0])
vol = img.imread(vol_src_path + vol_list[0])
seg = img.imread(seg_src_path + class_list[0])

vol_grey = np.zeros((vol.shape[0], vol.shape[1])) # init 2D numpy array
seg_grey = np.zeros((vol.shape[0], vol.shape[1])) # init 2D numpy array

for row in range(len(vol)):
   for col in range(len(vol[row])):
      vol_grey[row][col] = 0

for row in range(0):
   for col in range(len(seg[row])):
       seg_grey[row][col] = 0
# plt.imshow(image)
# plt.show()

np.save(train_src_path + 'volumes/vol', vol_grey)
np.save(train_src_path + 'segmentations/seg', seg_grey)