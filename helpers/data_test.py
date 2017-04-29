import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import random
import math
import matplotlib.pyplot as plt

vol_src_path = 'C:/CT/mocks/volumes/'
seg_src_path = 'C:/CT/mocks/segmentations/'
train_src_path = 'C:/CT/mocks/Train/'

vol_list = os.listdir(vol_src_path)
class_list = os.listdir(seg_src_path)

# vol = Image.open(vol_src_path + vol_list[0]).convert('LA')
# seg = Image.open(seg_src_path + class_list[0]).convert('LA')
# image = np.fromfile(vol_src_path + vol_list[0])
# vol = img.imread(vol_src_path + vol_list[0])
# seg = img.imread(seg_src_path + class_list[0])

# black and white
# vol_black = np.zeros([14,14], dtype='uint8')
# seg_black = np.zeros([14,14], dtype='uint8')
# vol_white = np.ones([14,14], dtype='uint8')
# seg_white = np.ones([14,14], dtype='uint8')

# vol_black = np.array(range(196)).reshape(14,14)
# seg_black = np.array(range(196)).reshape(14,14)
# vol_white = np.array(range(196)).reshape(14,14)
# seg_white = np.array(range(196)).reshape(14,14)

# vol_white.fill(255)

# plt.imshow(vol)
# plt.show()

# np.save(train_src_path + 'volumes/vol' + str(0), vol_black)
# np.save(train_src_path + 'segmentations/seg' + str(0), seg_black)
# np.save(train_src_path + 'volumes/vol' + str(1), vol_white)
# np.save(train_src_path + 'segmentations/seg' + str(1), vol_white)

vol = np.asarray(Image.open(vol_src_path + vol_list[0]).convert('L'))
seg = np.asarray(Image.open(seg_src_path + class_list[0]).convert('L'))

np.save(train_src_path + 'volumes/vol' + str(0), vol)
np.save(train_src_path + 'segmentations/seg' + str(0), seg)

# number_of_colors = 100

# for i in range(number_of_colors):
#
#     vol_grey = np.zeros((vol.shape[0], vol.shape[1]))  # init 2D numpy array
#     seg_grey = np.zeros((vol.shape[0], vol.shape[1]))  # init 2D numpy array
#
#     rand_int = math.floor(random.random() * 255)
#
#     for row in range(len(vol)):
#         for col in range(len(vol[row])):
#             vol_grey[row][col] = rand_int
#
#     for row in range(len(seg)):
#         for col in range(len(seg[row])):
#             seg_grey[row][col] = rand_int
#
#     np.save(train_src_path + 'volumes/vol' + str(i), vol_grey)
#     np.save(train_src_path + 'segmentations/seg' + str(i), seg_grey)
