import cv2
import random
import numpy as np
from ct_list import census_trans,hamin_calcu

image_name = '0775'
data_root = 'data/driving'
left_path = data_root + '/left/' + image_name + '.png'
left_image = cv2.imread(left_path,0)

right_path = data_root + '/right/' + image_name + '.png'
right_image = cv2.imread(right_path,0)


img_h , img_w = left_image.shape
# ct window
win_h = 8
win_w = 8

# max disp
Dmax = 70
#feature calculate
ct_left = census_trans(left_image,win_w,win_h)
ct_right = census_trans(right_image,win_w,win_h)
co_height = img_h-2*win_h
co_width = img_w-2*win_w

# use hamming distance to be the cost
d_cost =hamin_calcu(ct_left,ct_right,Dmax,co_height,co_width)

file_name = data_root + '/disp/' +image_name +'.npy'
np.save(file_name, d_cost)
