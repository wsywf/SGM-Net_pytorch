import torch
import numpy as np
from SGMNet import SGM_NET
import cv2
import torch.optim
import random
import ctypes
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SGM_NET()
model.load_state_dict(torch.load('work_sapce/SGMNet_fin.pkl'))
model = model.to(device)
model.eval()

data_root = 'datasets/training'

for image_number in range(100)
    image_name = '0000'+ image_number +'_10'
    left_path = data_root + '/image_2/' + image_name + '.png'
#    left_image = cv2.imread('/home/wsy/datasets/training/image_2/000019_10.png',0)
    left_image = cv2.imread('left_path,0)
    max_pixel = np.max(left_image)
    image_patchs = []
    patch_cods = []
    img_h,img_w = left_image.shape
    for y in range(8,img_h-8):
        for x in range(8,img_w-8):
            x0 = x-3 
            x1 = x+4
            y0 = y-3
            y1 = y+4
            img_patch = left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/max_pixel
            image_patchs.append(img_patch_normlz)
            patch_cods.append([x,y])        
    image_patchs = np.array(image_patchs)
    image_patchs = torch.tensor(image_patchs)
    image_patchs = image_patchs.unsqueeze(1)  ## [n,1,5,5]
    image_patchs = image_patchs.type(torch.FloatTensor)

    patch_cods = np.array(patch_cods)
    patch_cods = torch.tensor(patch_cods)
    patch_cods = patch_cods.type(torch.FloatTensor)
    patch_cods[:,0]=patch_cods[:,0]/img_w
    patch_cods[:,1]=patch_cods[:,1]/img_h


    image_patchs = image_patchs.to(device)
    patch_cods = patch_cods.to(device)
    p1p2_volume = model(image_patchs , patch_cods)
    
    
    save_path = 'restut/'+str(image_number)+'.npy
    np.save(save_path,p1p2_volume)
#save as c_type    
    point_n = p1p2_volume[:,0].size()[0]
    parameter_n = p1p2_volume[0,:].size()[0]
    quantity = parameter_n*point_n
    p1p2_volume=p1p2_volume.view(quantity) 
    lib = ctypes.cdll.LoadLibrary("libsave.so")
    p_volume = (ctypes.c_float *len(p1p2_volume) )(*p1p2_volume)
    lib.show_matrix(p_volume,len(p1p2_volume),image_number)


