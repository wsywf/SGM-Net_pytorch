import cv2
import random
import numpy as np
import torch
from readpfm import readPFM
import sys
import os

data_root = '/home/wsy/datasets/training'
imglist = '/home/wsy/datasets/training/image_3/lists.txt'
save_root = '.'

train_file = open(imglist,'r')
image_lists = []
for line in train_file:
    line = line.strip('\n')
    image_lists.append(line)


pair_num = 0
lable = []

for image_num in range(len(image_lists)):
    image_name = image_lists[image_num]
    
    print('image',image_name,image_num)
    
    left_path = data_root + '/image_2/' + image_name + '.png'
    left_image = cv2.imread(left_path)
#    print(left_image.shape)

    right_path = data_root + '/image_3/' + image_name + '.png'
    right_image = cv2.imread(right_path)

    disp_path = data_root + '/disp_occ_0/' + image_name + '.png'
    disp_image = cv2.imread(disp_path,0)
    
    for res in range(7,27,2):
        res_path = os.path.join(save_root,'res_'+str(res))
        if not os.path.exist(res_path)
            os.mkdir(res_path)
        res_path = os.path.join(save_root,'res_'+str((res-1)*2+1)
        if not os.path.exist(res_path)
            os.mkdir(res_path)
    
    
    max_pixel_l = np.max(left_image)
    max_pixel_r = np.max(right_image)
    img_h,img_w,cha = left_image.shape
    
    vaid_points = []
       
    for i in range(200,img_w-145):
            for j in range(30, img_h-30):
                disp_val = disp_image[j,i]
                if(disp_val>0 and disp_val < 127):
                    vaid_points.append([i,j,disp_val])
                   
    random.shuffle(vaid_points)
    print(len(vaid_points))
    vaid_points = random.sample(vaid_points, 600)
    print(len(vaid_points))
    max_disp = 128
    
    for k in range(0,len(vaid_points)):
    
        point = vaid_points[k]
        x = point[0]
        y = point[1]
        disp_val = point[2]
        
        target_left_border = x - max_disp
        
        for res in range(7,27,2):
            half_window = (res-1)/2
            y0 = y  - half_window
            y1 = y  + half_window + 1
            
            left_ori = left_image[y0:y1, x - half_window:x+half_window + 1]
            right_strip = right_image[y0:y1, target_left_border-6:x+7]
            
                       
        #7*7
        y0 = y-6
        y1 = y+7
        left_ori = left_image[y0:y1, x-6:x+7]
        right_strip = right_image[y0:y1, xpos-6:xpos+7]
        
        #15*15        
        y0 = y-6
        y1 = y+7
        left_ori = left_image[y0:y1, x-6:x+7]
        right_strip = right_image[y0:y1, xpos-6:xpos+7]
        
        #9*9
        y0 = y-6
        y1 = y+7
        left_ori = left_image[y0:y1, x-6:x+7]
        right_strip = right_image[y0:y1, xpos-6:xpos+7]
        
        #17*17        
        y0 = y-6
        y1 = y+7
        left_ori = left_image[y0:y1, x-6:x+7]
        right_strip = right_image[y0:y1, xpos-6:xpos+7]
        
        #11*11
        y0 = y-6
        y1 = y+7
        left_ori = left_image[y0:y1, x-6:x+7]
        right_strip = right_image[y0:y1, xpos-6:xpos+7]

        #23*23
        y0 = y-13
        y1 = y+14
        left_downsam = left_image[y0:y1, x-13:x+14]
        right_downsam_strip = right_image[y0:y1, xpos-13:xpos+14]
        
        #13*13
        y0 = y-12
        y1 = y+13
        left_downsam = left_image[y0:y1, x-13:x+14]
        right_downsam_strip = right_image[y0:y1, xpos-13:xpos+14]

        #25*25
        y0 = y-12
        y1 = y+13
        left_downsam = left_image[y0:y1, x-13:x+14]
        right_downsam_strip = right_image[y0:y1, xpos-13:xpos+14]


        #先正样本
        cv2.imwrite('./left_origin/'+str(pair_num)+'left_ori.png',left_ori)
        cv2.imwrite('./right_origin/'+str(pair_num)+'right_origin.png',right_ori_pos)
        cv2.imwrite('./left_downsample/'+str(pair_num)+'left_downsample.png',left_downsam)
        cv2.imwrite('./right_downsample/'+str(pair_num)+'right_downsample.png',right_downsam_pos)
        lable.append(int(1))
#        print(pair_num)
        pair_num = pair_num + 1
        #再负样本
        cv2.imwrite('./left_origin/'+str(pair_num)+'left_ori.png',left_ori)
        cv2.imwrite('./right_origin/'+str(pair_num)+'right_origin.png',right_ori_neg)
        cv2.imwrite('./left_downsample/'+str(pair_num)+'left_downsample.png',left_downsam)
        cv2.imwrite('./right_downsample/'+str(pair_num)+'right_downsample.png',right_downsam_neg)
        lable.append(int(0))
#        print(pair_num)
        pair_num = pair_num + 1
    print(pair_num)











#scenenflow 数据集       
#sc_data_root = '/home/wsy/datasets/sceneflow_driving/frames_cleanpass/35mm_focallength/scene_forwards/fast'
#sc_imglist = '/home/wsy/datasets/sceneflow_driving/frames_cleanpass/35mm_focallength/scene_forwards/fast/left/lists.txt'        


#sc_train_file = open(sc_imglist,'r')
#sc_image_lists = []
#for line in sc_train_file:
#    line = line.strip('\n')
#    sc_image_lists.append(line)






#        
#for image_num in range(len(sc_image_lists)):
#    image_name = sc_image_lists[image_num]
#    
#    print('image',image_name,image_num)
#    
#    left_path = sc_data_root + '/left/' + image_name + '.png'
#    left_image = cv2.imread(left_path)

#    right_path = sc_data_root + '/right/' + image_name + '.png'
#    right_image = cv2.imread(right_path)

#    disp_path = '/home/wsy/datasets/sceneflow_driving/disparity/35mm_focallength/scene_forwards/fast/left/' + image_name + '.pfm'
#    disp_image, _ = readPFM(disp_path)
#    disp_image = disp_image.astype(np.int32)
#    
#    max_pixel_l = np.max(left_image)
#    max_pixel_r = np.max(right_image)
#    img_h,img_w,cha = left_image.shape
#    
#    vaid_points1 = []
#       
#    for i in range(200,img_w-145):
#            for j in range(15, img_h-15):
#                disp_val = disp_image[j,i]
#                if(disp_val>0 and disp_val < 5):
#                    vaid_points1.append([i,j,disp_val])
#                   
#    random.shuffle(vaid_points1)
#    print(len(vaid_points1))
#    if len(vaid_points1) >= 300:
#        vaid_points1 = random.sample(vaid_points1, 300)
#    print(len(vaid_points1))
#    points2_num = 400-len(vaid_points1)
#    
#    vaid_points2 = []
#    for i in range(200,img_w-145):
#        for j in range(15, img_h-15):
#            disp_val = disp_image[j,i]
#            if(disp_val>0 and disp_val < 70):
#                vaid_points2.append([i,j,disp_val])
#                         
#    random.shuffle(vaid_points2)
#    print(len(vaid_points2))
#    vaid_points2 = random.sample(vaid_points2, points2_num)
#    print(len(vaid_points2))
#    
#    vaid_points = vaid_points1 + vaid_points2
#    print(len(vaid_points))
#    
#    
#    for k in range(0,len(vaid_points)):
#    
#        point = vaid_points[k]
#        x = point[0]
#        y = point[1]
#        disp_val = point[2]
#        pos =[n for n in range(0,1)]
#        opos = random.sample(pos,1)[0]
#        xpos = x-disp_val +opos
#        
#        
#        neg =[n for n in range(1,5)] + [n for n in range(-4,0)]
#        oneg = random.sample(neg,1)[0]
#        xneg = x-disp_val + oneg
#        #13*13
#        y0 = y-6
#        y1 = y+7
#        left_ori = left_image[y0:y1, x-6:x+7]
#        right_ori_pos = right_image[y0:y1, xpos-6:xpos+7]
#        right_ori_neg = right_image[y0:y1, xneg-6:xneg+7]
#        #27*27
#        y0 = y-13
#        y1 = y+14
#        left_downsam = left_image[y0:y1, x-13:x+14]
#        right_downsam_pos = right_image[y0:y1, xpos-13:xpos+14]
#        right_downsam_neg = right_image[y0:y1, xneg-13:xneg+14]
#        
#        #先正样本
#        cv2.imwrite('./left_origin/'+str(pair_num)+'left_ori.png',left_ori)
#        cv2.imwrite('./right_origin/'+str(pair_num)+'right_origin.png',right_ori_pos)
#        cv2.imwrite('./left_downsample/'+str(pair_num)+'left_downsample.png',left_downsam)
#        cv2.imwrite('./right_downsample/'+str(pair_num)+'right_downsample.png',right_downsam_pos)
#        lable.append(int(1))
##        print(pair_num)
#        pair_num = pair_num + 1
#        #再负样本
#        cv2.imwrite('./left_origin/'+str(pair_num)+'left_ori.png',left_ori)
#        cv2.imwrite('./right_origin/'+str(pair_num)+'right_origin.png',right_ori_neg)
#        cv2.imwrite('./left_downsample/'+str(pair_num)+'left_downsample.png',left_downsam)
#        cv2.imwrite('./right_downsample/'+str(pair_num)+'right_downsample.png',right_downsam_neg)
#        lable.append(int(0))
##        print(pair_num)
#        pair_num = pair_num + 1        
#    print(pair_num)
#lable = np.array(lable)
#np.save('./lable/lable.npy',lable)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
