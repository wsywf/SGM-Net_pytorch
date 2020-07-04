import cv2
import random
import numpy as np
import torch

def census_trans(img,win_w,win_h):
    img_h,img_w = img.shape
    img_trans = []
    
    for y in range(win_h,img_h-win_h):
        for x in range(win_w,img_w-win_w):
            count = 0

            for m in range(y-win_h,y+win_h+1):
                for n in range(x-win_w,x+win_w+1):
                    count<<=1
                    if img[m][n] > img[y][x]:
                        count = count | 1
                    else:
                        count = count | 0
            #print("count",count,type(count))
            img_trans.append(count)
            #print("changdu",len(img_trans))

    return img_trans
                    


def hamin_calcu(ct_l,ct_r,Dmax,height,width):
    
    ct_cost = np.empty((Dmax,height,width))
    for y in range(0,height):
        for x in range (0,width):
            base_ct = ct_l[y*width+x]
            for d in range (0,Dmax):
                if x >= d:
                    contrast_ct = ct_r[y*width+x-d]

                    hamin_dis = bin(base_ct^contrast_ct).count('1')

                else:
                    hamin_dis = 81
                ct_cost[d,y,x] = hamin_dis
    return ct_cost
                    
            
    
