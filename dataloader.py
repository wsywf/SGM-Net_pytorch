import cv2
import random
import numpy as np
import torch
from readpfm import readPFM

class KittiDataset():
    def __init__(self, train_file, mode='train', n_samples=None):
        self.mode = mode
        self.train_file = open(train_file,'r')
        self.image_lists = []
        for line in self.train_file:
            line = line.strip('\n')
            self.image_lists.append(line)
        self.data_root = 'data/kitti'
        #left image        
        self.left_image = cv2.imread('data/driving/left/0350.png',0)
        self.left_image = self.left_image.astype(np.float32)
        self.max_pixel = np.max(self.left_image)
        #use disp image .pfm          
        self.disp_image, _ = readPFM('data/driving/disp/0350.pfm')
        self.disp_image = self.disp_image.astype(np.int32)
#        #use disp image .png          
#        disp_path = self.data_root + '/disp_occ_0/' + image_name + '.png'
#        disp_image = cv2.imread(disp_path,0)
        
        # use the ct_cost from calculate_ct_cost in local file
        self.d_cost = np.load('data/driving/disp/0350.npy')
        self.d_cost = np.array(self.d_cost)
        self.d_cost = torch.tensor(self.d_cost)
        self.d_cost = self.d_cost.type(torch.FloatTensor)
        
        #use the cost volume geted from mc-cnn https://github.com/jzbontar/mc-cnn        
#        cost_path = self.data_root + '/leftbin/' + image_name + '.bin'
#        d_cost = np.memmap(cost_path, dtype=np.float32, shape=(1, 70, img_h, img_w))
#        p = np.isnan(d_cost)
#        d_cost[p] = 0.0  
#        d_cost = torch.tensor(d_cost)
#        d_cost = d_cost.squeeze(0)
#        d_cost = d_cost.type(torch.FloatTensor)  # [70,h,w]
        
        
        self.img_h , self.img_w = self.left_image.shape
        self.valid_points = []
        #get the valid points list
        for i in range(300, self.img_w-200):
            for j in range(150, self.img_h -150):
                disp_val = self.disp_image[j,i]
                if(disp_val>5 and disp_val < 69):
                    self.valid_points.append([i,j,disp_val])
                   
        random.shuffle(self.valid_points)
        self.valid_points = random.sample(self.valid_points, int(len(self.valid_points)/20))
        
        print(self.d_cost.size())
        print(self.img_h,self.img_w)

    def getlen(self):
        return len(self.valid_points)

    def getitem(self, i):
        
        train_point = self.valid_points[i]
        x = train_point[0]
        y = train_point[1]
        disp_val = train_point[2]

        image_patchs_l = []
        patch_cods_l = []
        
        image_patchs_r = []
        patch_cods_r = []
        
        image_patchs_u = []
        patch_cods_u = []
        
        image_patchs_d = []
        patch_cods_d = []
        
        for i in range(120,x):#left
            x0 = i-3 
            x1 = i+4
            y0 = y-3
            y1 = y+4
            img_patch = self.left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/self.max_pixel
            
            image_patchs_l.append(img_patch_normlz)
            patch_cods_l.append([i,y])

        for i in range(x+1,self.img_w-8):#right
            x0 = i-3 
            x1 = i+4
            y0 = y-3
            y1 = y+4
            img_patch = self.left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/self.max_pixel
            
            image_patchs_r.append(img_patch_normlz)
            patch_cods_r.append([i,y])
            
        for j in range(8,y):#up
            x0 = x-3
            x1 = x+4
            y0 = j-3
            y1 = j+4
            img_patch = self.left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/self.max_pixel
            
            image_patchs_u.append(img_patch_normlz)
            patch_cods_u.append([x,j])
            
        for j in range(y+1,self.img_h-8):#down
            x0 = x-3
            x1 = x+4
            y0 = j-3
            y1 = j+4
            img_patch = self.left_image[y0:y1, x0:x1]
            patch_mean = np.mean(img_patch)
            img_patch_normlz = (img_patch - patch_mean)/self.max_pixel
            
            image_patchs_d.append(img_patch_normlz)
            patch_cods_d.append([x,j])

        origin_patch = []
        origin_cod = []
        x0 = x-3
        x1 = x+4
        y0 = y-3
        y1 = y+4
        img_patch = self.left_image[y0:y1, x0:x1]
        patch_mean = np.mean(img_patch)
        img_patch_normlz = (img_patch - patch_mean)/self.max_pixel
        origin_patch.append(img_patch_normlz)
        origin_cod.append([x,y])

        ### left path
        image_patchs_l = np.array(image_patchs_l)
        image_patchs_l = torch.tensor(image_patchs_l)
        image_patchs_l = image_patchs_l.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_l = image_patchs_l.type(torch.FloatTensor)

        patch_cods_l = np.array(patch_cods_l)
        patch_cods_l = torch.tensor(patch_cods_l)
        patch_cods_l = patch_cods_l.type(torch.FloatTensor)
        patch_cods_l[:,0]=patch_cods_l[:,0]/self.img_w
        patch_cods_l[:,1]=patch_cods_l[:,1]/self.img_h

        ### right path
        image_patchs_r = np.array(image_patchs_r)
        image_patchs_r = torch.tensor(image_patchs_r)
        image_patchs_r = image_patchs_r.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_r = image_patchs_r.type(torch.FloatTensor)

        patch_cods_r = np.array(patch_cods_r)
        patch_cods_r = torch.tensor(patch_cods_r)
        patch_cods_r = patch_cods_r.type(torch.FloatTensor)
        patch_cods_r[:,0]=patch_cods_r[:,0]/self.img_w
        patch_cods_r[:,1]=patch_cods_r[:,1]/self.img_h
        
        ### up path
        image_patchs_u = np.array(image_patchs_u)
        image_patchs_u = torch.tensor(image_patchs_u)
        image_patchs_u = image_patchs_u.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_u = image_patchs_u.type(torch.FloatTensor)

        patch_cods_u = np.array(patch_cods_u)
        patch_cods_u = torch.tensor(patch_cods_u)
        patch_cods_u = patch_cods_u.type(torch.FloatTensor)
        patch_cods_u[:,0]=patch_cods_u[:,0]/self.img_w
        patch_cods_u[:,1]=patch_cods_u[:,1]/self.img_h
        
        ### down path
        image_patchs_d = np.array(image_patchs_d)
        image_patchs_d = torch.tensor(image_patchs_d)
        image_patchs_d = image_patchs_d.unsqueeze(1)  ## [n,1,5,5]
        image_patchs_d = image_patchs_d.type(torch.FloatTensor)

        patch_cods_d = np.array(patch_cods_d)
        patch_cods_d = torch.tensor(patch_cods_d)
        patch_cods_d = patch_cods_d.type(torch.FloatTensor)
        patch_cods_d[:,0]=patch_cods_d[:,0]/self.img_w
        patch_cods_d[:,1]=patch_cods_d[:,1]/self.img_h

        ### x0,y0 
        origin_patch = np.array(origin_patch)
        origin_patch = torch.tensor(origin_patch)
        origin_patch = origin_patch.unsqueeze(1)  ## [n,1,5,5]
        origin_patch = origin_patch.type(torch.FloatTensor)
        
        origin_cod = np.array(origin_cod)
        origin_cod = torch.tensor(origin_cod)
        origin_cod = origin_cod.type(torch.FloatTensor)
        origin_cod[:,0]=origin_cod[:,0]/self.img_w
        origin_cod[:,1]=origin_cod[:,1]/self.img_h

        ### match cost
        d_cost_l = self.d_cost[:,y-8,112:x-7]
        d_cost_r = self.d_cost[:,y-8,x-8:self.img_w-16]
        d_cost_u = self.d_cost[:,0:y-7,x-8]
        d_cost_d = self.d_cost[:,y-8:self.img_h-16,x-8]
        d_cost_x0 = self.d_cost[:,y-8,x-8]

        return image_patchs_l, patch_cods_l, image_patchs_r, patch_cods_r, image_patchs_u, patch_cods_u, image_patchs_d, patch_cods_d, origin_patch, origin_cod ,disp_val, d_cost_l , d_cost_r, d_cost_u, d_cost_d,d_cost_x0, x, y

