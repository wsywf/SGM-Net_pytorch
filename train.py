import torch
import torch.optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from dataloader import KittiDataset
from SGMNet import SGM_NET
from loss.sgm_pathloss import pathloss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter("work_sapce/")

model = SGM_NET()
#model.load_state_dict(torch.load('work_sapce/SGMNet.pkl'))
model = model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

train_loader = KittiDataset('Datasets/kitti/train2.txt')
train_len = train_loader.getlen()

print('Train size : ', train_len)

batch_size = 5

for epoch in range(5):
    exp_lr_scheduler.step()
    trained_num = 0
    for iter_num in range(int(train_len/batch_size-1)):
        if(epoch*iter_num + iter_num == 300):
            exp_lr_scheduler.step()
            
        patch_x0 = []
        cod_x0 = []
        grad_all = []
        
        loss_sum = 0
        cout_valid = 0
        for batch_num in range(batch_size):
            image_patchs_l, patch_cods_l, image_patchs_r, patch_cods_r, image_patchs_u, patch_cods_u, image_patchs_d, patch_cods_d, origin_patch, origin_cod ,disp_val, d_cost_l , d_cost_r, d_cost_u, d_cost_d, d_cost_x0 ,x,y= train_loader.getitem(trained_num + batch_num)

#            print(d_cost_l.size())
#            print(d_cost_r.size())
#            print(d_cost_u.size())
#            print(d_cost_d.size())
#            print(d_cost_x0.size())
#            print(image_patchs_l.size(), patch_cods_l.size())
#            print(image_patchs_r.size(), patch_cods_r.size())
#            print(image_patchs_u.size(), patch_cods_u.size())
#            print(image_patchs_d.size(), patch_cods_d.size())

            image_patchs_l = image_patchs_l.to(device)
            patch_cods_l = patch_cods_l.to(device)

            image_patchs_r = image_patchs_r.to(device)
            patch_cods_r = patch_cods_r.to(device)

            image_patchs_u = image_patchs_u.to(device)
            patch_cods_u = patch_cods_u.to(device)
            
            image_patchs_d = image_patchs_d.to(device)
            patch_cods_d = patch_cods_d.to(device)

            d_cost_l = d_cost_l.to(device)
            d_cost_r = d_cost_r.to(device)
            d_cost_u = d_cost_u.to(device)
            d_cost_d = d_cost_d.to(device)

            patch_x0.append(origin_patch)
            cod_x0.append(origin_cod)
            
            ### forward 
            p1p2_l = model(image_patchs_l , patch_cods_l)
            p1p2_r = model(image_patchs_r , patch_cods_r)
            p1p2_u = model(image_patchs_u , patch_cods_u)
            p1p2_d = model(image_patchs_d , patch_cods_d)
            
            p1p2_l = p1p2_l.detach_()
            p1p2_r = p1p2_r.detach_()
            p1p2_u = p1p2_u.detach_()
            p1p2_d = p1p2_d.detach_()

            loss,grad = pathloss(x,y,disp_val,d_cost_l,d_cost_r,d_cost_u,d_cost_d,p1p2_l,p1p2_r,p1p2_u,p1p2_d)
            loss_sum = loss_sum + loss 
            if(loss>0):
                cout_valid = cout_valid + 1
            grad_all.append(grad)
        
        if(cout_valid == 0):
            trained_num = trained_num + batch_size
            continue
        
        loss_sum = 1.0*loss_sum/cout_valid

        patch_x0 = torch.cat(patch_x0, dim=0)
        cod_x0 = torch.cat(cod_x0, dim=0)
        grad_all = torch.cat(grad_all, dim=0)
        
        patch_x0 = patch_x0.to(device)
        cod_x0 = cod_x0.to(device)
        grad_all = grad_all.to(device)
        
#use a patch to accept grad 
        p1p2_x0s = model(patch_x0, cod_x0)
        ### backward
        optimizer.zero_grad()
        p1p2_x0s.backward(grad_all)
        
        optimizer.step()
        
        print(p1p2_x0s)
        print('Iter_num: {}\t Loss: {}\t Grad: {}'.format(epoch*iter_num + iter_num,loss_sum,grad_all))
        writer.add_scalar('loss',loss_sum,epoch*iter_num + iter_num)
        if iter_num%10 == 0:
            torch.save(model.state_dict(), 'work_sapce/SGMNet.pkl')
            
        trained_num = trained_num + batch_size
    
    torch.save(model.state_dict(), 'work_sapce/SGMNet_fin.pkl')

