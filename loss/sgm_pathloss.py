import torch
import numpy as np
import cv2

def left_aggregation(d_cost_l, p1p2_l, p1_left_ptr,p2_left_ptr): ### [Dmax, nl], [nl , 8]
    Dmax, num_pix = d_cost_l.size()
    last_Lr=d_cost_l[:,0]  
    grad = torch.zeros(Dmax,num_pix)

    for i in range(1,num_pix):
        minlast=torch.min(last_Lr)
        next_Lr=d_cost_l[:,i]
        
        tmp_grad = grad.clone()
        for nextd in range(0,Dmax):
            last_Lrp2 = last_Lr.clone()
            
            min0 = next_Lr[nextd] + last_Lr[nextd]
            last_Lrp2[nextd] = 1000.0
            
            min1_pos = 0
            if nextd == 0:
                min1 = next_Lr[nextd] + last_Lr[nextd+1] + p1p2_l[i-1,p1_left_ptr]
                last_Lrp2[nextd+1] = 1000.0
                min1_pos = nextd+1
                
            elif nextd == Dmax-1:
                min1 = next_Lr[nextd] + last_Lr[nextd-1] + p1p2_l[i-1,p1_left_ptr]
                last_Lrp2[nextd-1] = 1000.0
                min1_pos = nextd-1
            else:
                min1u = next_Lr[nextd] + last_Lr[nextd+1] + p1p2_l[i-1,p1_left_ptr]
                min1d = next_Lr[nextd] + last_Lr[nextd-1] + p1p2_l[i-1,p1_left_ptr]
                if(min1u < min1d):
                    min1 = min1u
                    last_Lrp2[nextd+1] = 1000.0
                    min1_pos = nextd+1
                else:
                    min1 = min1d
                    last_Lrp2[nextd-1] = 1000.0
                    min1_pos = nextd-1

            min2_pos = torch.argmin(last_Lrp2,dim=0).item()
            min2 = last_Lr[min2_pos] + next_Lr[nextd] + p1p2_l[i-1,p2_left_ptr]

            if(min1 < min0 and min1 < min2):
                next_Lr[nextd] = min1 - minlast
                grad[nextd] = tmp_grad[min1_pos]
                grad[nextd,i] = 1
            elif(min2 < min0 and min2 < min1):
                next_Lr[nextd] = min2 - minlast
                grad[nextd] = tmp_grad[min2_pos]
                grad[nextd,i] = 2
            else:
                next_Lr[nextd] = min0 - minlast
                grad[nextd] = tmp_grad[nextd]
        last_Lr = next_Lr
    return last_Lr,grad

def right_aggregation(d_cost_r, p1p2_r ,p1_right_ptr,p2_right_ptr):
    Dmax, num_pix = d_cost_r.size()
    last_Lr=d_cost_r[:,num_pix-1]
    grad = torch.zeros(Dmax,num_pix)
    
    for i in range(num_pix-2,-1,-1):
        minlast=torch.min(last_Lr)
        next_Lr=d_cost_r[:,i]
        
        tmp_grad = grad.clone()
        for nextd in range(0,Dmax):
            last_Lrp2 = last_Lr.clone()
                
            min0 = next_Lr[nextd] + last_Lr[nextd]
            last_Lrp2[nextd] = 1000.0
            
            min1_pos = 0
            if nextd == 0:
                min1 = next_Lr[nextd] + last_Lr[nextd+1] + p1p2_r[i,p1_right_ptr]
                last_Lrp2[nextd+1] = 1000.0
                min1_pos = nextd +1
            elif nextd == Dmax-1:
                min1 = next_Lr[nextd] + last_Lr[nextd-1] + p1p2_r[i,p1_right_ptr]
                last_Lrp2[nextd-1] = 1000.0
                min1_pos = nextd-1
            else:
                min1u = next_Lr[nextd] + last_Lr[nextd+1] + p1p2_r[i,p1_right_ptr]
                min1d = next_Lr[nextd] + last_Lr[nextd-1] + p1p2_r[i,p1_right_ptr]
                if(min1u < min1d):
                    min1 = min1u
                    last_Lrp2[nextd+1] = 1000.0
                    min1_pos = nextd+1
                else:
                    min1 = min1d
                    last_Lrp2[nextd-1] = 1000.0
                    min1_pos = nextd-1

            min2_pos = torch.argmin(last_Lrp2,dim=0).item()
            min2 = last_Lr[min2_pos] + next_Lr[nextd] + p1p2_r[i,p2_right_ptr]
            
            if(min1 < min0 and min1 < min2):
                next_Lr[nextd] = min1 - minlast
                grad[nextd] = tmp_grad[min1_pos]
                grad[nextd,i] = 1
            elif(min2 < min0 and min2 < min1):
                next_Lr[nextd] = min2 - minlast
                grad[nextd] = tmp_grad[min2_pos]
                grad[nextd,i] = 2
            else:
                next_Lr[nextd] = min0 - minlast
                grad[nextd] = tmp_grad[nextd]
        last_Lr = next_Lr
    return last_Lr,grad
    

def down_aggregation(d_cost_d,p1p2_d , p1_down_ptr,p2_down_ptr):
    Dmax, num_pix = d_cost_d.size()
    last_Lr=d_cost_d[:,num_pix-1]
    grad = torch.zeros(Dmax,num_pix)
    
    for i in range(num_pix-2,-1,-1):
        minlast=torch.min(last_Lr)
        next_Lr=d_cost_d[:,i]

        tmp_grad = grad.clone()
        for nextd in range(0,Dmax):
            last_Lrp2 = last_Lr.clone()
            min0 = next_Lr[nextd] + last_Lr[nextd]
            last_Lrp2[nextd] = 1000.0
            
            min1_pos = 0
            if nextd == 0:
                min1 = next_Lr[nextd] + last_Lr[nextd+1] + p1p2_d[i,p1_down_ptr]
                last_Lrp2[nextd+1] = 1000.0
                min1_pos = nextd+1
            elif nextd == Dmax-1:
                min1 = next_Lr[nextd] + last_Lr[nextd-1] + p1p2_d[i,p1_down_ptr]
                last_Lrp2[nextd-1] = 1000.0
                min1_pos = nextd-1
            else:
                min1u = next_Lr[nextd] + last_Lr[nextd+1] + p1p2_d[i,p1_down_ptr]
                min1d = next_Lr[nextd] + last_Lr[nextd-1] + p1p2_d[i,p1_down_ptr]
                if(min1u < min1d):
                    min1 = min1u
                    last_Lrp2[nextd+1] = 1000.0
                    min1_pos = nextd+1
                else:
                    min1 = min1d
                    last_Lrp2[nextd-1] = 1000.0
                    min1_pos = nextd-1
            
            min2_pos = torch.argmin(last_Lrp2,dim=0).item()
            min2 = last_Lr[min2_pos] + next_Lr[nextd] + p1p2_d[i,p2_down_ptr]

            if(min1 < min0 and min1 < min2):
                next_Lr[nextd] = min1 - minlast
                grad[nextd] = tmp_grad[min1_pos]
                grad[nextd,i] = 1
            elif(min2 < min0 and min2 < min1):
                next_Lr[nextd] = min2 - minlast
                grad[nextd] = tmp_grad[min2_pos]
                grad[nextd,i] = 2
            else:
                next_Lr[nextd] = min0 - minlast
                grad[nextd] = tmp_grad[nextd]
        last_Lr = next_Lr
    return last_Lr,grad

def up_aggregation(d_cost_u,p1p2_u, p1_up_ptr,p2_up_ptr):
    Dmax, num_pix = d_cost_u.size()
    last_Lr=d_cost_u[:,0]
    grad = torch.zeros(Dmax,num_pix)
    
    for i in range(1,num_pix):
        minlast=torch.min(last_Lr)
        next_Lr=d_cost_u[:,i]
          
        tmp_grad = grad.clone()
        for nextd in range(0,Dmax):
            last_Lrp2 = last_Lr.clone()
            min0 = next_Lr[nextd] + last_Lr[nextd]
            last_Lrp2[nextd] = 1000.0
                
            min1_pos = 0
            if nextd == 0:
                min1 = next_Lr[nextd] + last_Lr[nextd+1] + p1p2_u[i-1,p1_up_ptr]
                last_Lrp2[nextd+1] = 1000.0
                min1_pos = nextd+1
            elif nextd == Dmax-1:
                min1 = next_Lr[nextd] + last_Lr[nextd-1] + p1p2_u[i-1,p1_up_ptr]
                last_Lrp2[nextd-1] = 1000.0
                min1_pos = nextd-1
            else:
                min1u = next_Lr[nextd] + last_Lr[nextd+1] + p1p2_u[i-1,p1_up_ptr]
                min1d = next_Lr[nextd] + last_Lr[nextd-1] + p1p2_u[i-1,p1_up_ptr]
                if(min1u < min1d):
                    min1 = min1u
                    last_Lrp2[nextd+1] = 1000.0
                    min1_pos = nextd +1
                else:
                    min1 = min1d
                    last_Lrp2[nextd-1] = 1000.0
                    min1_pos = nextd-1

            min2_pos = torch.argmin(last_Lrp2,dim=0).item()
            min2 = last_Lr[min2_pos] + next_Lr[nextd] + p1p2_u[i-1,p2_up_ptr]

            if(min1 < min0 and min1 < min2):
                next_Lr[nextd] = min1 - minlast
                grad[nextd] = tmp_grad[min1_pos]
                grad[nextd,i] = 1
            elif(min2 < min0 and min2 < min1):
                next_Lr[nextd] = min2 - minlast
                grad[nextd] = tmp_grad[min2_pos]
                grad[nextd,i] = 2
            else:
                next_Lr[nextd] = min0 - minlast
                grad[nextd] = tmp_grad[nextd]
        last_Lr = next_Lr
    return last_Lr,grad

def pathloss(x,y,disp_val,d_cost_l,d_cost_r,d_cost_u,d_cost_d,p1p2_l,p1p2_r,p1p2_u,p1p2_d):
    p1_left_ptr,p2_left_ptr = 0,1
    p1_right_ptr,p2_right_ptr = 2,3
    p1_up_ptr,p2_up_ptr = 4,5
    p1_down_ptr,p2_down_ptr = 6,7
    m = 5.0
    disp_gt = disp_val

    left_aggr,left_grad = left_aggregation(d_cost_l, p1p2_l , p1_left_ptr,p2_left_ptr)
    right_aggr,right_grad = right_aggregation(d_cost_r, p1p2_r, p1_right_ptr,p2_right_ptr)
    up_aggr,up_grad= up_aggregation(d_cost_u,p1p2_u ,p1_up_ptr,p2_up_ptr)
    down_aggr,down_grad= down_aggregation(d_cost_d,p1p2_d,p1_down_ptr,p2_down_ptr)

#    print(left_aggr)
#    print(right_aggr)
#    print(up_aggr)
#    print(down_aggr)
    path_grad = torch.zeros(1,8)
    loss = 0
    left_cost_gt = left_aggr[disp_gt]
    num1 = (left_grad[disp_gt] == 1).nonzero().size()[0]
    num2 = (left_grad[disp_gt] == 2).nonzero().size()[0]
    for ind in range(left_aggr.size()[0]):
        if(ind == disp_gt):
            continue
        else:
            if((left_cost_gt - left_aggr[ind] +m)>0):
                loss = loss + left_cost_gt - left_aggr[ind] +m
               # loss = loss + left_aggr[ind] - left_cost_gt
                path_grad[0,p1_left_ptr] = path_grad[0,p1_left_ptr] + num1 - (left_grad[ind] == 1).nonzero().size()[0]
                path_grad[0,p2_left_ptr] = path_grad[0,p2_left_ptr] + num2 - (left_grad[ind] == 2).nonzero().size()[0]

    right_cost_gt = right_aggr[disp_gt]
    num1 = (right_grad[disp_gt] == 1).nonzero().size()[0]
    num2 = (right_grad[disp_gt] == 2).nonzero().size()[0]
    for ind in range(right_aggr.size()[0]):
        if(ind == disp_gt):
            continue
        else:
            if((right_cost_gt - right_aggr[ind] +m)>0):
                loss = loss + right_cost_gt - right_aggr[ind] +m
              #  loss = loss + right_aggr[ind] -  right_cost_gt
                path_grad[0,p1_right_ptr] = path_grad[0,p1_right_ptr] + num1 - (right_grad[ind] == 1).nonzero().size()[0]
                path_grad[0,p2_right_ptr] = path_grad[0,p2_right_ptr] + num2 - (right_grad[ind] == 2).nonzero().size()[0]

    up_cost_gt = up_aggr[disp_gt]
    num1 = (up_grad[disp_gt] == 1).nonzero().size()[0]
    num2 = (up_grad[disp_gt] == 2).nonzero().size()[0]
    for ind in range(up_aggr.size()[0]):
        if(ind == disp_gt):
            continue
        else:
            if((up_cost_gt - up_aggr[ind] +m)>0):
                loss = loss + up_cost_gt - up_aggr[ind] +m
               # loss = loss + up_aggr[ind] - up_cost_gt
                path_grad[0,p1_up_ptr] = path_grad[0,p1_up_ptr] + num1 - (up_grad[ind] == 1).nonzero().size()[0]
                path_grad[0,p2_up_ptr] = path_grad[0,p2_up_ptr] + num2 - (up_grad[ind] == 2).nonzero().size()[0]

    down_cost_gt = down_aggr[disp_gt]
    num1 = (down_grad[disp_gt] == 1).nonzero().size()[0]
    num2 = (down_grad[disp_gt] == 2).nonzero().size()[0]
    for ind in range(down_aggr.size()[0]):
        if(ind == disp_gt):
            continue
        else:
            if((down_cost_gt - down_aggr[ind] +m)>0):
                loss = loss + down_cost_gt - down_aggr[ind] +m
              #  loss = loss + down_aggr[ind] - down_cost_gt
                path_grad[0,p1_down_ptr] = path_grad[0,p1_down_ptr] + num1 - (down_grad[ind] == 1).nonzero().size()[0]
                path_grad[0,p2_down_ptr] = path_grad[0,p2_down_ptr] + num2 - (down_grad[ind] == 2).nonzero().size()[0]
    
    left_min = torch.argmin(left_aggr,dim=0).item()
    right_min = torch.argmin(right_aggr,dim=0).item()
    up_min = torch.argmin(up_aggr,dim=0).item()
    down_min = torch.argmin(down_aggr,dim=0).item()
    disp_abs = abs(left_min-disp_gt) + abs(right_min-disp_gt) + abs(up_min-disp_gt) + abs(down_min-disp_gt)
#outliter restrain    
    if(abs(left_min-disp_gt)>=10 and abs(right_min-disp_gt)>=10 and abs(up_min-disp_gt)>=10 and abs(down_min-disp_gt)>=10):
        path_grad = torch.zeros(1,8)
        loss = 0
    print('x: {}\t y: {}\t gt: {}\t left: {}\t right: {}\t up: {}\t down: {}\t disp_abs: {}'.format(x, y, disp_gt, left_min,right_min,up_min,down_min,disp_abs))
    return loss,path_grad

