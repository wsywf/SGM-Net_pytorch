import torch
from torch.nn import Conv2d, Module, ReLU, MaxPool2d, AvgPool2d ,init, BatchNorm2d
import torch.nn.functional as F
import numpy as np
from torch import nn



#####   7x7
class SGM_NET(Module):
    def __init__(self):
        super(SGM_NET, self).__init__()
        self.conv_1 = Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1, padding = 0)
        self.conv_2 = Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 0)
        self.conv_3 = Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 0)
        self.fc1 = nn.Linear(in_features = 128 , out_features = 128)
        self.fc2 = nn.Linear(in_features = 128 , out_features = 8)
        self.relu = ReLU()
        self.avgpool = AvgPool2d(kernel_size = 2, stride=2)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                init.constant_(m.bias.data, 0.01)
        
    def forward(self, x,cod):
        x = self.conv_1(x)
        x = self.relu(x)
        
        x = self.conv_2(x)
        x = self.relu(x)
        
        x = self.conv_3(x)
        x = self.relu(x)
        
        x = x.view(-1, 128)
        
    #    x = torch.cat((x,cod),1)
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = F.elu(x)
        
        x = torch.add(x,1.0)

        return x

