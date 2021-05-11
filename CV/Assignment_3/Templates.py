from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import torch
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from Utils import *
from Templates import *
from torch.utils.data import Dataset, DataLoader

def encode_m1(in_channels, out_channels):
    L = nn.Sequential(
        nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding= 1),
        nn.LeakyReLU(negative_slope = 0.2,inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope = 0.2,inplace =True),
#         nn.MaxPool2d(2)
    )
    return L

def decode_m1(in_channels, out_channels):
    L = nn.Sequential(
        nn.Upsample(scale_factor = (2,2),mode='bilinear'),
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
        nn.LeakyReLU(negative_slope = 0.2,inplace = True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1,padding=1),
#         nn.ReLU(True)
    )
        
    return L

class Model(nn.Module):
    def __init__(self,in_ch= 1, out_ch = 1):
        super(Model,self).__init__()
        
        self.out_ch = out_ch
        self.l1 = encode_m1(in_ch,64)
        self.l2 = encode_m1(64,128)
        self.l3 = encode_m1(128,128)
        self.l4 = encode_m1(128,128)
        self.l5 = decode_m1(128 + 128,64)
        self.l6 = decode_m1(64 + 64,out_ch)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self,x):
        x1 = self.pool(self.l1(x)) #64 
        x2 = self.pool(self.l2(x1)) #128
        x3 = self.l5(torch.cat([ self.l4(self.l3(x2)), x2],dim = 1))
        x4 = self.l6(torch.cat([x3,x1],dim = 1))
#         print(x4.shape)
        if self.out_ch >1 :
            x = torch.log_softmax(x4,dim = 1)
        else :
            x = torch.sigmoid(x4)
        return x
    
class Model_Q2(nn.Module):
    def __init__(self):
        super(Model_Q2,self).__init__()
    
        self.m1 = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 6,kernel_size = 5, stride = 1, padding = 2),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.MaxPool2d(kernel_size = 2),
            nn.Tanh(),
            nn.Conv2d(in_channels = 6, out_channels = 18,kernel_size = 5, stride = 1, padding = 0),
            nn.LeakyReLU(negative_slope = 0.2),
            nn.MaxPool2d(kernel_size = 2),
            nn.Tanh(),
            nn.Conv2d(in_channels = 18, out_channels = 120,kernel_size = 5, stride = 1, padding = 0),
            nn.Tanh()
        )
        self.m2 = nn.Sequential(
            nn.Linear(in_features = 120 , out_features = 84),
            nn.Tanh(),
            nn.Linear(in_features = 84, out_features = 10)
        )
        self.m3 = nn.Sequential(
            nn.Linear(in_features = 120, out_features = 120), 
            nn.Linear(in_features = 120, out_features = 84),
            nn.Linear(in_features = 84, out_features = 3)
        )

    def forward(self,x):
        x = self.m1(x)
        x = torch.flatten(x, 1)
        logits = self.m2(x)
        labels = self.m3(x)
        return torch.softmax(logits,dim = 1), torch.sigmoid(labels)
