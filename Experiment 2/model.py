from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchsummary import summary
import pywt
from wavelet import wt2,wt,iwt,iwt2
class Waveletnet(nn.Module):
    def __init__(self):
        super(Waveletnet, self).__init__()
        self.num=1
        #original c=16
        c = 28
        '''
        # original
        self.conv1 = nn.Conv2d(12,c,3, 1,padding=1)
        self.conv2 = nn.Conv2d(4*c,4*c,3, 1,padding=1)
        self.conv3 = nn.Conv2d(16*c,16*c,3, 1,padding=1)
        self.conv4 = nn.Conv2d(64*c,64*c,3, 1,padding=1)
        self.bn = nn.BatchNorm2d(320)
        self.convd1 = nn.Conv2d(c,12,3, 1,padding=1)
        self.convd2 = nn.Conv2d(2*c,c,3, 1,padding=1)
        self.convd3 = nn.Conv2d(8*c,4*c,3, 1,padding=1)
        self.convd4 = nn.Conv2d(32*c,16*c,3, 1,padding=1)
        self.relu = nn.LeakyReLU(0.2)
        
        ##two level in all layers
        
                                                           # 3 to 24 by w1
        self.conv1 = nn.Conv2d(24,c,3, 1,padding=1)        #24  to c  increase by 4
                                                           # c to 8c by w2
        self.conv2 = nn.Conv2d(8*c,8*c,3, 1,padding=1)     # 8c to 8c preserve
                                                           # 8c to 64c by w3
        self.conv3 = nn.Conv2d(64*c,64*c,3, 1,padding=1)   # 64c to 64c preserve
                                                           # 64c to 512c by w4
        self.conv4 = nn.Conv2d(64*8*c,64*8*c,3, 1,padding=1)#512c to 512c preserve
        self.bn = nn.BatchNorm2d(320) # batchnorm2d
                                                           # 512c to 64c by iw4
                                                           # 64c to 128c by cat to output of conv3
        self.convd4 = nn.Conv2d(2*64*c,64*c,3, 1,padding=1)# 128c to 64c divide by 2
                                                           # 64c to 8c by iw3
                                                           # 8c to 16c by cat to output of conv2
        self.convd3 = nn.Conv2d(16*c,8*c,3, 1,padding=1)   #16c to 8c divide cat to output
                                                           # 8c to c by iw2
                                                           # c to 2c by cat to output of conv1
        self.convd2 = nn.Conv2d(2*c,c,3, 1,padding=1)      # 2c to c divide by 2
        self.convd1 = nn.Conv2d(c,24,3, 1,padding=1)       # c to 24 decrease by 2
                                                           # 24 to 3 to have output channels
        self.relu = nn.LeakyReLU(0.2)
        '''
        # first layer two level 
        self.conv1 = nn.Conv2d(24,c,3, 1,padding=1)        #24  to c  increase by 4
                                                           # c to 8c by w2
        self.conv2 = nn.Conv2d(4*c,4*c,3, 1,padding=1)     # 8c to 8c preserve
                                                           # 8c to 64c by w3
        self.conv3 = nn.Conv2d(16*c,16*c,3, 1,padding=1)   # 64c to 64c preserve
                                                           # 64c to 512c by w4
        self.conv4 = nn.Conv2d(64*c,64*c,3, 1,padding=1)#512c to 512c preserve
        self.bn = nn.BatchNorm2d(320) # batchnorm2d
                                                           # 512c to 64c by iw4
                                                           # 64c to 128c by cat to output of conv3
        self.convd4 = nn.Conv2d(32*c,16*c,3, 1,padding=1)# 128c to 64c divide by 2
                                                           # 64c to 8c by iw3
                                                           # 8c to 16c by cat to output of conv2
        self.convd3 = nn.Conv2d(8*c,4*c,3, 1,padding=1)   #16c to 8c divide cat to output
                                                           # 8c to c by iw2
                                                           # c to 2c by cat to output of conv1
        self.convd2 = nn.Conv2d(2*c,c,3, 1,padding=1)      # 2c to c divide by 2
        self.convd1 = nn.Conv2d(c,24,3, 1,padding=1)       # c to 24 decrease by 2
                                                           # 24 to 3 to have output channels
        self.relu = nn.LeakyReLU(0.2)


    def forward(self, x):
        '''
        device1 = torch.device("cuda:0")
        device2 = torch.device("cuda:1")
        device3 = torch.device("cuda:2")
        device4 = torch.device("cuda:3")
        
        
        
        #print(x.shape)
        w1=wt2(x)
        #print(w1.shape)
        c1=self.relu(self.conv1(w1))
        #print(c1.shape)
        w2=wt(c1)
        #print(w2.shape)
        c2=self.relu(self.conv2(w2))
        
        #change GPU
        #c2 = c2.to(device2)
        
        #print(c2.shape)
        w3=wt(c2)
        #print(w3.shape)
        c3=self.relu(self.conv3(w3))
        #print(c3.shape)
#        w4=wt(c3)
#        c4=self.relu(self.conv4(w4))
#       c3=self.reu(self.conv4(w3))
#        c5=self.relu(self.conv4(c4))
        
        # Change GPU
        #c3 = 
        
        c4=self.relu(self.conv3(c3))
        #print(c4.shape)
#        c6=(self.conv4(c5))
        c5=(self.conv3(c4))
        #print(c5.shape)
#        ic4=self.relu(c6+w4)
        ic3=self.relu(c5+w3)   #new
        #print(ic3.shape)
#        iw4=iwt(ic4)
        iw3=iwt(ic3) #new
        #print(iw3.shape)
#        iw4=torch.cat([c3,iw4],1)
        iw3=torch.cat([c2,iw3],1)   #new
        #print(iw3.shape)
#        ic3=self.relu(self.convd4(iw4))
#       ic2=self.relu(self.convd3(iw3)) #new
#        iw3=iwt(ic3)
#       iw2=iwt(ic2) #new
#        iw3=torch.cat([c2,iw3],1)
        ic2=self.relu(self.convd3(iw3))
        #print(ic2.shape)
        iw2=iwt(ic2)
        #print(iw2.shape)
        iw2=torch.cat([c1,iw2],1)
        #print(iw2.shape)
        ic1=self.relu(self.convd2(iw2))
        #print(ic1.shape)
        iw1=self.relu(self.convd1(ic1))
        #print(iw1.shape)
        
        y=iwt2(iw1)
        #print(y.shape)
        '''
        w1=wt2(x)
        c1=self.relu(self.conv1(w1))
        w2=wt(c1)
        
        
        c2=self.relu(self.conv2(w2))
        
        w3=wt(c2)
        c3=self.relu(self.conv3(w3))
        w4=wt(c3)
        c4=self.relu(self.conv4(w4))
        c5=self.relu(self.conv4(c4))
        c6=(self.conv4(c5))
        ic4=self.relu(c6+w4)
        iw4=iwt(ic4)
        iw4=torch.cat([c3,iw4],1)
        ic3=self.relu(self.convd4(iw4))
        iw3=iwt(ic3)
        iw3=torch.cat([c2,iw3],1)
        ic2=self.relu(self.convd3(iw3))
        iw2=iwt(ic2)
        iw2=torch.cat([c1,iw2],1)
        ic1=self.relu(self.convd2(iw2))
        iw1=self.relu(self.convd1(ic1))

        y=iwt2(iw1)


        return y
class ACT(nn.Module):
    def __init__(self):
        super(ACT, self).__init__()
        self.net = Waveletnet()
        self.c = torch.nn.Conv2d(3,3,1,padding=0, bias=False)
    def forward(self, x):
        x = self.net(x)
        x1 = self.c(x)
        x2 =x + x1
        return x