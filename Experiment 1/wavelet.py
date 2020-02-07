# -*- coding: utf-8 -*-
import pywt
import torch
from torch.autograd import Variable
from torchsummary import summary
torch.autograd.set_detect_anomaly(True)
w=pywt.Wavelet('haar') # Daubechies
# TODO change db1 inro 'haar'

dec_hi = torch.Tensor(w.dec_hi[::-1]) 
dec_lo = torch.Tensor(w.dec_lo[::-1])
rec_hi = torch.Tensor(w.rec_hi)
rec_lo = torch.Tensor(w.rec_lo)

# TODO write a script to understand the following:

#
filters = torch.stack([dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)/2.0,
                       dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                       dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)], dim=0)

inv_filters = torch.stack([rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)*2.0,
                           rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                           rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)], dim=0)




def wt2(vimg):
    #padded = torch.nn.functional.pad(vimg,(0,0,0,0)) 
    padded=vimg
    # TODO draw the padded torch
    # print the size of the output image and res
    # first dim of vimg is batch size
    '''
    print("filters size: ", filters.shape) #4,2,2
    print("dec_lo size: ",dec_lo.shape) #  torch.Size([2])
    print("dec_lo.unsqueeze(0) size: ",dec_lo.unsqueeze(0).shape) #torch.Size([1, 2])
    print("dec_lo.unsqueeze(1) size: ",dec_lo.unsqueeze(1).shape) #torch.Size([2, 1])
    
    print("dec_lo.unsqueeze(0): ",dec_lo.unsqueeze(0)) #torch.Size([1, 2])
    print("dec_lo.unsqueeze(1): ",dec_lo.unsqueeze(1)) #torch.Size([2, 1])
    
    print("filters: ", filters)
    
    
    print("vimg.shape: ", vimg.shape) # num_epochs,3 channels, width , height
    '''
    
 # original   res=torch.zeros(vimg.shape[0],4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))

    res=torch.zeros(vimg.shape[0],2*4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))


    res=res.cuda()
    temp_pad=torch.zeros(res.shape[0],1,res.shape[2]+1,res.shape[3]+1)
    temp_pad=temp_pad.cuda()
    #print(padded.shape)
    
    for i in range(padded.shape[1]):
        #print(i)
        # for res 4i->4i+4 will fill it with conv of padded(i-th)
     
        res[:,8*i:8*i+4] = torch.nn.functional.conv2d(padded[:,i:i+1], Variable(filters[:,None].cuda(),requires_grad=True),stride=2) # stride does the downsampling

        res[:,8*i+1:8*i+4]=(res[:,8*i+1:8*i+4]+1)/2.0 # added
  

  #second level of WT
        #if i==27:
           # print([k for k in range(8*i,8*i+1)])
        temp_pad[:,:,1:int(vimg.shape[2]/2)+1,1:int(vimg.shape[3]/2)+1]=res[:,8*i:8*i+1,:,:].clone()
        #print(temp_pad.shape)
        temp2 = torch.nn.functional.conv2d(temp_pad.clone(), Variable(filters[:,None].cuda(),requires_grad=True),stride=1)
        #if i==27:
            #print(temp2.shape)
        res[:,8*i+4:8*i+8]=temp2.clone()
        #original res[:,4*i+1:4*i+4]=(res[:,4*i+1:4*i+4]+1)/2.0
        res[:,8*i+5:8*i+8]=(res[:,8*i+5:8*i+8].clone()+1)/2.0


    return res

def iwt2(vres):
    #vres=2*vres-1
    ## original 
    #res=torch.zeros(vres.shape[0],int(vres.shape[1]/4),int(vres.shape[2]*2),int(vres.shape[3]*2))
    
    res=torch.zeros(vres.shape[0],int(vres.shape[1]/8),int(vres.shape[2]*2),int(vres.shape[3]*2))
    res=res.cuda()
    #print(res.shape)
    for i in range(res.shape[1]):
#        if res.shape[1]==3:
#            print(i)
        '''        
        vres[:,4*i+1:4*i+4]=2*vres[:,4*i+1:4*i+4]-1
        
        temp = torch.nn.functional.conv_transpose2d(vres[:,4*i:4*i+4], Variable(inv_filters[:,None].cuda(),requires_grad=True),stride=2)
        '''      
#temp[]
        vres[:,8*i+1:8*i+4]=2*vres[:,8*i+1:8*i+4]-1
#        temp1 = torch.nn.functional.conv_transpose2d(vres[:,8*i+4:8*i+8], Variable(inv_filters[:,None].cuda(),requires_grad=True),stride=2)
#        vres[:,8*i:8*i+1]=(temp1+vres[:,8*i:8*i+1])/2
        temp = torch.nn.functional.conv_transpose2d(vres[:,8*i:8*i+4], Variable(inv_filters[:,None].cuda(),requires_grad=True),stride=2)


        res[:,i:i+1,:,:]=temp
#        if res.shape[1]==3:
#            print('check')        
    return res








def iwt(vres):
    #vres=2*vres-1
    ## original 
    res=torch.zeros(vres.shape[0],int(vres.shape[1]/4),int(vres.shape[2]*2),int(vres.shape[3]*2))
    
    #res=torch.zeros(vres.shape[0],int(vres.shape[1]/8),int(vres.shape[2]*2),int(vres.shape[3]*2))
    res=res.cuda()
    #print(res.shape)
    for i in range(res.shape[1]):
#        if res.shape[1]==3:
#            print(i)
                
        vres[:,4*i+1:4*i+4]=2*vres[:,4*i+1:4*i+4]-1
        
        temp = torch.nn.functional.conv_transpose2d(vres[:,4*i:4*i+4], Variable(inv_filters[:,None].cuda(),requires_grad=True),stride=2)
        '''      
#temp[]
        vres[:,8*i+1:8*i+4]=2*vres[:,8*i+1:8*i+4]-1
#        temp1 = torch.nn.functional.conv_transpose2d(vres[:,8*i+4:8*i+8], Variable(inv_filters[:,None].cuda(),requires_grad=True),stride=2)
#        vres[:,8*i:8*i+1]=(temp1+vres[:,8*i:8*i+1])/2
        temp = torch.nn.functional.conv_transpose2d(vres[:,8*i:8*i+4], Variable(inv_filters[:,None].cuda(),requires_grad=True),stride=2)

        '''
        res[:,i:i+1,:,:]=temp
      
    return res















def wt(vimg):
    #padded = torch.nn.functional.pad(vimg,(0,0,0,0)) 
    padded=vimg
    
    
 # original   res=torch.zeros(vimg.shape[0],4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))

    res=torch.zeros(vimg.shape[0],4*vimg.shape[1],int(vimg.shape[2]/2),int(vimg.shape[3]/2))


    res=res.cuda()
    temp_pad=torch.zeros(res.shape[0],1,res.shape[2]+1,res.shape[3]+1)
    temp_pad=temp_pad.cuda()
    #print(padded.shape)
    
    for i in range(padded.shape[1]):
        #print(i)
        # for res 4i->4i+4 will fill it with conv of padded(i-th)
     
        res[:,4*i:4*i+4] = torch.nn.functional.conv2d(padded[:,i:i+1], Variable(filters[:,None].cuda(),requires_grad=True),stride=2) # stride does the downsampling

        res[:,4*i+1:4*i+4]=(res[:,4*i+1:4*i+4]+1)/2.0 # added
  



    return res

