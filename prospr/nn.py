''' HOW THIS MODULE IS USED'''
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.functional import softmax 
import pickle as pkl
import numpy as np
import random
import torch.optim as optim
import subprocess
import torch.utils.data
#helper functions
#GLOBAL VARIABLE
NUM_RESNET_BLOCKS = 220
INPUT_DIM = 675
OUTPUT_BINS = 64 #number of bins in output
RESNET_DIM = 128 #number of layers inside of resnet
CROP_SIZE = 64
BATCH_SIZE = 1

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, dilation = 1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.project_down = conv1x1(128, 64, stride=1)
        self.project_up   = conv1x1(64, 128, stride=1)
        self.bn64_1 = norm_layer(64)
        self.bn64_2 = norm_layer(64)
        self.bn128 = norm_layer(128)

        #dilations deal now with 64 incoming and 64 outcoming layers
        self.dilation = conv3x3(64, 64, stride, dilation = dilation) #when the block is initialized, the only thing that changes is the dilation filter used!
        self.elu = nn.ELU(inplace=True)

    def forward(self, x):
        
        identity = x
    
        #the deepmind basic block goes:
        
        #batchnorm
        out = self.bn128(x)
        
        #elu
        out = self.elu(out)
    
        #project down to 64
        out = self.project_down(out)
        
        #batchnorm
        out = self.bn64_1(out)

        #elu
        out = self.elu(out)   
        
        #cycle through 4 dilations
        out = self.dilation(out)  
        
        #batchnorm
        out = self.bn64_2(out)

        #elu
        out = self.elu(out)
        
        #project up to 128
        out = self.project_up(out)
        
        #identitiy addition 
        out = out + identity

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=64*64, zero_init_residual=False, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = RESNET_DIM
        
        self.conv1 = conv1x1(INPUT_DIM, RESNET_DIM, stride=1)
        self.conv2 = conv1x1(RESNET_DIM, OUTPUT_BINS )
        
        self.proj_aux = conv1x1(RESNET_DIM, 83,stride=1)
        self.conv_aux = conv64x1(83,83,stride=1,groups=1)
  
        self.elu = nn.ELU(inplace=True)
        
        self.bn1 = norm_layer(INPUT_DIM) 

        self.resnet_blocks = self._make_layer(block, RESNET_DIM, layers[0], norm_layer=norm_layer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer))
        self.inplanes = planes * block.expansion
        
        #here I need to pass in the correct dilations 1,2,4,8
        dilations = [1,2,4,8]
        
        for i,_ in enumerate(range(1, blocks)):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, dilation = dilations[i % 4]))

        return nn.Sequential(*layers)

    def forward(self, x):
        #fix input dimensions
        x = self.bn1(x)    #Why?
        x = self.conv1(x)
  
        #propagate through RESNET blocks
        resnet_out = self.resnet_blocks(x)
        #renet_out has shape 1,128,64,64
        
        #fix output dimensions
        x = self.conv2(resnet_out) #return 64x64x64 

        aux = self.proj_aux(resnet_out)        
        #should we have elu / batchnorm(s) here??
        aux_i = self.conv_aux(torch.transpose(aux,2,3))
        aux_j = self.conv_aux(aux)
        
        #FIX THIS TO WORK WITH BATCHES!
        return x, aux_i[:,:9], aux_j[:,:9], aux_i[:,9:9+37], aux_j[:,9:9+37], aux_i[:,9+37:9+2*37], aux_j[:,9+37:9+2*37]


def is_training():
    pass  # change BATCH_SIZE

def conv3x3(in_planes, out_planes, stride=1, dilation = 1):
    """3x3 convolution with padding"""
    padding = 1 + (dilation -1 ) #derived to ensure consistent size
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, bias=True, dilation = dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)

def conv64x1(in_planes, out_planes, stride=1, groups=1):
    """64x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(64,1), stride=stride, groups=groups, bias=True)
    
#Define the ProSPr CNN:

def prospr( **kwargs):
    model = ResNet(BasicBlock, [NUM_RESNET_BLOCKS, 0, 0, 0], **kwargs)
    return model  



