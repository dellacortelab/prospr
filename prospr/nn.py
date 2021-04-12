import torch
import torch.nn as nn

INPUT_DIM = 547
CROP_SIZE = 64
DIST_BINS = 10 
AUX_BINS = 94
SS_BINS = 9
ANGLE_BINS = 37
ASA_BINS = 11
DROPOUT_RATE = 0.15 


CUDA = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

def load_model(model, fname):
    """load pytoch state_dict into predefined model"""
    model.load_state_dict(torch.load(fname,map_location=CUDA))
    return model


def conv3x3(in_channels, out_channels):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

def conv3x3_dilated(in_channels, out_channels, dilation = 1):
    """dilated 3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation = dilation)

def conv1x1(in_channels, out_channels):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)

def conv64x1(in_channels, out_channels):
    """64x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=(64,1))

def conv1x64(in_channels, out_channels):
    """1x64 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=(1,64))


class Block(nn.Module):
    def __init__(self, in_channels, dilation=1):
        super(Block, self).__init__()
        self.norm1    = nn.BatchNorm2d(in_channels)
        self.project_down = conv1x1(in_channels, in_channels//2)
        self.norm2    = nn.BatchNorm2d(in_channels//2)
        self.dilation = conv3x3_dilated(in_channels//2, in_channels//2, dilation=dilation)
        self.norm3    = nn.BatchNorm2d(in_channels//2)
        self.project_up = conv1x1(in_channels//2, in_channels)
        self.elu      = nn.ELU(inplace=True)
        self.dropout = nn.Dropout2d(p=DROPOUT_RATE, inplace=True)

    def forward(self, x):
        identity = x
        out = self.norm1(x)
        out = self.elu(out)
        out = self.project_down(out)
        out = self.norm2(out)
        out = self.elu(out)   
        out = self.dilation(out) 
        out = self.dropout(out)
        out = self.norm3(out)
        out = self.elu(out)
        out = self.project_up(out)
        return out + identity

class ProsprNetwork(nn.Module):
    def __init__(self):
        super(ProsprNetwork, self).__init__()
        self.bn1 = nn.BatchNorm2d(INPUT_DIM)
        self.conv1 = conv1x1(INPUT_DIM, 256)
        self.dropout = nn.Dropout2d(p=DROPOUT_RATE, inplace=True)
        self.conv2 = conv1x1(128, DIST_BINS)
        self.conv_aux_i = conv64x1(128,AUX_BINS)
        self.conv_aux_j = conv1x64(128,AUX_BINS)
        self.blocks = self._make_layer()

    def _make_layer(self):
        layers = []
        dilations = [1,2,4,8]
        for i in range(28):
            layers.append(Block(256, dilation = dilations[i % 4]))
        layers.append(conv1x1(256,128))
        for i in range(192):
            layers.append(Block(128, dilation = dilations[i % 4]))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.blocks(x)
        distogram = self.conv2(x)
        aux_i_out = torch.squeeze(self.conv_aux_i(x), dim=2)
        aux_j_out = torch.squeeze(self.conv_aux_j(x), dim=3)
        aux_i = dict()
        aux_i["ss"]  = aux_i_out[:,:SS_BINS,:]
        aux_i["phi"] = aux_i_out[:,SS_BINS:SS_BINS+ANGLE_BINS,:]
        aux_i["psi"] = aux_i_out[:,SS_BINS+ANGLE_BINS:SS_BINS+ANGLE_BINS+ANGLE_BINS,:]
        aux_i['asa'] = aux_i_out[:,SS_BINS+ANGLE_BINS+ANGLE_BINS:,:]
        aux_j = dict() 
        aux_j["ss"]  = aux_j_out[:,:SS_BINS,:]
        aux_j["phi"] = aux_j_out[:,SS_BINS:SS_BINS+ANGLE_BINS,:]
        aux_j["psi"] = aux_j_out[:,SS_BINS+ANGLE_BINS:SS_BINS+ANGLE_BINS+ANGLE_BINS,:]
        aux_j['asa'] = aux_j_out[:,SS_BINS+ANGLE_BINS+ANGLE_BINS:,:]

        return distogram, aux_i, aux_j
   