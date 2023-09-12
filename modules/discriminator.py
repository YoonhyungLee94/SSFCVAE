import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm as sn
from torch.nn.utils import weight_norm as wn
from utils import *


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super(Discriminator, self).__init__()
        self.conv1 = sn(nn.Conv1d(in_dim, 128, kernel_size=3, dilation=1, padding=1))
        self.conv2 = wn(nn.Conv1d(128, 1, kernel_size=3, dilation=1, padding=1))
        self.ResBlocks = nn.ModuleList([ResBlock() for _ in range(5)])
        
    def forward(self, x, x_mask):
        x = F.dropout(self.conv1(x) * x_mask.unsqueeze(1), 0.5, self.training)
        for block in self.ResBlocks:
            x = block(x, x_mask)
            
        y = self.conv2(x).squeeze(1)
        
        return y
    
    
class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv1 = wn(nn.Conv1d(128, 128, kernel_size=3, dilation=3, padding=3))
        self.conv2 = wn(nn.Conv1d(128, 128, kernel_size=3, dilation=3, padding=3))
        self.conv3 = wn(nn.Conv1d(128, 128, kernel_size=1, dilation=1, padding=0))
        
    def forward(self, x, x_mask=None):
        y = self.conv1(self.lrelu(x)) * x_mask.unsqueeze(1)
        y = self.conv2(self.lrelu(y)) * x_mask.unsqueeze(1)
        x = (self.conv3(x) + y) * x_mask.unsqueeze(1)
        return x
