import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import modules.transforms as transforms


class NFBlock(nn.Module):
    def __init__(self, latent_dim, hdim, n_flows):
        super(NFBlock, self).__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows//2):
            self.flows.append(IAFlow(latent_dim, hdim, 'forward'))
            self.flows.append(IAFlow(latent_dim, hdim, 'backward'))
        
        
    def forward(self, x, x_mask):
        logdet_tot = 0
        for flow in self.flows:
            x, logdet = flow(x, x_mask)
            logdet_tot += logdet
        return x, logdet_tot


class IAFlow(nn.Module):
    def __init__(self, latent_dim, hdim, direction, num_bins=16, tail_bound=5.0):
        super(IAFlow, self).__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.latent_dim = latent_dim
        self.hdim = hdim
        self.direction = direction
        
        self.pre = nn.Conv1d(latent_dim, hdim, 1)
        self.lstm = nn.LSTM(hdim, hdim, 1, batch_first=True)
        self.proj = nn.Conv1d(hdim, latent_dim * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        
        bias = torch.zeros(latent_dim, (num_bins * 3 - 1))
        bias[:, 2*num_bins:].fill_(math.log(math.exp(1-1e-3)-1))
        self.proj.bias.data.copy_(bias.reshape(-1))
        
    def forward(self, x, x_mask):
        B, D, T = x.size()
        
        h = self.pre(x)*x_mask
        if self.direction=='forward':
            h = F.pad(h, (1, -1)).transpose(1,2)
            h = self.lstm(h)[0].transpose(1, 2)
            
        elif self.direction=='backward':
            h = F.pad(h, (-1, 1)).transpose(1,2)
            h = torch.flip(h, [1])
            h = self.lstm(h)[0]
            h = torch.flip(h, [1]).transpose(1, 2)
            
        h = self.proj(h)*x_mask
        
        b, c, t = x.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)
        
        unnormalized_widths = h[..., :self.num_bins] / math.sqrt(self.hdim)
        unnormalized_heights = h[..., self.num_bins:2*self.num_bins] / math.sqrt(self.hdim)
        unnormalized_derivatives = h[..., 2*self.num_bins:]

        x, logabsdet  = transforms.piecewise_rational_quadratic_transform(x,
                                                                          unnormalized_widths,
                                                                          unnormalized_heights,
                                                                          unnormalized_derivatives,
                                                                          inverse=False,
                                                                          tails='linear',
                                                                          tail_bound=self.tail_bound
                                                                          )

        return x, logabsdet.sum(1)*x_mask.squeeze(1)
