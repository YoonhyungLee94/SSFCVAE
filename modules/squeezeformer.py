import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import *

class Scaling(nn.Module):
    def __init__(self, num_channels):
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, 1, num_channels))
        self.shift = nn.Parameter(torch.zeros(1, 1, num_channels))
        
    def forward(self, x, x_mask):
        return self.scale*x + self.shift

    
class FFNModule(nn.Module):
    def __init__(self, d_model, d_inner):
        super(FFNModule, self).__init__()
        self.scale = Scaling(d_model)
        self.sequential = nn.Sequential(nn.Linear(d_model, d_inner),
                                        nn.SiLU(),
                                        nn.Dropout(p=0.1),
                                        nn.Linear(d_inner, d_model),
                                        nn.Dropout(p=0.1))
        
    def forward(self, x, x_mask):
        return x+self.sequential(self.scale(x, x_mask))


class ConvModule(nn.Module):
    def __init__(self, hdim, groups=8):
        super(ConvModule, self).__init__()
        self.scale = Scaling(hdim)
        self.pw_conv1 = nn.Linear(hdim, hdim)
        self.dw_conv = nn.Conv1d(in_channels=hdim, out_channels=hdim, kernel_size=31, stride=1, padding=15, groups=hdim)
        self.gn = nn.GroupNorm(groups, hdim)
        self.pw_conv2 = nn.Linear(hdim, hdim)
        self.silu = nn.SiLU()
        
    def forward(self, x, x_mask):
        x2 = self.scale(x, x_mask)
        x2 = self.silu(self.pw_conv1(x2).transpose(1,2))*x_mask.transpose(1,2) # B, D, T
        x2 = self.silu(self.gn(self.dw_conv(x2)*x_mask.transpose(1,2)).transpose(1,2))  # B, T, D
        x2 = F.dropout(self.pw_conv2(x2), 0.1, self.training)
        
        return x+x2


class MultiHeadedSelfAttentionModule(nn.Module):
    """
    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout
    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs** (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """
    def __init__(self, d_model: int, num_heads: int, dropout_p: float = 0.1):
        super(MultiHeadedSelfAttentionModule, self).__init__()
        self.scale = Scaling(d_model)
        self.attention = RelativeMultiHeadAttention(d_model, num_heads, dropout_p)
        self.dropout = nn.Dropout(p=dropout_p)
        self.build_relative_positional_encoding(d_model, 2048)

    def forward(self, inputs, mask):
        batch_size, seq_length, _ = inputs.size()
        pos_embedding = self.pe[(2049-seq_length):(2048+seq_length)].unsqueeze(0)

        x2 = self.scale(inputs, mask)
        x2 = self.attention(x2, x2, x2, pos_embs=pos_embedding, key_padding_mask=~mask.squeeze(-1))
        x2 = self.dropout(x2)

        return inputs+x2
    
    def build_relative_positional_encoding(self, embedding_dim, max_length):
        position = torch.arange(-max_length, max_length+1, dtype=torch.float).unsqueeze(-1) # 2T+1, 1
        w_angular = torch.reciprocal(torch.pow(10000, torch.arange(0, embedding_dim, 2).float() / embedding_dim)).unsqueeze(0) # 1, D//2

        pe = torch.zeros(2*max_length+1, embedding_dim) # 2T+1, D
        pe[:, 0::2] = torch.sin(position * w_angular)
        pe[:, 1::2] = torch.cos(position * w_angular)
        self.register_buffer('pe', pe) # 2T+1, D
        return


class SqueezeformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_inner):
        super(SqueezeformerBlock, self).__init__()
        self.MHA = MultiHeadedSelfAttentionModule(d_model, n_head)
        self.ln1 = nn.LayerNorm(d_model)
        
        self.FFN1 = FFNModule(d_model, d_inner)
        self.ln2 = nn.LayerNorm(d_model)
        
        self.Conv = ConvModule(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        
        self.FFN2 = FFNModule(d_model, d_inner)
        self.ln4 = nn.LayerNorm(d_model)
        
    def forward(self, x, x_mask):
        y = self.ln1(self.MHA(x, x_mask))
        y = self.ln2(self.FFN1(y, x_mask))
        y = self.ln3(self.Conv(y, x_mask))
        y = self.ln4(self.FFN2(y, x_mask))
        
        # additional skip-connection is added for faster convergence
        return ((x+y) * x_mask) / 2**0.5 


class Squeezeformer(nn.Module):
    def __init__(self, hdim, n_head, d_inner, n_layers):
        super(Squeezeformer, self).__init__()
        self.SqueezeBlocks = nn.ModuleList([ SqueezeformerBlock(hdim, n_head, d_inner) for _ in range(n_layers) ])
        
    def forward(self, x, x_mask):
        for i, block in enumerate(self.SqueezeBlocks):
            x = block(x, x_mask)
        return x
    
