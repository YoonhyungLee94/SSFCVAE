import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from modules.squeezeformer import Squeezeformer
from modules.flows import NFBlock


class Model(nn.Module):
    def __init__(self, n_mel_channels, n_spec_channels, n_w2v_channels, hdim, latent_dim, n_head, d_inner, n_layers, n_flows):
        super(Model, self).__init__()
        self.hdim = hdim
        self.latent_dim = latent_dim
        
        self.spec_embed = nn.Conv1d(n_spec_channels, hdim, 1)
        self.w2v_embed = nn.Conv1d(n_w2v_channels, hdim, 1)
        self.encoder1 = Squeezeformer(hdim, n_head, d_inner, n_layers)
        self.encoder2 = Squeezeformer(hdim, n_head, d_inner, n_layers)
            
        self.encoder3 = Squeezeformer(hdim, n_head, d_inner, n_layers)
        self.mu_p = nn.Conv1d(hdim, latent_dim, 1)
        self.mu_q = nn.Conv1d(hdim, latent_dim, 1)

        self.flow = NFBlock(latent_dim, hdim, n_flows)
        self.w2v_proj = nn.Conv1d(latent_dim, hdim, 1)
            
        self.proj_block = Squeezeformer(hdim, n_head, d_inner, n_layers)
        self.proj_conv = nn.Conv1d(hdim, n_mel_channels, 1)
        
        self.register_buffer('indices', torch.arange(2048))
       
    
    def forward(self, spec_noisy, spec_lengths, w2v_feats_clean, w2v_feats_noisy, w2v_lengths):
        x_mask = self.get_mask_from_lengths(spec_lengths, spec_noisy.size(-1))
        w2v_mask = self.get_mask_from_lengths(w2v_lengths, w2v_feats_noisy.size(-1))
        
        x1 = self.spec_embed(spec_noisy)
        x1 = self.encoder1(x1.transpose(1, 2), x_mask[:,:,None]).transpose(1, 2)
        
        x1_d = self.resample(x1, spec_lengths, w2v_lengths, w2v_feats_noisy.size(-1)) + self.w2v_embed(w2v_feats_noisy)
        x2 = self.encoder2(x1_d.transpose(1, 2), w2v_mask[:,:,None]).transpose(1, 2)
        m_p = self.mu_p(x2)

        x3 = self.w2v_embed(w2v_feats_clean) - x1_d
        x3 = self.encoder3(x3.transpose(1, 2), w2v_mask[:,:,None]).transpose(1, 2)
        dm = self.mu_q(x3)
        m_q = m_p + dm

        z_q = m_q + torch.randn_like(m_q)
        z_p, logdet = self.flow(z_q, w2v_mask.unsqueeze(1))

        kl = (-0.5 + 0.5 * (z_p - m_p)**2).sum(1) - logdet
        kl_loss = torch.masked_select(kl, w2v_mask).mean()

        x4 = x1 + self.resample(self.w2v_proj(z_p)*w2v_mask.unsqueeze(1), w2v_lengths, spec_lengths, spec_noisy.size(-1))
        x4 = self.proj_block(x4.transpose(1, 2), x_mask[:,:,None]).transpose(1, 2)
        mel_pred = self.proj_conv(x4)

        return mel_pred, kl_loss, x_mask, w2v_mask, z_p
    
    
    @torch.no_grad()
    def inference(self, spec_noisy, spec_lengths, w2v_feats_noisy, w2v_lengths, noise_scale=0.2):
        x_mask = self.get_mask_from_lengths(spec_lengths, spec_noisy.size(-1))
        w2v_mask = self.get_mask_from_lengths(w2v_lengths, w2v_feats_noisy.size(-1))
        
        x1 = self.spec_embed(spec_noisy)
        x1 = self.encoder1(x1.transpose(1, 2), x_mask[:,:,None]).transpose(1, 2)
        
        x1_d = self.resample(x1, spec_lengths, w2v_lengths, w2v_feats_noisy.size(-1)) + self.w2v_embed(w2v_feats_noisy)
        x2 = self.encoder2(x1_d.transpose(1, 2), w2v_mask[:,:,None]).transpose(1, 2)
        m_p = self.mu_p(x2)
        
        z_p = m_p + noise_scale*torch.randn_like(m_p)
        x4 = x1 + self.resample(self.w2v_proj(z_p)*w2v_mask.unsqueeze(1), w2v_lengths, spec_lengths, spec_noisy.size(-1))
            
        x4 = self.proj_block(x4.transpose(1, 2), x_mask[:,:,None]).transpose(1, 2)
        mel_pred = self.proj_conv(x4)

        return mel_pred
    
    
    def get_mask_from_lengths(self, x_lengths, max_len):
        mask = self.indices[:max_len].unsqueeze(0) < x_lengths.unsqueeze(-1)
        return mask.detach()
    
    
    def resample(self, x, source_lengths, target_lengths, max_len):
        B, D, T = x.size()
        x_resampled = torch.zeros(B, D, max_len, device=x.device)
        for i in range(len(x)):
            if source_lengths[0] > target_lengths[0]:
                interpolated_sequence = F.interpolate(x[i:i+1, :, :source_lengths[i]], size=target_lengths[i], mode='area')
            elif source_lengths[0] < target_lengths[0]:
                interpolated_sequence = F.interpolate(x[i:i+1, :, :source_lengths[i]], size=target_lengths[i], mode='nearest-exact')
            x_resampled[i, :, :target_lengths[i]] = interpolated_sequence[0]
            
        x_mask = self.get_mask_from_lengths(target_lengths, max_len) # [B, T]
        x_resampled = x_resampled * x_mask.unsqueeze(1)
        return x_resampled
