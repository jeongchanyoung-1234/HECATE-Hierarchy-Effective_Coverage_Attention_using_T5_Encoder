import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForSeq2SeqLM


class Attention(nn.Module):
    def __init__(self, config, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.coverage = config.coverage
        self.coverage_version = config.coverage_version
        self.W_h = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.W_s = nn.Linear(self.hidden_dim, self.hidden_dim)
        if self.coverage:
            self.W_c = nn.Linear(1, self.hidden_dim, bias=False)
            self.W_v = nn.Linear(self.hidden_dim, 1)
        self.W_y = nn.Linear(self.hidden_dim, self.hidden_dim)

    def init_weights(self):
        for weight in self.parameters():
            weight.data.normal_(mean=0.0, std=self.hidden_dim ** -0.5)

    def forward(self, h_t_dec, h_enc, z_enc=None, mask=None, return_logits=False):
        B, L_ENC, H = list(h_enc.size())

        mask = mask.transpose(0, 1).contiguous().unsqueeze(-1)
        h_enc = h_enc.permute(1, 0, 2).contiguous() # L, B, H
        h_enc_2d = h_enc.view(-1, H)
        phi_h_enc_2d = torch.tanh(self.W_h(h_enc_2d)) # L * B, H
        phi_h_enc = phi_h_enc_2d.view(L_ENC, B, H)# L, B, H

        gamma_h = torch.tanh(self.W_s(h_t_dec)) # B, H
        att_features = phi_h_enc * gamma_h # L, B, H
        if z_enc is not None:
            if self.coverage_version >= 2:
                # |z_enc| = |coverage| = (B, L)
                z_enc = z_enc.transpose(0, 1).contiguous().unsqueeze(-1) # L, B, 1
                phi_z_enc_2d = z_enc.view(-1, 1)
                phi_z_enc_2d = torch.tanh(self.W_c(phi_z_enc_2d))
                phi_z_enc = phi_z_enc_2d.view(L_ENC, B, H)

                if self.coverage_version == 3:
                    eta_h_enc_2d = torch.tanh(self.W_y(h_enc_2d))
                    eta_h_enc = eta_h_enc_2d.view(L_ENC, B, H)
                    phi_z_enc = phi_z_enc * eta_h_enc

                att_features = att_features + phi_z_enc

        weights = torch.sum(att_features, dim=2, keepdim=True)  # L, B, 1
        weights = torch.exp(weights - torch.max(weights, dim=0, keepdim=True).values)
        weights = torch.div(weights, (1e-6 + torch.sum(weights, dim=0, keepdim=True))) # L, B, 1

        if z_enc is not None and self.coverage_version > 3:
            z_enc = z_enc.permute(1, 0, 2).contiguous() # L, B, H

            z_enc_2d = z_enc.view(-1, z_enc.size(-1))
            phi_z_enc_2d = torch.tanh(self.W_x(z_enc_2d)) # L * B, H
            phi_z_enc = phi_z_enc_2d.view(L_ENC, B, H) # L, B, H

            alpha_h = torch.tanh(self.W_y(h_t_dec)) # B, H

            fd_weights = torch.sum(phi_z_enc * alpha_h, dim=2, keepdim=True) # L, B, 1
            fd_weights = torch.exp(fd_weights - torch.max(fd_weights, dim=0, keepdim=True).values)
            fd_weights = torch.div(fd_weights, (1e-6 + torch.sum(fd_weights, dim=0, keepdim=True))) # L, B, 1
            
            weights = weights * fd_weights * (1 - mask)
            weights = torch.div(weights, (1e-6 + torch.sum(weights, dim=0, keepdim=True)))

        c_t = torch.sum(h_enc * weights, dim=0) # B, H

        weights = weights.squeeze(-1) # L, B
        weights = weights.transpose(0, 1).contiguous() # B, L
        
        ret = (c_t, weights, att_features.sum(dim=-1).transpose(0, 1).contiguous()) if return_logits else (c_t, weights) 
        return ret
    
class P_GEN(nn.Module):
    def __init__(self, hidden_dim):
        super(P_GEN, self).__init__()
        self.hidden_dim = hidden_dim
        self.w_c = nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.w_h = nn.Linear(in_features=self.hidden_dim, out_features=1, bias=False)
        self.w_x = nn.Linear(in_features=self.hidden_dim, out_features=1, bias=False)

    def init_weights(self):
        for weight in self.parameters():
            weight.data.normal_(mean=0.0, std=self.hidden_dim ** -0.5)

    def forward(self, context_vector, h_t_dec, x_t_dec):
        p_gen = self.w_c(context_vector)
        p_gen += self.w_h(h_t_dec)
        p_gen += self.w_x(x_t_dec)
        p_gen = torch.sigmoid(p_gen)
        return p_gen

class Model(object):
    def __init__(self, config):
        #self.use_copy = config.use_copy 
        self.use_copy = True
        model_type = AutoModelForSeq2SeqLM
        self.generator = model_type.from_pretrained(config.huggingface_model_name, output_hidden_states=True)
        if self.use_copy:
            self.attention = Attention(config, self.generator.config.d_model)
            self.p_gen = P_GEN(self.generator.config.d_model)
    
    def init(self):
        if self.use_copy:
            self.attention.init_weights()
            self.p_gen.init_weights()

    def run_train(self):
        self.generator = self.generator.train()
        if self.use_copy:
            self.attention = self.attention.train()
            self.p_gen = self.p_gen.train()

    def run_eval(self):
        self.generator = self.generator.eval()
        if self.use_copy:
            self.attention = self.attention.eval()
            self.p_gen = self.p_gen.eval()

    def on_device(self, device):
        self.generator = self.generator.to(device)
        if self.use_copy:
            self.attention = self.attention.to(device)
            self.p_gen = self.p_gen.to(device)