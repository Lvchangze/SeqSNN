import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional
from functools import partial
import collections.abc

from typing import Optional

from pathlib import Path
from spikingjelly.activation_based import surrogate, neuron, functional

from ..base import NETWORKS
import math

__all__ = ['QKFormer_Log']

import torch

tau = 2.0  # beta = 1 - 1/tau
detach_reset = True

def create_symmetric_matrix(L):
    diag_value = math.ceil(math.log2(L))
    matrix = torch.zeros((L, L), dtype=torch.int)
    for i in range(L):
        matrix[i, i] = diag_value
        if i < L - 1:
            values = torch.linspace(diag_value - 1, 0, steps=L - i - 1).round().int()
            matrix[i, i+1:] = values
            matrix[i+1:, i] = values
    return matrix


class ConvPE(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000, num_steps=4):

        super().__init__()
        self.T = num_steps
        self.rpe_conv = nn.Conv1d(
            d_model, d_model, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.rpe_bn = nn.BatchNorm1d(d_model)
        self.rpe_lif = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        T, B, L, _ = x.shape  # x: T, B, L, D
        x = x.flatten(0, 1)  # TB, L, D
        x = x.transpose(0, 1) # L, TB, D
        # x: L, TB, D
        L, TB, D = x.shape
        x_feat = x.permute(1, 2, 0)  # TB, D, L
        # print(x_feat.shape)
        x_feat = self.rpe_conv(x_feat)  # TB, D, L
        x_feat = (
            self.rpe_bn(x_feat).reshape(self.T, int(TB / self.T), D, L).contiguous()
        )  # T, B, D, L
        x_feat = self.rpe_lif(x_feat)
        x_feat = x_feat.flatten(0, 1)  # TB, D, L
        x_feat = self.dropout(x_feat)  # TB, D, L
        x_feat = x_feat.permute(2, 0, 1)  # L, TB, D
        x = x + x_feat # L, TB, D
        x = x.transpose(0, 1) # TB, L, D
        x = x.reshape(T, B, L, -1) # T, B, L, D
        return x

class ConvEncoder(nn.Module):
    def __init__(self, output_size: int, kernel_size: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=output_size,
                kernel_size=(1, kernel_size),
                stride=1,
                padding=(0, kernel_size // 2),
            ),
            nn.BatchNorm2d(output_size),
        )
        self.lif = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())
        
    def forward(self, inputs: torch.Tensor):
        # inputs: B, L, D
        inputs = inputs.permute(0, 2, 1).unsqueeze(1) # B, 1, D, L
        enc = self.encoder(inputs) # B, T, D, L
        enc = enc.permute(1, 0, 2, 3)  # T, B, D, L
        spks = self.lif(enc) # T, B, D, L
        spks = spks 
        return spks

class Token_QK_Attention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.q_linear = nn.Linear(dim, dim, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.k_linear = nn.Linear(dim, dim, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.attn_lif = neuron.LIFNode(tau = tau, v_threshold=0.5, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.proj_linear = nn.Linear(dim, dim, bias=False)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

    def forward(self, x):
        T, B, L, D = x.shape
        x_for_qkv = x.flatten(0, 1) # TB, L, D

        q_conv_out = self.q_linear(x_for_qkv) # TB, L, D
        q_conv_out = self.q_bn(q_conv_out.transpose(-1, -2)).reshape(T, B, D, L) # T, B, D, L
        q_conv_out = self.q_lif(q_conv_out) # T, B, D, L
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, D // self.num_heads, L) # T, B, head, D//head, L

        k_conv_out = self.k_linear(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out.transpose(-1, -2)).reshape(T, B, D, L) # T, B, D, L
        k_conv_out = self.k_lif(k_conv_out) # T, B, D, L
        k = k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, D // self.num_heads, L) # T, B, head, D//head, L

        q = torch.sum(q, dim = 3, keepdim = True) # T, B, head, 1, L
        attn = self.attn_lif(q) # T, B, head, 1, L
        x = torch.mul(attn, k) # 对应位置相乘，广播 -> T, B, head, D//head, L

        x = x.flatten(2, 3).flatten(0, 1) # TB, D, L
        x = x.transpose(-2, -1) # TB, L, D
        x = self.proj_bn(self.proj_linear(x).transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(T, B, L, D)
        x = self.proj_lif(x)
        return x

class Spiking_Self_Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} should be divided by num_heads {heads}."

        self.dim = dim
        self.heads = heads
        # self.qk_scale = qk_scale
        self.scale = nn.Parameter(data=torch.tensor(-4.0), requires_grad=True)

        self.q_m = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.k_m = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.v_m = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.attn_lif = neuron.LIFNode(tau=tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan(), v_threshold=0.5)

        self.last_m = nn.Linear(dim, dim)
        self.last_bn = nn.BatchNorm1d(dim)
        self.last_lif = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

    def forward(self, x):
        # x = x.transpose(0, 1)

        T, B, L, D = x.shape
        x_for_qkv = x.flatten(0, 1) # TB L D
        q_m_out = self.q_m(x_for_qkv) # TB L D
        q_m_out = self.q_bn(q_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        q_m_out = self.q_lif(q_m_out)
        q = q_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        k_m_out = self.k_m(x_for_qkv)
        k_m_out = self.k_bn(k_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        k_m_out = self.k_lif(k_m_out)
        k = k_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        v_m_out = self.v_m(x_for_qkv)
        v_m_out = self.v_bn(v_m_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        v_m_out = self.v_lif(v_m_out)
        v = v_m_out.reshape(T, B, L, self.heads, D // self.heads).permute(0, 1, 3, 2, 4).contiguous()

        # attn = (q @ k.transpose(-2, -1)) * self.qk_scale

        tmp = create_symmetric_matrix(L).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # original
        # attn = torch.sum(1 - (q.unsqueeze(3) - k.unsqueeze(4)) ** 2, dim=-1)
        
        # To save memory
        D_new = q.size(-1)  # Get feature dimension C // num_heads
        sum_q = q.sum(dim=-1)  # Calculate the sum of each query vector [T, B, D, L]
        sum_k = k.sum(dim=-1)  # Calculate the sum of each key vector [T, B, D, L]
        # Matrix multiplication to calculate the dot product of q and k [T, B, D, L, L]
        qk_matmul = torch.matmul(q, k.transpose(-1, -2))
        # Calculate the sum of sum_q and sum_k via broadcasting
        sum_q = sum_q.unsqueeze(-1)  # [T, B, D, L, 1]
        sum_k = sum_k.unsqueeze(-2)  # [T, B, D, 1, L]
        # Final attention calculation (mathematically equivalent form)
        attn = (D_new - sum_q - sum_k + 2 * qk_matmul)  # [T, B, H, L, L]

        attn = (attn + tmp.cuda() ) * torch.sigmoid(self.scale)

        x = attn @ v  # x_shape: T * B * heads * L * D//heads

        x = x.transpose(2, 3).reshape(T, B, L, D).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.last_m(x)
        x = self.last_bn(x.transpose(-1, -2)).transpose(-1, -2)
        x = self.last_lif(x.reshape(T, B, L, D).contiguous())
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.lif1 = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.lif2 = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

    def forward(self, x):
        T, B, L, D = x.shape
        x = x.flatten(0, 1) # TB L D
        x = self.fc1(x) # TB L H
        x = self.bn1(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, self.hidden_features).contiguous()
        x = self.lif1(x)
        x = x.flatten(0, 1) # TB L H
        x = self.fc2(x) # TB L D
        x = self.bn2(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, L, D).contiguous()
        x = self.lif2(x)
        return x

class TokenSpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.tssa = Token_QK_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.tssa(x)
        x = x + self.mlp(x)
        return x

class SpikingTransformer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.ssa = Spiking_Self_Attention(dim, num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features= dim, hidden_features=mlp_hidden_dim)

    def forward(self, x):
        x = x + self.ssa(x)
        x = x + self.mlp(x)
        return x

@NETWORKS.register_module("QKFormer_XNOR_Log")
class QKFormer_XNOR_Log(nn.Module):
    def __init__(
            self,
            dim: int,
            d_ff: Optional[int] = None,
            num_pe_neuron: int = 10,
            pe_type: str = "conv",
            pe_mode: str = "concat",  # "add" or concat
            neuron_pe_scale: float = 1000.0,  # "100" or "1000" or "10000"
            depths: int = 4,
            common_thr: float = 1.0,
            max_length: int = 5000,
            num_steps: int = 4,
            heads: int = 8,
            qkv_bias: bool = False,
            qk_scale: float = 0.125,
            input_size: Optional[int] = None,
            weight_file: Optional[Path] = None):
        super().__init__()
        self.depths = depths
        self.dim = dim
        self.pe_type = pe_type
        self.temporal_encoder = ConvEncoder(num_steps)
        self.pe = ConvPE(d_model=input_size)
        
        self.encoder = nn.Linear(input_size, dim)
        self.init_bn = nn.BatchNorm1d(dim)
        self.init_lif = neuron.LIFNode(tau = tau, detach_reset=detach_reset, surrogate_function=surrogate.ATan())

        self.stage1 = nn.ModuleList([TokenSpikingTransformer(
            dim=dim, num_heads=heads, mlp_ratio=4.)
            for j in range(1)])

        self.stage2 = nn.ModuleList([TokenSpikingTransformer(
            dim=dim, num_heads=heads, mlp_ratio=4.)
            for j in range(1)])

        self.stage3 = nn.ModuleList([SpikingTransformer(
            dim=dim, num_heads=heads, mlp_ratio=4.)
            for j in range(depths - 2)])
        
        self.apply(self._init_weights)
        functional.set_step_mode(self, "m")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        functional.reset_net(self)

        x = self.temporal_encoder(x) # B L C -> T B C L
        x = x.transpose(-2, -1) # T B L C
        if self.pe_type != "none":
            # print(x.shape)
            x = self.pe(x) # T B L C
        T, B, L, C = x.shape

        x = self.encoder(x.flatten(0, 1)) # TB L C -> # T B L D
        x = self.init_bn(x.transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(T, B, L, -1) # T B L D

        for blk in self.stage1:
            x = blk(x)

        for blk in self.stage2:
            x = blk(x)

        for blk in self.stage3:
            x = blk(x)
        # T B L D
        out = x.mean(0) # B L D
        return out, out.mean(dim=1) # B L D, B D
    
    @property
    def output_size(self):
        return self.dim
    
    @property
    def hidden_size(self):
        return self.dim

