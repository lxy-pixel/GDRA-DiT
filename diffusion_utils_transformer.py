import torch
import torch.nn as nn
import numpy as np
import math
import sys
from dit_models import DiT

# 这里的 BETA 数组保持与你之前的一致
BETA = np.array([-3.01,-2.61,-2.3,-2.06,-1.86,-1.69,-1.55,-1.43,-1.33,-1.24,-1.16,-1.09,-1.02,-0.962,-0.909,-0.86,-0.816,-0.774,-0.736,-0.701,-0.668,-0.637,-0.608,-0.58,-0.554,-0.53,-0.507,-0.484,-0.463,-0.443,-0.424,-0.405,-0.388,-0.371,-0.354,-0.338,-0.323,-0.308,-0.293,-0.279,-0.266,-0.252,-0.239,-0.227,-0.214,-0.202,-0.19,-0.179,-0.167,-0.156,-0.145,-0.134,-0.123,-0.112,-0.102,-0.0912,-0.0809,-0.0706,-0.0604,-0.0502,-0.0401,-0.03,-0.02,-0.01,0.,0.01,0.02,0.03,0.0401,0.0502,0.0604,0.0706,0.0809,0.0912,0.102,0.112,0.123,0.134,0.145,0.156,0.167,0.179,0.19,0.202,0.214,0.227,0.239,0.252,0.266,0.279,0.293,0.308,0.323,0.338,0.354,0.371,0.388,0.405,0.424,0.443,0.463,0.484,0.507,0.53,0.554,0.58,0.608,0.637,0.668,0.701,0.736,0.774,0.816,0.86,0.909,0.962,1.02,1.09,1.16,1.24,1.33,1.43,1.55,1.69,1.86,2.06,2.3,2.61], dtype=np.float32)
BETA_T = torch.tensor(BETA, dtype=torch.float32)

def lambda_from_u(u):
    b = torch.atan(torch.exp(torch.tensor(-10.0, device=u.device)))
    a = torch.atan(torch.exp(torch.tensor(10.0, device=u.device))) - b
    return -2.0 * torch.log(torch.tan(a * u + b))

def alpha_from_lambda(l): return torch.sqrt(torch.sigmoid(l))
def sigma_from_lambda(l): return torch.sqrt(torch.clamp(1.0 - torch.sigmoid(l), min=0.0))

def lambda_encoding(lmbd):
    lmbd = lmbd.view(-1, 1)
    angles = 2.0 * math.pi * lmbd * BETA_T.to(lmbd.device)
    return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

class DiT_Wrapper(nn.Module):
    """包装器，将输入格式适配给 DiT"""
    def __init__(self, dit_model):
        super().__init__()
        self.model = dit_model
        self.t_embedder = nn.Sequential(
            nn.Linear(2 * len(BETA), 512),
            nn.SiLU(),
            nn.Linear(512, 512),
        )

    def forward(self, x, lmbd_enc):
        t_cond = self.t_embedder(lmbd_enc)
        return self.model(x, t_cond)

# === diffusion_utils_transformer.py ===
def build_model(device):
    # 🚀 修改点：in_channels 从 3 改成 5 
    # (1 个加噪雪深 + 4 个条件: 上一刻雪深, DEM, tp_norm, t2m_norm)
    base_dit = DiT(input_size=120, patch_size=4, in_channels=5, hidden_size=512, depth=12, num_heads=8)
    return DiT_Wrapper(base_dit).to(device)
