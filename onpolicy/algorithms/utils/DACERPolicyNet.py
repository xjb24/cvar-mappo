import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import init
import math

class DACERPolicyNet(nn.Module):
    def __init__(self, args, action_dim, t_dim=16, device=torch.device("cpu")):
        super(DACERPolicyNet, self).__init__()
        self.device = device
        self.action_dim = action_dim
        self.obs_feature_dim = args.hidden_size
        self.t_dim = t_dim
        self.inputs_dim = self.action_dim + self.obs_feature_dim + self.t_dim
        self.hidden_size = args.hidden_size
        self.use_orthogonal = args.use_orthogonal
        self.gain = args.gain
        self.te_net = nn.Sequential(
            nn.Linear(self.t_dim, self.t_dim * 2),
            nn.Mish(),
            nn.Linear(self.t_dim * 2, self.t_dim),
            nn.Identity(),
        )
        self.mlp = PolicyMLP(self.inputs_dim, self.action_dim, self.hidden_size, self.use_orthogonal, self.gain).to(self.device)
        self.to(self.device)

    def forward(self, obs, action, t):
        obs = obs.to(self.device)
        action = action.to(self.device)
        t = t.to(self.device)
        te = scaled_sinusoidal_encoding(t, dim=self.t_dim, batch_shape=obs.shape[:-1])
        te = te.to(self.device)
        te = self.te_net(te)
        inputs = torch.cat([obs, action, te], dim=-1)
        inputs = inputs.to(self.device)
        noises = self.mlp(inputs)
        return noises

class PolicyMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, use_orthogonal, gain):
        super(PolicyMLP, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.network = nn.Sequential(
            init_(nn.Linear(input_dim, hidden_size)), 
            nn.Mish(), 
            init_(nn.Linear(hidden_size, hidden_size)), 
            nn.Mish(), 
            init_(nn.Linear(hidden_size, hidden_size)), 
            nn.Mish(),
            init_(nn.Linear(hidden_size, output_dim)), 
            nn.Identity()  # Output layer, no activation
            )

    def forward(self, x):
        return self.network(x)

class Mish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))

def scaled_sinusoidal_encoding(t: torch.Tensor, *, dim: int, theta: int = 10000, batch_shape = None) -> torch.Tensor:
    assert dim % 2 == 0, "dim 必须是偶数"
    if batch_shape is not None:
        # 检查是否可广播
        assert all(ts == bs or ts == 1 or bs == 1 
                  for ts, bs in zip(t.shape, batch_shape)), "形状不可广播"
    
    scale = 1 / math.sqrt(dim)
    half_dim = dim // 2
    
    # 生成频率序列
    freq_seq = torch.arange(half_dim, device=t.device) / half_dim  # [half_dim]
    inv_freq = theta ** -freq_seq  # [half_dim]
    
    # 计算编码
    emb = t.unsqueeze(-1) * inv_freq  # [..., half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [..., dim]
    emb = emb * scale
    
    # 显式广播（如果需要）
    if batch_shape is not None:
        emb = emb.expand(*batch_shape, dim)
    
    return emb