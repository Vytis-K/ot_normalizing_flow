import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        half_dim = self.dim // 2
        device = t.device
        freq_scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -freq_scale)
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class VelocityMLP(nn.Module):
    """
    Velocity field v_theta(x_t, t) for 2D flow matching.
    Input: x_t in R^2 and scalar time t in [0, 1]
    Output: velocity vector in R^2
    """
    def __init__(self, data_dim: int = 2, hidden_dim: int = 256, time_dim: int = 64):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(dim=time_dim)
        self.net = nn.Sequential(
            nn.Linear(data_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        t_emb = self.time_embed(t)
        features = torch.cat([x, t_emb], dim=1)
        return self.net(features)


__all__ = ["VelocityMLP", "SinusoidalTimeEmbedding"]
