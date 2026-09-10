"""Conditional flow matching for 28x28 Quick, Draw! drawings (COMP6010 Part C, option C2).

Method summary
--------------
Straight-line ("rectified"/conditional) flow matching. For a noise sample x0 ~ N(0, I) and a
data sample x1 ~ p_data, define the interpolation and its constant target velocity

    x_t = (1 - t) x0 + t x1,        u_t = dx_t/dt = x1 - x0,      t ~ U[0, 1]

A time- and class-conditional network v_theta(t, x_t, y) regresses that velocity:

    L(theta) = E_{t, x0, x1, y} || v_theta(t, x_t, y) - (x1 - x0) ||^2

Generation integrates the learned ODE dx/dt = v_theta(t, x, y) from t=0 (noise) to t=1 (image).

This extends the 2-D two-moons network from COMP6010 Practical 7 to images: the MLP is replaced
by a small time-conditioned U-Net, and a class embedding is added for label conditioning.
"""
from __future__ import annotations
import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


def enable_deterministic():
    """Make GPU sampling BITWISE reproducible for a given seed.

    Without this, cuDNN is free to pick non-deterministic convolution algorithms and TF32
    reductions, which makes two runs of the same seed differ by ~5e-5 per forward pass. Over 50
    ODE steps that accumulates to ~7e-4 -- visually irrelevant but enough that `torch.equal` on
    two same-seed runs returns False. Called at import time by the notebook so that the
    predetermined seed list reproduces samples exactly, not just to plotting precision.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


# --------------------------------------------------------------------- conditioning
class SinusoidalTimeEmbedding(nn.Module):
    """Fourier features of the scalar time t in [0,1] (as used by diffusion/FM U-Nets)."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:                       # t: (B,) or (B,1)
        t = t.view(-1).float()
        half = self.dim // 2
        freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device) / (half - 1))
        ang = t[:, None] * freqs[None, :] * 1000.0
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)  # (B, dim)


class ResBlock(nn.Module):
    """GroupNorm-SiLU-Conv residual block with additive time+class conditioning."""

    def __init__(self, c_in: int, c_out: int, c_emb: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(c_emb, c_out)
        self.norm2 = nn.GroupNorm(min(8, c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]           # broadcast conditioning
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class VelocityUNet(nn.Module):
    """Small time- and class-conditional U-Net predicting the velocity field v_theta(t, x, y).

    Input : x (B,1,28,28) in [-1,1], t (B,) in [0,1], y (B,) int64 class labels
    Output: velocity (B,1,28,28) -- same shape as x, since it lives in the data space.

    Resolutions: 28 -> 14 -> 7 (two downsamples), mirrored on the way up with skip
    connections. `n_classes + 1` embeddings are allocated so index `n_classes` can serve as a
    "null"/unconditional token if label dropout is enabled.
    """

    def __init__(self, n_classes: int = 3, base: int = 64, emb_dim: int = 128,
                 in_ch: int = 1):
        super().__init__()
        self.n_classes = n_classes
        self.time_embed = nn.Sequential(SinusoidalTimeEmbedding(emb_dim),
                                        nn.Linear(emb_dim, emb_dim), nn.SiLU(),
                                        nn.Linear(emb_dim, emb_dim))
        self.class_embed = nn.Embedding(n_classes + 1, emb_dim)    # +1 = null token
        c1, c2, c3 = base, base * 2, base * 2

        self.stem = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.down1 = ResBlock(c1, c1, emb_dim)                     # 28
        self.pool1 = nn.Conv2d(c1, c1, 3, stride=2, padding=1)     # 28 -> 14
        self.down2 = ResBlock(c1, c2, emb_dim)                     # 14
        self.pool2 = nn.Conv2d(c2, c2, 3, stride=2, padding=1)     # 14 -> 7

        self.mid1 = ResBlock(c2, c3, emb_dim)                      # 7
        self.mid2 = ResBlock(c3, c2, emb_dim)

        self.up2 = nn.ConvTranspose2d(c2, c2, 4, stride=2, padding=1)   # 7 -> 14
        self.dec2 = ResBlock(c2 + c2, c1, emb_dim)
        self.up1 = nn.ConvTranspose2d(c1, c1, 4, stride=2, padding=1)   # 14 -> 28
        self.dec1 = ResBlock(c1 + c1, c1, emb_dim)

        self.out_norm = nn.GroupNorm(min(8, c1), c1)
        self.out_conv = nn.Conv2d(c1, in_ch, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight); nn.init.zeros_(self.out_conv.bias)

    def forward(self, t: Tensor, x: Tensor, y: Tensor) -> Tensor:
        emb = self.time_embed(t) + self.class_embed(y)

        h0 = self.stem(x)
        h1 = self.down1(h0, emb)          # (B, c1, 28, 28)
        p1 = self.pool1(h1)               # (B, c1, 14, 14)
        h2 = self.down2(p1, emb)          # (B, c2, 14, 14)
        p2 = self.pool2(h2)               # (B, c2, 7, 7)

        m = self.mid2(self.mid1(p2, emb), emb)                     # (B, c2, 7, 7)

        u2 = self.up2(m)                                           # (B, c2, 14, 14)
        d2 = self.dec2(torch.cat([u2, h2], dim=1), emb)            # (B, c1, 14, 14)
        u1 = self.up1(d2)                                          # (B, c1, 28, 28)
        d1 = self.dec1(torch.cat([u1, h1], dim=1), emb)            # (B, c1, 28, 28)

        return self.out_conv(F.silu(self.out_norm(d1)))


# --------------------------------------------------------------------- training target
def flow_matching_batch(x1: Tensor, generator: torch.Generator | None = None):
    """Build one conditional-flow-matching training example set from clean images x1.

    Returns (x_t, t, target_velocity):
        x0     ~ N(0, I)                 same shape as x1
        t      ~ U[0,1]                  one scalar per image, broadcast over pixels
        x_t    = (1-t) x0 + t x1         straight-line interpolation
        target = x1 - x0                 constant velocity of that straight line
    """
    x0 = torch.randn(x1.shape, device=x1.device, dtype=x1.dtype, generator=generator)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype, generator=generator)
    t_b = t.view(-1, 1, 1, 1)
    x_t = (1.0 - t_b) * x0 + t_b * x1
    return x_t, t, (x1 - x0)


# --------------------------------------------------------------------- ODE samplers
@torch.no_grad()
def sample_ode(model: nn.Module, n: int, labels: Tensor, n_steps: int = 50,
               method: str = "euler", device: str = "cuda",
               seed: int | None = None, img_shape=(1, 28, 28),
               return_trajectory: bool = False, traj_every: int | None = None):
    """Integrate dx/dt = v_theta(t, x, y) from t=0 (noise) to t=1 (image).

    method : "euler"    -> 1 network evaluation per step  (NFE = n_steps)
             "midpoint" -> 2 network evaluations per step (NFE = 2 * n_steps)

    Randomness comes ONLY from the initial noise x0 (seeded here); both solvers are
    deterministic given x0, which is why a fixed seed list reproduces the same images.
    Returns (images, nfe) or (images, nfe, trajectory).
    """
    model.eval()
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    x = torch.randn((n, *img_shape), device=device, generator=gen)

    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    traj, nfe = [], 0
    traj_every = traj_every or max(1, n_steps // 8)

    for i in range(n_steps):
        t0, t1 = ts[i], ts[i + 1]
        dt = t1 - t0
        tb = t0.expand(n)
        if method == "euler":
            v = model(tb, x, labels);                       nfe += 1
            x = x + dt * v
        elif method == "midpoint":
            v1 = model(tb, x, labels);                      nfe += 1
            x_mid = x + 0.5 * dt * v1
            v2 = model((t0 + 0.5 * dt).expand(n), x_mid, labels);  nfe += 1
            x = x + dt * v2
        else:
            raise ValueError(f"unknown method {method!r}")
        if return_trajectory and (i % traj_every == 0 or i == n_steps - 1):
            traj.append(x.detach().cpu().clone())

    if return_trajectory:
        return x.detach(), nfe, traj
    return x.detach(), nfe


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
