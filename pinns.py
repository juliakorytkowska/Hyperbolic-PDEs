from __future__ import annotations

from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn

import config as cfg


def _make_activation(activation: Union[str, nn.Module]) -> nn.Module:
    if isinstance(activation, nn.Module):
        return activation.__class__()
    key = str(activation).lower()
    if key == "tanh":
        return nn.Tanh()
    if key == "relu":
        return nn.ReLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError("activation must be one of: tanh, relu, gelu (or nn.Module)")

class LWRPINN(nn.Module):
    def __init__(
        self,
        layers: List[int],
        activation: Union[str, nn.Module] = "tanh",
        hard_init: bool = True,
        n_fourier: int = 6,     # number of frequencies
        scale: float = 2.0,     # frequency growth
    ) -> None:
        super().__init__()
        self.hard_init = bool(hard_init)
        self.n_fourier = int(n_fourier)
        self.scale = float(scale)

        # input dim: (x,t) plus 2*sin/cos for each freq for x and t
        in_dim = 2 + 2 * self.n_fourier * 2

        blocks: list[nn.Module] = []
        dim = in_dim
        for w in layers:
            blocks.append(nn.Linear(dim, w))
            blocks.append(_make_activation(activation))
            dim = w
        blocks.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*blocks)

    def _embed(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feats = [x, t]
        for k in range(self.n_fourier):
            w = (self.scale ** k) * torch.pi
            feats += [torch.sin(w * x), torch.cos(w * x), torch.sin(w * t), torch.cos(w * t)]
        return torch.cat(feats, dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, u0_fn=None) -> torch.Tensor:
        z = self._embed(x, t)
        u_nn = torch.sigmoid(self.net(z))  # keep in (0,1)

        if self.hard_init:
            if u0_fn is None:
                raise ValueError("hard_init=True requires u0_fn")
            u0 = u0_fn(x)

            tau = 0.15
            alpha = 1.0 - torch.exp(-(t - cfg.T_MIN) / tau)
            return (1.0 - alpha) * u0 + alpha * u_nn

        return u_nn


def fv_residual_midpoint(
    model: LWRPINN,
    x_center: torch.Tensor,
    t: torch.Tensor,
    dx: float,
    u0_fn=None,
) -> torch.Tensor:
    """
    FV-style residual:
        r = u_t(x_j,t) + ( f(u(x_{j+1/2},t)) - f(u(x_{j-1/2},t)) ) / dx

    This avoids computing u_x directly (better behaved near shocks).
    """
    x_center = x_center.requires_grad_(True)
    t = t.requires_grad_(True)

    u_c = model(x_center, t, u0_fn=u0_fn)
    u_t = torch.autograd.grad(u_c, t, torch.ones_like(u_c), create_graph=True)[0]

    xR = x_center + 0.5 * dx
    xL = x_center - 0.5 * dx

    # Keep evaluation points inside domain (important near boundaries)
    xR = torch.clamp(xR, cfg.X_MIN, cfg.X_MAX)
    xL = torch.clamp(xL, cfg.X_MIN, cfg.X_MAX)

    uR = model(xR, t, u0_fn=u0_fn)
    uL = model(xL, t, u0_fn=u0_fn)

    fR = uR * (1.0 - uR)
    fL = uL * (1.0 - uL)

    flux_div = (fR - fL) / dx
    return u_t + flux_div