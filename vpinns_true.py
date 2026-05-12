from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

import config as cfg
from pinns import LWRPINN


def flux(u: torch.Tensor) -> torch.Tensor:
    return u * (1.0 - u)


@dataclass
class TrainHistory:
    loss: list
    loss_var: list
    loss_ic: list
    loss_bc: list


class VPINN(nn.Module):
    """
    Weak-form VPINN for the conservation law

        u_t + (f(u))_x = 0,   f(u)=u(1-u)

    using global sinusoidal test functions on the normalized
    space-time domain.

    The weak form used is

        ∫∫ [ -u * phi_t - f(u) * phi_x ] dx dt

    which comes from integration by parts.
    """

    def __init__(
        self,
        layers: List[int],
        activation: Union[str, nn.Module] = "tanh",
        hard_init: bool = True,
        n_fourier: int = 6,
        scale: float = 2.0,
        x_min: float = cfg.X_MIN,
        x_max: float = cfg.X_MAX,
        t_min: float = cfg.T_MIN,
        t_max: float = cfg.T_MAX,
        n_test: int = 2,
    ) -> None:
        super().__init__()

        self.model = LWRPINN(
            layers=layers,
            activation=activation,
            hard_init=hard_init,
            n_fourier=n_fourier,
            scale=scale,
        )

        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.t_min = float(t_min)
        self.t_max = float(t_max)
        self.n_test = int(n_test)

        self.domain_area = (self.x_max - self.x_min) * (self.t_max - self.t_min)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        u0_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        return self.model(x, t, u0_fn=u0_fn)

    def weak_residual_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        u0_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Weak residual loss using sinusoidal test functions:
            phi_mn(x,t) = sin(m*pi*xi) sin(n*pi*tau)

        where xi,tau are normalized coordinates in [0,1].
        """
        if self.n_test < 1:
            raise ValueError("n_test must be >= 1")

        u = self.forward(x, t, u0_fn=u0_fn)
        fu = flux(u)

        xi = (x - self.x_min) / (self.x_max - self.x_min + 1e-12)
        tau = (t - self.t_min) / (self.t_max - self.t_min + 1e-12)

        loss = torch.zeros((), device=x.device)

        for m in range(1, self.n_test + 1):
            for n in range(1, self.n_test + 1):
                phi_x = (
                    (m * math.pi / (self.x_max - self.x_min + 1e-12))
                    * torch.cos(m * math.pi * xi)
                    * torch.sin(n * math.pi * tau)
                )

                phi_t = (
                    (n * math.pi / (self.t_max - self.t_min + 1e-12))
                    * torch.sin(m * math.pi * xi)
                    * torch.cos(n * math.pi * tau)
                )

                integrand = -u * phi_t - fu * phi_x

                # Monte Carlo quadrature over full space-time domain
                residual = self.domain_area * torch.mean(integrand)

                loss = loss + residual.pow(2)

        return loss / float(self.n_test * self.n_test)


def _sample_x(n: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n, 1, device=device) * (cfg.X_MAX - cfg.X_MIN) + cfg.X_MIN


def _sample_t(n: int, device: torch.device, bias_early: bool = True) -> torch.Tensor:
    if bias_early:
        z = torch.rand(n, 1, device=device)
        return cfg.T_MIN + (cfg.T_MAX - cfg.T_MIN) * (z ** 2)
    return torch.rand(n, 1, device=device) * (cfg.T_MAX - cfg.T_MIN) + cfg.T_MIN


def _sample_initial(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = _sample_x(n, device)
    t = torch.full_like(x, cfg.T_MIN)
    return x, t


def _dirichlet_bc_loss(
    model: VPINN,
    u0_fn: Callable,
    uL: float,
    uR: float,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    t = _sample_t(n, device, bias_early=False)
    xL = torch.full_like(t, cfg.X_MIN)
    xR = torch.full_like(t, cfg.X_MAX)

    predL = model(xL, t, u0_fn=u0_fn)
    predR = model(xR, t, u0_fn=u0_fn)

    targetL = torch.full_like(predL, float(uL))
    targetR = torch.full_like(predR, float(uR))

    return torch.mean((predL - targetL) ** 2) + torch.mean((predR - targetR) ** 2)


def train_vpinn_onejump(
    layers: List[int],
    u0_fn: Callable,
    uL: float,
    uR: float,
    steps: int = cfg.STEPS,
    lr: float = cfg.LR,
    n_var: int = 4096,
    n_ic: int = 4096,
    n_bc: int = 1024,
    w_var: float = 1.0,
    w_ic: float = 50.0,
    w_bc: float = 10.0,
    activation: Union[str, nn.Module] = "tanh",
    hard_init: bool = True,
    n_fourier: int = 6,
    scale: float = 2.0,
    n_test: int = 2,
    device: Optional[torch.device] = None,
    log_every: int = cfg.LOG_EVERY,
) -> Tuple[VPINN, TrainHistory]:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VPINN(
        layers=layers,
        activation=activation,
        hard_init=hard_init,
        n_fourier=n_fourier,
        scale=scale,
        x_min=cfg.X_MIN,
        x_max=cfg.X_MAX,
        t_min=cfg.T_MIN,
        t_max=cfg.T_MAX,
        n_test=n_test,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)

    hist = TrainHistory(loss=[], loss_var=[], loss_ic=[], loss_bc=[])

    for it in range(1, steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        # variational samples
        xv = _sample_x(n_var, device)
        tv = _sample_t(n_var, device, bias_early=True)
        loss_var = model.weak_residual_loss(xv, tv, u0_fn=u0_fn)

        # initial condition
        x0, t0 = _sample_initial(n_ic, device)
        u0_true = u0_fn(x0)
        u0_pred = model(x0, t0, u0_fn=u0_fn)
        loss_ic = torch.mean((u0_pred - u0_true) ** 2)

        # Dirichlet boundary
        loss_bc = _dirichlet_bc_loss(
            model=model,
            u0_fn=u0_fn,
            uL=uL,
            uR=uR,
            n=n_bc,
            device=device,
        )

        loss = w_var * loss_var + w_ic * loss_ic + w_bc * loss_bc
        loss.backward()
        opt.step()

        if it % log_every == 0 or it == 1:
            lv = float(loss_var.detach().cpu())
            li = float(loss_ic.detach().cpu())
            lb = float(loss_bc.detach().cpu())
            lt = float(loss.detach().cpu())

            print(
                f"[{it:6d}/{steps}] "
                f"loss={lt:.3e} | var={lv:.3e} ic={li:.3e} bc={lb:.3e}"
            )

            hist.loss.append(lt)
            hist.loss_var.append(lv)
            hist.loss_ic.append(li)
            hist.loss_bc.append(lb)

    return model, hist