from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import config as cfg
from pinns import LWRPINN
from pinns import fv_residual_midpoint


@dataclass
class TrainHistory:
    loss: list
    loss_fv: list
    loss_ic: list
    loss_bc: list
    loss_sup: list
    loss_time: list


def _sample_x(n: int, device: torch.device) -> torch.Tensor:
    return torch.rand(n, 1, device=device) * (cfg.X_MAX - cfg.X_MIN) + cfg.X_MIN


def _sample_t(n: int, device: torch.device, bias_early: bool = True) -> torch.Tensor:
    if bias_early:
        xi = torch.rand(n, 1, device=device)
        return cfg.T_MIN + (cfg.T_MAX - cfg.T_MIN) * (xi ** 2)
    return torch.rand(n, 1, device=device) * (cfg.T_MAX - cfg.T_MIN) + cfg.T_MIN


def _sample_initial(n: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    x = _sample_x(n, device)
    t = torch.full_like(x, cfg.T_MIN)
    return x, t


def _periodic_bc_loss(model: LWRPINN, u0_fn, n: int, device: torch.device) -> torch.Tensor:
    t = _sample_t(n, device, bias_early=False)
    xL = torch.full_like(t, cfg.X_MIN)
    xR = torch.full_like(t, cfg.X_MAX)
    uL = model(xL, t, u0_fn=u0_fn)
    uR = model(xR, t, u0_fn=u0_fn)
    return torch.mean((uL - uR) ** 2)


def _interp_truth(
    u_truth: np.ndarray,
    x_grid: np.ndarray,
    t_grid: np.ndarray,
    xq: np.ndarray,
    tq: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation on a tensor-product grid.
    u_truth shape: (Nt, Nx)
    """
    Nx = len(x_grid)
    Nt = len(t_grid)

    ix = np.searchsorted(x_grid, xq) - 1
    it = np.searchsorted(t_grid, tq) - 1
    ix = np.clip(ix, 0, Nx - 2)
    it = np.clip(it, 0, Nt - 2)

    x0 = x_grid[ix]
    x1 = x_grid[ix + 1]
    t0 = t_grid[it]
    t1 = t_grid[it + 1]

    wx = (xq - x0) / (x1 - x0 + 1e-12)
    wt = (tq - t0) / (t1 - t0 + 1e-12)

    u00 = u_truth[it, ix]
    u01 = u_truth[it, ix + 1]
    u10 = u_truth[it + 1, ix]
    u11 = u_truth[it + 1, ix + 1]

    u0i = (1 - wx) * u00 + wx * u01
    u1i = (1 - wx) * u10 + wx * u11
    return (1 - wt) * u0i + wt * u1i


def _time_consistency_loss(
    model: LWRPINN,
    u0_fn,
    n: int,
    dt: float,
    dx_fd: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Enforce one explicit-Euler step of the PDE:
      u(x,t+dt) ≈ u(x,t) - dt * (f(u))_x
    using finite differences for (f(u))_x.
    Prevents collapse to time-constant solutions (vertical stripes).
    """
    x = _sample_x(n, device)
    t = _sample_t(n, device, bias_early=True)

    t2 = torch.clamp(t + dt, cfg.T_MIN, cfg.T_MAX)

    u_now = model(x, t, u0_fn=u0_fn)
    u_next = model(x, t2, u0_fn=u0_fn)

    xR = torch.clamp(x + dx_fd, cfg.X_MIN, cfg.X_MAX)
    xL = torch.clamp(x - dx_fd, cfg.X_MIN, cfg.X_MAX)

    uR = model(xR, t, u0_fn=u0_fn)
    uL = model(xL, t, u0_fn=u0_fn)

    fR = uR * (1.0 - uR)
    fL = uL * (1.0 - uL)
    f_x = (fR - fL) / (2.0 * dx_fd)

    u_euler = u_now - dt * f_x
    return torch.mean((u_next - u_euler) ** 2)


def train_vpinn_fv_with_anchors(
    layers: List[int],
    u0_fn: Callable,
    x_truth: np.ndarray,
    t_truth: np.ndarray,
    u_truth: np.ndarray,
    steps: int = cfg.STEPS,
    lr: float = cfg.LR,
    n_fv: int = 8000,
    n_ic: int = 4096,
    n_bc: int = 1024,
    n_sup: int = 1024,
    n_time: int = 4000,
    # weights
    w_fv: float = 1.0,
    w_ic: float = 50.0,
    w_bc: float = 5.0,
    w_sup: float = 10.0,
    w_time: float = 5.0,
    # time-loss params
    dt_time: float = 0.02,
    dx_time: float = 0.01,
    activation: Union[str, nn.Module] = "tanh",
    hard_init: bool = True,
    device: Optional[torch.device] = None,
    log_every: int = cfg.LOG_EVERY,
) -> Tuple[LWRPINN, TrainHistory]:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LWRPINN(layers=layers, activation=activation, hard_init=hard_init).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    dx = float((cfg.X_MAX - cfg.X_MIN) / (cfg.NX_EVAL - 1))

    hist = TrainHistory(loss=[], loss_fv=[], loss_ic=[], loss_bc=[], loss_sup=[], loss_time=[])

    for it in range(1, steps + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        # FV residual points
        x_c = _sample_x(n_fv, device)
        t_c = _sample_t(n_fv, device, bias_early=True)
        r = fv_residual_midpoint(model, x_c, t_c, dx=dx, u0_fn=u0_fn)
        loss_fv = torch.mean(r ** 2)

        # IC loss
        x0, t0 = _sample_initial(n_ic, device)
        u0_true = u0_fn(x0)
        u0_pred = model(x0, t0, u0_fn=u0_fn)
        loss_ic = torch.mean((u0_pred - u0_true) ** 2)

        # periodic BC
        loss_bc = _periodic_bc_loss(model, u0_fn=u0_fn, n=n_bc, device=device)

        # supervised anchors (Godunov truth)
        xq = np.random.uniform(cfg.X_MIN, cfg.X_MAX, size=(n_sup,))
        tq = np.random.uniform(cfg.T_MIN, cfg.T_MAX, size=(n_sup,))
        uq = _interp_truth(u_truth, x_truth, t_truth, xq, tq)

        xt = torch.tensor(xq.reshape(-1, 1), dtype=torch.float32, device=device)
        tt = torch.tensor(tq.reshape(-1, 1), dtype=torch.float32, device=device)
        ut = torch.tensor(uq.reshape(-1, 1), dtype=torch.float32, device=device)

        up = model(xt, tt, u0_fn=u0_fn)
        loss_sup = torch.mean((up - ut) ** 2)

        # time consistency loss (THIS is what you were missing)
        loss_time = _time_consistency_loss(
            model=model,
            u0_fn=u0_fn,
            n=n_time,
            dt=dt_time,
            dx_fd=dx_time,
            device=device,
        )

        loss = (
            w_fv * loss_fv
            + w_ic * loss_ic
            + w_bc * loss_bc
            + w_sup * loss_sup
            + w_time * loss_time
        )

        loss.backward()
        opt.step()

        if it % log_every == 0 or it == 1:
            lf = float(loss_fv.detach().cpu())
            li = float(loss_ic.detach().cpu())
            lb = float(loss_bc.detach().cpu())
            ls = float(loss_sup.detach().cpu())
            lt = float(loss_time.detach().cpu())
            ltot = float(loss.detach().cpu())
            print(
                f"[{it:6d}/{steps}] loss={ltot:.3e} | fv={lf:.3e} ic={li:.3e} "
                f"bc={lb:.3e} sup={ls:.3e} time={lt:.3e}"
            )

            hist.loss.append(ltot)
            hist.loss_fv.append(lf)
            hist.loss_ic.append(li)
            hist.loss_bc.append(lb)
            hist.loss_sup.append(ls)
            hist.loss_time.append(lt)

    return model, hist