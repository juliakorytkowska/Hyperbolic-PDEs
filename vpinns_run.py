from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt

import config as cfg
from gudonov import solve_fvm
from vpinns_true import train_vpinn_onejump


@dataclass
class JumpConfig:
    uL: float = 0.10
    uR: float = 0.30
    x0: float = 0.0
    clamp_eps: float = 1e-4


def make_u0(jc: JumpConfig):
    uL, uR, x0 = float(jc.uL), float(jc.uR), float(jc.x0)
    eps = float(jc.clamp_eps)

    def u0_numpy(x: np.ndarray) -> np.ndarray:
        u = np.where(x < x0, uL, uR).astype(float)
        u = np.clip(u, eps, 1.0 - eps)
        return u

    def u0_torch(x: torch.Tensor) -> torch.Tensor:
        u = torch.where(x < x0, torch.full_like(x, uL), torch.full_like(x, uR))
        u = torch.clamp(u, eps, 1.0 - eps)
        return u

    return u0_numpy, u0_torch


@torch.no_grad()
def eval_on_grid(model, u0_torch, x: np.ndarray, t: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X, T = np.meshgrid(x, t, indexing="xy")
    xt = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=device)
    tt = torch.tensor(T.reshape(-1, 1), dtype=torch.float32, device=device)
    u = model(xt, tt, u0_fn=u0_torch).reshape(len(t), len(x)).cpu().numpy()
    return u


def mse_field(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    return float(np.mean((u_pred - u_true) ** 2))


def rel_l2_field(u_pred: np.ndarray, u_true: np.ndarray) -> float:
    return float(
        np.linalg.norm((u_pred - u_true).ravel()) /
        (np.linalg.norm(u_true.ravel()) + 1e-12)
    )


def mse_slice(u_pred: np.ndarray, u_true: np.ndarray, time_index: int = -1) -> float:
    return float(np.mean((u_pred[time_index] - u_true[time_index]) ** 2))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    outdir = "results/one_jump_vpinn"
    os.makedirs(outdir, exist_ok=True)

    jc = JumpConfig(uL=0.10, uR=0.30, x0=0.0)
    u0_numpy, u0_torch = make_u0(jc)

    nx = cfg.NX_EVAL
    x_grid = np.linspace(cfg.X_MIN, cfg.X_MAX, nx)
    u0_disc = u0_numpy(x_grid)

    x_t, t_t, u_truth = solve_fvm(
        u0=u0_disc,
        nt_out=cfg.NT_EVAL,
        x_min=cfg.X_MIN,
        x_max=cfg.X_MAX,
        t_max=cfg.T_MAX,
        cfl=0.3,
        bc="copy",
    )

    layers = [128, 128, 128, 128]

    model, hist = train_vpinn_onejump(
        layers=layers,
        u0_fn=u0_torch,
        uL=jc.uL,
        uR=jc.uR,
        steps=cfg.STEPS,
        lr=cfg.LR,
        n_var=4096,
        n_ic=4096,
        n_bc=1024,
        w_var=1.0,
        w_ic=50.0,
        w_bc=10.0,
        activation="tanh",
        hard_init=True,
        n_fourier=6,
        scale=2.0,
        n_test=3,
        device=device,
        log_every=cfg.LOG_EVERY,
    )

    u_pred = eval_on_grid(model, u0_torch, x_t, t_t, device)
    err = np.abs(u_pred - u_truth)

    mse_global = mse_field(u_pred, u_truth)
    rel_global = rel_l2_field(u_pred, u_truth)

    mse_final = mse_slice(u_pred, u_truth, time_index=-1)
    rel_l2_final = np.linalg.norm(u_pred[-1] - u_truth[-1]) / (np.linalg.norm(u_truth[-1]) + 1e-12)
    linf_final = np.max(np.abs(u_pred[-1] - u_truth[-1]))

    print(
        f"[one-jump VPINN] "
        f"MSE(global)={mse_global:.3e} | "
        f"relL2(global)={rel_global:.3e} | "
        f"MSE(final)={mse_final:.3e} | "
        f"relL2(final)={rel_l2_final:.3e} | "
        f"Linf(final)={linf_final:.3e}"
    )

    f_ic = os.path.join(outdir, "u0_one_jump.png")
    plt.figure(figsize=(7, 3))
    plt.title("Initial condition u(x,0)")
    plt.plot(x_grid, u0_disc)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.tight_layout()
    plt.savefig(f_ic, dpi=200)

    f_trip = os.path.join(outdir, "pred_truth_error.png")
    plt.figure(figsize=(13, 4))

    plt.subplot(1, 3, 1)
    plt.title("Truth (Godunov)")
    plt.pcolormesh(x_t, t_t, u_truth, shading="auto")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("VPINN prediction")
    plt.pcolormesh(x_t, t_t, u_pred, shading="auto")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Absolute error")
    plt.pcolormesh(x_t, t_t, err, shading="auto")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f_trip, dpi=200)

    f_slice = os.path.join(outdir, "final_slice.png")
    plt.figure(figsize=(8, 4))
    plt.title(f"Final time slice (t={t_t[-1]:.2f})")
    plt.plot(x_t, u_truth[-1], label="Godunov")
    plt.plot(x_t, u_pred[-1], label="VPINN")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u")
    plt.tight_layout()
    plt.savefig(f_slice, dpi=200)

    f_loss = os.path.join(outdir, "loss_history.png")
    plt.figure(figsize=(8, 4))
    plt.plot(hist.loss, label="total")
    plt.plot(hist.loss_var, label="variational")
    plt.plot(hist.loss_ic, label="IC")
    plt.plot(hist.loss_bc, label="BC")
    plt.yscale("log")
    plt.xlabel(f"logging step (every {cfg.LOG_EVERY} iters)")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f_loss, dpi=200)


if __name__ == "__main__":
    main()