from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt

import config as cfg
from gudonov import solve_fvm
from vpinns import train_vpinn_fv_with_anchors


# -------------------------
# Sinusoidal (optionally quantized) IC
# -------------------------
@dataclass
class SinusoidConfig:
    u_bar: float = 0.25
    A: float = 0.20
    k: int = 4
    quantize: bool = True
    n_levels: int = 16
    clamp_eps: float = 1e-4


def make_u0(sc: SinusoidConfig):
    u_bar, A, k = float(sc.u_bar), float(sc.A), int(sc.k)
    quantize, n_levels = bool(sc.quantize), int(sc.n_levels)
    eps = float(sc.clamp_eps)

    def u0_numpy(x: np.ndarray) -> np.ndarray:
        u = u_bar + A * np.sin(2.0 * np.pi * k * x)
        u = np.clip(u, eps, 1.0 - eps)
        if quantize:
            u = np.round(u * (n_levels - 1)) / (n_levels - 1)
            u = np.clip(u, eps, 1.0 - eps)
        return u.astype(float)

    def u0_torch(x: torch.Tensor) -> torch.Tensor:
        u = u_bar + A * torch.sin(2.0 * torch.pi * k * x)
        u = torch.clamp(u, eps, 1.0 - eps)
        if quantize:
            u = torch.round(u * (n_levels - 1)) / (n_levels - 1)
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

    outdir = "results/multiple"
    os.makedirs(outdir, exist_ok=True)

    # -------------------------
    # Choose your sinusoid IC here
    # -------------------------
    sc = SinusoidConfig(
        u_bar=0.25,
        A=0.20,
        k=4,
        quantize=True,     # set False to test smooth sinusoid
        n_levels=16,
    )
    u0_numpy, u0_torch = make_u0(sc)

    # Discrete u0 for Godunov grid
    nx = cfg.NX_EVAL
    x0 = np.linspace(cfg.X_MIN, cfg.X_MAX, nx)
    u0_disc = u0_numpy(x0)

    # -------------------------
    # Godunov truth (periodic)
    # -------------------------
    x_t, t_t, u_truth = solve_fvm(
        u0=u0_disc,
        nt_out=cfg.NT_EVAL,
        x_min=cfg.X_MIN,
        x_max=cfg.X_MAX,
        t_max=cfg.T_MAX,
        cfl=0.3,
        bc="periodic",   # IMPORTANT for sinusoid
    )

    # -------------------------
    # Train NN (FV + anchors + time-consistency must be inside vpinns.py)
    # -------------------------
    layers = [256, 256, 256, 256]
    model, hist = train_vpinn_fv_with_anchors(
        layers=layers,
        u0_fn=u0_torch,
        x_truth=x_t,
        t_truth=t_t,
        u_truth=u_truth,
        steps=cfg.STEPS,
        lr=cfg.LR,
        n_fv=12000,
        n_ic=cfg.INITIAL_SAMPLES,
        n_bc=cfg.BOUNDARY_SAMPLES,
        n_sup=2048,
        w_fv=1.0,
        w_ic=50.0,
        w_bc=5.0,
        w_sup=10.0,
        hard_init=True,
        activation="tanh",
        device=device,
        log_every=cfg.LOG_EVERY,
    )

    # -------------------------
    # Predict & metrics
    # -------------------------
    u_pred = eval_on_grid(model, u0_torch, x_t, t_t, device)
    err = np.abs(u_pred - u_truth)

    mse_global = mse_field(u_pred, u_truth)
    rel_global = rel_l2_field(u_pred, u_truth)

    mse_final = mse_slice(u_pred, u_truth, time_index=-1)
    rel_l2_final = np.linalg.norm(u_pred[-1] - u_truth[-1]) / (np.linalg.norm(u_truth[-1]) + 1e-12)
    linf_final = np.max(np.abs(u_pred[-1] - u_truth[-1]))

    print(
        f"[sinusoid] "
        f"MSE(global)={mse_global:.3e} | "
        f"relL2(global)={rel_global:.3e} | "
        f"MSE(final)={mse_final:.3e} | "
        f"relL2(final)={rel_l2_final:.3e} | "
        f"Linf(final)={linf_final:.3e}"
    )
    # -------------------------
    # Save plots
    # -------------------------
    f_ic = os.path.join(outdir, "u0.png")
    plt.figure(figsize=(7, 3))
    plt.title("Initial condition u(x,0)")
    plt.plot(x0, u0_disc)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.tight_layout()
    plt.savefig(f_ic, dpi=200)
    print("Saved:", f_ic)

    f_trip = os.path.join(outdir, "pred_truth_error2.png")
    plt.figure(figsize=(13, 4))

    plt.subplot(1, 3, 1)
    plt.title("Truth (Godunov, periodic)")
    plt.pcolormesh(x_t, t_t, u_truth, shading="auto")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("NN prediction")
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
    print("Saved:", f_trip)

    f_slice = os.path.join(outdir, "final_slice.png")
    plt.figure(figsize=(8, 4))
    plt.title(f"Final time slice (t={t_t[-1]:.2f})")
    plt.plot(x_t, u_truth[-1], label="Truth")
    plt.plot(x_t, u_pred[-1], label="NN")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.tight_layout()
    plt.savefig(f_slice, dpi=200)
    print("Saved:", f_slice)


if __name__ == "__main__":
    main()