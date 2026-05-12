from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from gudonov import godunov_flux, add_ghost_cells


# ============================================================
# 1) LWR physics (torch — for model's internal flux)
# ============================================================
def flux_lwr_torch(u: torch.Tensor) -> torch.Tensor:
    return u * (1.0 - u)


def godunov_flux_torch(u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
    """Godunov flux for concave LWR flux f(u)=u(1-u)."""
    f_left = flux_lwr_torch(u_left)
    f_right = flux_lwr_torch(u_right)

    f_min = torch.minimum(f_left, f_right)
    f_max = torch.maximum(f_left, f_right)

    u_lo = torch.minimum(u_left, u_right)
    u_hi = torch.maximum(u_left, u_right)
    has_mid = (u_lo <= 0.5) & (0.5 <= u_hi)
    f_max = torch.where(has_mid, torch.maximum(f_max, f_max.new_full((), 0.25)), f_max)

    return torch.where(u_left <= u_right, f_min, f_max)


def make_riemann_ic(
    x: np.ndarray, u_left: float, u_right: float, x_jump: float = 0.0
) -> np.ndarray:
    return np.where(x < x_jump, u_left, u_right).astype(np.float32)


# ============================================================
# 2) Truth solver — gudonov.py flux + ghost cells, fixed dt
#
# Why fixed dt instead of gudonov.py's adaptive compute_time_step?
#   Adaptive dt gives different nt for different ICs (different wave
#   speeds => different step counts). Fixed dt = cfl*dx/1.0 uses the
#   worst-case max speed=1, so ALL ICs produce identical time grids
#   and y_train can be stacked and compared step-by-step with the model.
# ============================================================
def solve_truth_fixed_dt(
    u0: np.ndarray,
    dx: float,
    dt: float,
    t_max: float,
    bc: str = "copy",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Godunov solver using your gudonov.py flux and ghost cells,
    with a fixed dt so all ICs produce the same nt.

    Returns:
        times : (nt,)
        U     : (nt, nx)
    """
    u = u0.copy().astype(float)
    t = 0.0
    times = [0.0]
    states = [u.copy()]

    while t < t_max - 1e-14:
        dt_step = min(dt, t_max - t)
        u_ext = add_ghost_cells(u, bc=bc)
        fhat = godunov_flux(u_ext)
        u = u - (dt_step / dx) * (fhat[1:] - fhat[:-1])
        t += dt_step
        times.append(t)
        states.append(u.copy())

    return np.array(times), np.stack(states, axis=0)


# ============================================================
# 3) Small MLP helper
# ============================================================
def make_mlp(
    in_dim: int, hidden: int, out_dim: int, depth: int, activation: str
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be >= 1")

    act_cls = nn.GELU if activation.lower() == "gelu" else nn.Tanh
    dims = [in_dim] + [hidden] * (depth - 1) + [out_dim]

    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


# ============================================================
# 4) FluxGNN 1D with latent space
#
# Theorem 3.3: linear encoder + pseudoinverse decoder
#   h = e * u,  d = e / (e^T e),  d^T e = 1 always
#
# Theorem 3.4: permutation-invariant flux
#   features = [h_L + h_R, |h_L - h_R|]  (symmetric)
# Eq. 9: conservative FV update in latent space
#   h^{n+1} = h^n - (dt/dx) * (fhat_{i+1/2} - fhat_{i-1/2})
# ============================================================
class FluxGNN1DLatent(nn.Module):

    def __init__(
        self,
        latent_dim: int = 32,
        hidden: int = 64,
        depth: int = 3,
        activation: str = "gelu",
        use_base_flux: bool = False,
        base_flux_weight: float = 0.5,
        latent_flux_scale: float = 0.25,
    ) -> None:
        super().__init__()

        self.latent_dim = int(latent_dim)
        self.use_base_flux = bool(use_base_flux)
        self.base_flux_weight = float(base_flux_weight)
        self.latent_flux_scale = float(latent_flux_scale)

        e = torch.ones(self.latent_dim, dtype=torch.float32) / np.sqrt(self.latent_dim)
        self.encoder_vec = nn.Parameter(e) # h = eu. ; h is latent state of each cell;This is the vector e which is trained by gradient descent together with the MLP weights, MLP(hi​,hi+1​) is the flux 

        in_dim = 2 * self.latent_dim
        self.flux_mlp = make_mlp(in_dim, hidden, self.latent_dim, depth, activation)

    def decoder_vec(self) -> torch.Tensor:
        e = self.encoder_vec
        return e / (torch.sum(e * e) + 1e-12)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        return u.unsqueeze(-1) * self.encoder_vec

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sum(h * self.decoder_vec(), dim=-1)

    def build_edge_features(self, h_left: torch.Tensor, h_right: torch.Tensor) -> torch.Tensor:
        return torch.cat([h_left + h_right, torch.abs(h_right - h_left)], dim=-1)

    def latent_neural_flux(self, h_left: torch.Tensor, h_right: torch.Tensor) -> torch.Tensor:
        edge_feat = self.build_edge_features(h_left, h_right)
        flat = edge_feat.reshape(-1, edge_feat.shape[-1])
        out = self.flux_mlp(flat).reshape_as(h_left)
        return torch.tanh(out) * self.latent_flux_scale

    def compute_latent_flux(self, u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
        h_left = self.encode(u_left)
        h_right = self.encode(u_right)
        flux_learned = self.latent_neural_flux(h_left, h_right)

        if self.use_base_flux:
            flux_base_latent = self.encode(godunov_flux_torch(u_left, u_right))
            w = self.base_flux_weight
            return (1.0 - w) * flux_base_latent + w * flux_learned

        return flux_learned

    def step(self, u: torch.Tensor, dt: float, dx: float, boundary: str = "copy") -> torch.Tensor:
        h = self.encode(u)

        if boundary == "copy":
            u_ext = torch.empty(u.size(0), u.size(1) + 2, device=u.device, dtype=u.dtype)
            u_ext[:, 1:-1] = u
            u_ext[:, 0] = u[:, 0]
            u_ext[:, -1] = u[:, -1]
            flux = self.compute_latent_flux(u_ext[:, :-1], u_ext[:, 1:])
            h_new = h - (dt / dx) * (flux[:, 1:] - flux[:, :-1])
            return self.decode(h_new)

        if boundary == "periodic":
            u_right = torch.roll(u, shifts=-1, dims=1)
            flux = self.compute_latent_flux(u, u_right)
            h_new = h - (dt / dx) * (flux - torch.roll(flux, shifts=1, dims=1))
            return self.decode(h_new)

        if boundary == "fixed":
            flux = self.compute_latent_flux(u[:, :-1], u[:, 1:])
            h_new = h.clone()
            h_new[:, 1:-1] = h[:, 1:-1] - (dt / dx) * (flux[:, 1:] - flux[:, :-1])
            u_new = self.decode(h_new)
            u_new[:, 0] = u[:, 0]
            u_new[:, -1] = u[:, -1]
            return u_new

        raise ValueError("boundary must be 'periodic', 'copy', or 'fixed'")

    def rollout(
        self, u0: torch.Tensor, dt: float, dx: float, n_steps: int, boundary: str = "copy"
    ) -> torch.Tensor:
        """Returns (B, n_steps, nx)"""
        u = u0
        outs = [u]
        for _ in range(1, n_steps):
            u = self.step(u, dt, dx, boundary)
            outs.append(u)
        return torch.stack(outs, dim=1)


# ============================================================
# 5) Config
# ============================================================
@dataclass
class Config:
    x_min: float = -1.0
    x_max: float = 1.0
    nx: int = 256
    t_max: float = 0.8
    cfl: float = 0.45
    boundary: str = "copy"

    # For CONCAVE flux f(u)=u(1-u):
    #   uL > uR  =>  rarefaction
    #   uL < uR  =>  shock
    train_pairs: Tuple[Tuple[float, float], ...] = (
        (0.75, 0.20),   # rarefaction
        (0.35, 0.80),   # shock
        (0.60, 0.10),   # rarefaction
    )
    x_jump_train: float = 0.0

    test_pair: Tuple[float, float] = (0.65, 0.15)
    x_jump_test: float = 0.0

    latent_dim: int = 32
    hidden: int = 64
    depth: int = 3
    activation: str = "gelu"
    use_base_flux: bool = False
    base_flux_weight: float = 0.50
    latent_flux_scale: float = 0.25

    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    seed: int = 0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "results_fluxgnn1d_latent"


# ============================================================
# 6) Grid and dt
# ============================================================
def build_grid(cfg: Config) -> Tuple[np.ndarray, float]:
    # Match gudonov.py exactly: dx = x[1]-x[0] = L/(nx-1)
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=np.float32)
    dx = float(x[1] - x[0])
    return x, dx


def build_fixed_dt(cfg: Config, dx: float) -> float:
    # Worst-case max speed = 1 for f'(u)=1-2u on [0,1]
    # Smallest safe dt for all ICs
    return cfg.cfl * dx / 1.0


# ============================================================
# 7) Dataset generation
# ============================================================
def build_train_dataset(
    cfg: Config, x: np.ndarray, dx: float, dt: float
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    u0_list, y_list = [], []
    times_ref = None

    for uL, uR in cfg.train_pairs:
        u0 = make_riemann_ic(x, uL, uR, cfg.x_jump_train)
        times, U = solve_truth_fixed_dt(
            u0, dx=dx, dt=dt, t_max=cfg.t_max, bc=cfg.boundary
        )

        if times_ref is None:
            times_ref = times
        else:
            assert len(times) == len(times_ref), \
                "Time grids differ — should not happen with fixed dt."

        u0_list.append(u0)
        y_list.append(U.astype(np.float32))

    u0_train = torch.from_numpy(np.stack(u0_list, axis=0))
    y_train = torch.from_numpy(np.stack(y_list, axis=0))
    return u0_train, y_train, times_ref


# ============================================================
# 8) Training
# ============================================================
def train_model(
    cfg: Config,
    model: FluxGNN1DLatent,
    u0_train: torch.Tensor,
    y_train: torch.Tensor,
    dt: float,
    dx: float,
) -> List[float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    u0_train = u0_train.to(cfg.device)
    y_train = y_train.to(cfg.device)
    n_steps = y_train.shape[1]

    loss_hist: List[float] = []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        opt.zero_grad(set_to_none=True)

        pred = model.rollout(u0_train, dt=dt, dx=dx, n_steps=n_steps, boundary=cfg.boundary)
        loss = torch.mean((pred - y_train) ** 2)

        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        loss_hist.append(float(loss.item()))

        if epoch == 1 or epoch % 100 == 0:
            print(f"epoch {epoch:5d} | train MSE = {loss.item():.8e}")

    print(f"Training finished in {time.time() - t0:.2f} s")
    return loss_hist


# ============================================================
# 9) Evaluation
# ============================================================
def evaluate_unknown_jump(
    cfg: Config,
    model: FluxGNN1DLatent,
    x: np.ndarray,
    dx: float,
    dt: float,
):
    uL, uR = cfg.test_pair
    u0_test = make_riemann_ic(x, uL, uR, cfg.x_jump_test)

    times_test, U_true = solve_truth_fixed_dt(
        u0_test, dx=dx, dt=dt, t_max=cfg.t_max, bc=cfg.boundary
    )
    U_true = U_true.astype(np.float32)

    model.eval()
    with torch.no_grad():
        u0_t = torch.from_numpy(u0_test[None, :]).to(cfg.device)
        U_pred = model.rollout(
            u0_t, dt=dt, dx=dx, n_steps=len(times_test), boundary=cfg.boundary
        )[0].detach().cpu().numpy()

    nt = min(U_true.shape[0], U_pred.shape[0])
    U_true = U_true[:nt]
    U_pred = U_pred[:nt]
    times_test = times_test[:nt]

    err = U_pred - U_true
    mse  = float(np.mean(err ** 2))
    mae  = float(np.mean(np.abs(err)))
    linf = float(np.max(np.abs(err)))

    print("\nUnknown jump test")
    print(f"  pair = ({uL:.3f}, {uR:.3f})")
    print(f"  MSE  = {mse:.8e}")
    print(f"  MAE  = {mae:.8e}")
    print(f"  Linf = {linf:.8e}")

    return u0_test, U_true, U_pred, err, times_test, mse, mae, linf


def conservation_error(u_hist: np.ndarray, dx: float) -> float:
    integrals = u_hist.sum(axis=-1) * dx
    return float(np.max(np.abs(integrals - integrals[0])))


# ============================================================
# 10) Plotting
# ============================================================
def plot_loss(cfg: Config, loss_hist: List[float]):
    plt.figure(figsize=(7, 4))
    plt.plot(loss_hist)
    plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Train MSE")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "training_loss.png"), dpi=160)
    plt.close()


def plot_final_snapshot(cfg: Config, x, U_true, U_pred):
    plt.figure(figsize=(8, 5))
    plt.plot(x, U_true[-1], label="Godunov truth", linewidth=2)
    plt.plot(x, U_pred[-1], "--", label="FluxGNN pred", linewidth=2)
    plt.xlabel("x"); plt.ylabel("u(x, t_max)")
    plt.title("Unknown jump: final snapshot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "final_snapshot.png"), dpi=160)
    plt.close()


def plot_final_error(cfg: Config, x, err):
    plt.figure(figsize=(8, 4))
    plt.plot(x, err[-1], linewidth=2)
    plt.xlabel("x"); plt.ylabel("pred - truth")
    plt.title("Unknown jump: final error")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "final_error.png"), dpi=160)
    plt.close()


def plot_space_time(cfg: Config, x, times, U_true, U_pred, err):
    nt = min(len(times), U_true.shape[0], U_pred.shape[0])
    extent = [x[0], x[-1], times[0], times[nt - 1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(U_true[:nt], aspect="auto", extent=extent, origin="lower", cmap="viridis")
    axes[0].set_title("Godunov truth"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(U_pred[:nt], aspect="auto", extent=extent, origin="lower", cmap="viridis")
    axes[1].set_title("FluxGNN prediction"); axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(np.abs(err[:nt]), aspect="auto", extent=extent, origin="lower", cmap="hot", vmin=0)
    axes[2].set_title("|Prediction error|"); axes[2].set_xlabel("x"); axes[2].set_ylabel("t")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "spacetime_truth_pred_error.png"), dpi=160)
    plt.close()


def plot_selected_times(cfg: Config, x, times, U_true, U_pred):
    nt = min(len(times), U_true.shape[0], U_pred.shape[0])
    chosen = [0, nt // 4, nt // 2, nt - 1]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    for ax, k in zip(axes, chosen):
        ax.plot(x, U_true[k], label="Godunov truth", linewidth=2)
        ax.plot(x, U_pred[k], "--", label="FluxGNN pred", linewidth=2)
        ax.set_title(f"t = {times[k]:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("u")
        ax.grid(alpha=0.25)

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "selected_snapshots.png"), dpi=160)
    plt.close()


def plot_conservation(cfg: Config, times, U_true, U_pred, dx):
    nt = min(len(times), U_true.shape[0], U_pred.shape[0])
    plt.figure(figsize=(6, 3))
    plt.plot(times[:nt], U_true[:nt].sum(axis=-1) * dx, label="Godunov")
    plt.plot(times[:nt], U_pred[:nt].sum(axis=-1) * dx, "--", label="FluxGNN")
    plt.xlabel("t"); plt.ylabel(r"$\int u\,dx$")
    plt.title("Global conservation")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "conservation.png"), dpi=160)
    plt.close()


# ============================================================
# 11) Main
# ============================================================
def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    x, dx = build_grid(cfg)
    dt = build_fixed_dt(cfg, dx)

    print("Config")
    print(f"  device        : {cfg.device}")
    print(f"  nx            : {cfg.nx}")
    print(f"  dx            : {dx:.6f}  (gudonov.py: L/(nx-1))")
    print(f"  dt            : {dt:.6f}  (fixed: cfl*dx/1.0)")
    print(f"  t_max         : {cfg.t_max}")
    print(f"  train pairs   : {cfg.train_pairs}")
    print(f"  unknown pair  : {cfg.test_pair}")
    print(f"  use_base_flux : {cfg.use_base_flux}")

    u0_train, y_train, times = build_train_dataset(cfg, x, dx, dt)
    print(f"u0_train shape : {tuple(u0_train.shape)}")
    print(f"y_train  shape : {tuple(y_train.shape)}")
    print(f"nt             : {len(times)}")

    model = FluxGNN1DLatent(
        latent_dim=cfg.latent_dim,
        hidden=cfg.hidden,
        depth=cfg.depth,
        activation=cfg.activation,
        use_base_flux=cfg.use_base_flux,
        base_flux_weight=cfg.base_flux_weight,
        latent_flux_scale=cfg.latent_flux_scale,
    ).to(cfg.device)

    with torch.no_grad():
        e = model.encoder_vec
        d = model.decoder_vec()
        print(f"Init check: d^T e = {torch.dot(d, e).item():.8f}  (must be 1.0)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    loss_hist = train_model(cfg, model, u0_train, y_train, dt, dx)

    with torch.no_grad():
        e = model.encoder_vec
        d = model.decoder_vec()
        print(f"Post-train: d^T e = {torch.dot(d, e).item():.8f}  (must be 1.0)")

    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "fluxgnn1d_latent_model.pt"))

    u0_test, U_true, U_pred, err, times_test, mse, mae, linf = evaluate_unknown_jump(
        cfg, model, x, dx, dt
    )

    cons_pred = conservation_error(U_pred, dx)
    cons_ref  = conservation_error(U_true, dx)
    print(f"  Conservation error (pred)   : {cons_pred:.6e}")
    print(f"  Conservation error (Godunov): {cons_ref:.6e}")

    np.savez_compressed(
        os.path.join(cfg.save_dir, "unknown_jump_results.npz"),
        x=x, times=times_test, u0=u0_test,
        godunov=U_true, pred=U_pred, error=err,
    )

    with open(os.path.join(cfg.save_dir, "metrics.txt"), "w") as f:
        f.write(f"test pair: {cfg.test_pair}\n")
        f.write(f"MSE  = {mse:.12e}\n")
        f.write(f"MAE  = {mae:.12e}\n")
        f.write(f"Linf = {linf:.12e}\n")
        f.write(f"Conservation error (pred)   = {cons_pred:.12e}\n")
        f.write(f"Conservation error (Godunov) = {cons_ref:.12e}\n")

    plot_loss(cfg, loss_hist)
    plot_final_snapshot(cfg, x, U_true, U_pred)
    plot_final_error(cfg, x, err)
    plot_space_time(cfg, x, times_test, U_true, U_pred, err)
    plot_selected_times(cfg, x, times_test, U_true, U_pred)
    plot_conservation(cfg, times_test, U_true, U_pred, dx)

    print("\nDone.")


if __name__ == "__main__":
    main()