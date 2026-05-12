from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from gudonov import solve_fvm


# ============================================================
# 1) Config
# ============================================================
@dataclass
class Config:
    x_min: float = -1.0
    x_max: float = 1.0
    nx: int = 256
    t_max: float = 1.0
    nt_out: int = 101

    hidden: int = 64
    depth: int = 3
    latent: int = 32

    batch_size: int = 1
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-6

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bc: str = "copy"

    # Train only on a few 1-jump Riemann problems
    train_riemann_states: Tuple[Tuple[float, float], ...] = (
        (0.20, 0.75),  # shock
        (0.80, 0.35),  # rarefaction
        (0.10, 0.60),  # shock
    )
    train_jump_locations: Tuple[float, ...] = (0.0,)

    # Sinusoidal staircase test
    sin_mean: float = 0.55
    sin_amp: float = 0.28
    sin_k: int = 2
    sin_levels: int = 8


cfg = Config()


# ============================================================
# 2) Initial conditions
# ============================================================
def make_riemann_ic(x: np.ndarray, uL: float, uR: float, x0: float = 0.0) -> np.ndarray:
    return np.where(x < x0, uL, uR).astype(np.float32)


def quantize_to_levels(u: np.ndarray, n_levels: int) -> np.ndarray:
    umin, umax = float(np.min(u)), float(np.max(u))
    if abs(umax - umin) < 1e-12:
        return u.copy()
    levels = np.linspace(umin, umax, n_levels)
    idx = np.argmin(np.abs(u[:, None] - levels[None, :]), axis=1)
    return levels[idx]


def make_piecewise_constant_sinusoid(
    x: np.ndarray,
    mean: float = 0.55,
    amp: float = 0.28,
    k: int = 2,
    n_levels: int = 8,
) -> np.ndarray:
    u = mean + amp * np.sin(2.0 * np.pi * k * (x - x.min()) / (x.max() - x.min()))
    u = np.clip(u, 0.0, 1.0)
    u = quantize_to_levels(u, n_levels)
    return u.astype(np.float32)


# ============================================================
# 3) Dataset
# ============================================================
def build_riemann_dataset(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=np.float32)

    U0, U = [], []
    T = None

    for (uL, uR) in cfg.train_riemann_states:
        for x0 in cfg.train_jump_locations:
            u0 = make_riemann_ic(x, uL, uR, x0=x0)
            _, t_out, u_hist = solve_fvm(
                u0=u0,
                nt_out=cfg.nt_out,
                x_min=cfg.x_min,
                x_max=cfg.x_max,
                t_max=cfg.t_max,
                bc=cfg.bc,
            )
            U0.append(u0)
            U.append(u_hist.astype(np.float32))
            if T is None:
                T = t_out.astype(np.float32)

    return np.stack(U0, axis=0), np.stack(U, axis=0), T


class RolloutDataset(torch.utils.data.Dataset):
    def __init__(self, u0: np.ndarray, target: np.ndarray):
        self.u0 = torch.from_numpy(u0).float()
        self.target = torch.from_numpy(target).float()

    def __len__(self) -> int:
        return self.u0.shape[0]

    def __getitem__(self, idx: int):
        return self.u0[idx], self.target[idx]


# ============================================================
# 4) MLP
# ============================================================
class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.Tanh())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 5) FluxGNN-style 1D model
# ============================================================
class FluxGNN1D(nn.Module):
    def __init__(self, hidden: int = 64, depth: int = 3, latent: int = 32):
        super().__init__()

        self.encoder = nn.Linear(1, latent, bias=False)
        self.decoder = nn.Linear(latent, 1, bias=False)

        nn.init.normal_(self.encoder.weight, mean=0.0, std=0.2)
        with torch.no_grad():
            w = self.encoder.weight.data
            denom = torch.sum(w * w).clamp_min(1e-8)
            self.decoder.weight.data = w.t() / denom

        self.flux_mlp = MLP(
            in_dim=4 * latent,
            out_dim=1,
            hidden=hidden,
            depth=depth,
        )
        self.flux_gate = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def physical_flux(u: torch.Tensor) -> torch.Tensor:
        return u * (1.0 - u)

    def encode(self, u: torch.Tensor) -> torch.Tensor:
        return self.encoder(u.unsqueeze(-1))

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        return self.decoder(h).squeeze(-1)

    def interface_flux(
        self,
        uL: torch.Tensor,
        uR: torch.Tensor,
        hL: torch.Tensor,
        hR: torch.Tensor,
    ) -> torch.Tensor:
        hC = 0.5 * (hL + hR)
        hJ = hR - hL
        feat = torch.cat([hL, hR, hC, hJ], dim=-1)

        learned_flux = self.flux_mlp(feat).squeeze(-1)
        phys_mid = self.physical_flux(0.5 * (uL + uR))
        gate = torch.sigmoid(self.flux_gate)

        return (1.0 - gate) * phys_mid + gate * learned_flux

    def one_step(self, u: torch.Tensor, dt: float, dx: float, bc: str = "copy") -> torch.Tensor:
        if bc == "periodic":
            u_ext = torch.cat([u[:, -1:], u, u[:, :1]], dim=1)
        else:
            u_ext = torch.cat([u[:, :1], u, u[:, -1:]], dim=1)

        h_ext = self.encode(u_ext)

        uL = u_ext[:, :-1]
        uR = u_ext[:, 1:]
        hL = h_ext[:, :-1, :]
        hR = h_ext[:, 1:, :]

        fhat = self.interface_flux(uL, uR, hL, hR)
        unew = u - (dt / dx) * (fhat[:, 1:] - fhat[:, :-1])

        return torch.clamp(unew, 0.0, 1.0)

    def rollout(
        self,
        u0: torch.Tensor,
        t_out: np.ndarray,
        x_min: float,
        x_max: float,
        bc: str = "copy",
    ) -> torch.Tensor:
        _, nx = u0.shape
        dx = (x_max - x_min) / (nx - 1)

        u = u0
        out = [u]
        for k in range(1, len(t_out)):
            dt = float(t_out[k] - t_out[k - 1])
            u = self.one_step(u, dt=dt, dx=dx, bc=bc)
            out.append(u)

        return torch.stack(out, dim=1)


# ============================================================
# 6) Training
# ============================================================
def train_model(
    model: FluxGNN1D,
    train_loader: torch.utils.data.DataLoader,
    t_out: np.ndarray,
    cfg: Config,
) -> FluxGNN1D:
    device = torch.device(cfg.device)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    best_loss = float("inf")
    best_state = None

    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0

        for u0, target in train_loader:
            u0 = u0.to(device)
            target = target.to(device)

            pred = model.rollout(
                u0,
                t_out=t_out,
                x_min=cfg.x_min,
                x_max=cfg.x_max,
                bc=cfg.bc,
            )

            loss = mse(pred, target)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item() * u0.size(0)

        train_loss /= len(train_loader.dataset)

        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 100 == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:4d} | train {train_loss:.6e} | elapsed {time.time() - t0:.1f}s")

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"best train loss = {best_loss:.6e}")
    return model


# ============================================================
# 7) Metric
# ============================================================
def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    num = np.linalg.norm(pred - true)
    den = np.linalg.norm(true) + 1e-12
    return float(num / den)

def make_multi_discontinuity_ic(
    x: np.ndarray,
    jumps: Tuple[float, ...] = (-0.60, -0.05, 0.35, 0.70),
    values: Tuple[float, ...] = (0.78, 0.18, 0.82, 0.30, 0.76),
) -> np.ndarray:
    """
    Piecewise-constant IC with specified jump locations and plateau values.
    len(values) must equal len(jumps) + 1.
    """
    if len(values) != len(jumps) + 1:
        raise ValueError("len(values) must be len(jumps) + 1")

    u = np.empty_like(x, dtype=np.float32)

    # left of first jump
    u[x < jumps[0]] = values[0]

    # between jumps
    for i in range(len(jumps) - 1):
        mask = (x >= jumps[i]) & (x < jumps[i + 1])
        u[mask] = values[i + 1]

    # right of last jump
    u[x >= jumps[-1]] = values[-1]

    return u
# ============================================================
# 8) Main
# ============================================================
def main():
    print(f"device: {cfg.device}")
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=np.float32)

    # --------------------------------------------------------
    # Train only on a few Riemann problems
    # --------------------------------------------------------
    print("Generating Riemann training data with Godunov...")
    train_u0, train_U, t_out = build_riemann_dataset(cfg)

    print("Training problems:")
    for i, (uL, uR) in enumerate(cfg.train_riemann_states):
        wave_type = "shock" if uL < uR else "rarefaction"
        print(f"  sample {i+1}: uL={uL:.2f}, uR={uR:.2f}  -> {wave_type}")

    train_ds = RolloutDataset(train_u0, train_U)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True
    )

    print("Training FluxGNN-style model...")
    model = FluxGNN1D(hidden=cfg.hidden, depth=cfg.depth, latent=cfg.latent)
    model = train_model(model, train_loader, t_out, cfg)

    # --------------------------------------------------------
    # Test on staircase sinusoid: many shocks/rarefactions
    # --------------------------------------------------------
    print("Testing on staircase sinusoidal IC...")

   # u0_test = make_piecewise_constant_sinusoid(
    #    x,
     #   mean=cfg.sin_mean,
      #  amp=cfg.sin_amp,
       # k=cfg.sin_k,
     #   n_levels=cfg.sin_levels,
    #)

    u0_test = make_multi_discontinuity_ic(
        x,
        jumps=(-0.60, -0.05, 0.35, 0.70),
        values=(0.78, 0.18, 0.82, 0.30, 0.76),
    )

    _, t_ref, U_ref = solve_fvm(
        u0=u0_test,
        nt_out=cfg.nt_out,
        x_min=cfg.x_min,
        x_max=cfg.x_max,
        t_max=cfg.t_max,
        bc=cfg.bc,
    )

    model.eval()
    with torch.no_grad():
        pred = model.rollout(
            torch.from_numpy(u0_test[None, :]).float().to(cfg.device),
            t_out=t_ref,
            x_min=cfg.x_min,
            x_max=cfg.x_max,
            bc=cfg.bc,
        )[0].cpu().numpy()

    mse_test = np.mean((pred - U_ref) ** 2)
    rel_test = relative_l2(pred, U_ref)

    print(f"Sinusoid test MSE        : {mse_test:.6e}")
    print(f"Sinusoid test relative L2: {rel_test:.6e}")

    # --------------------------------------------------------
    # Plot initial condition + Godunov solution
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(x, u0_test, linewidth=2)
    axes[0].set_title("Initial condition (piecewise-constant sinusoid)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("u")
    axes[0].grid(True, alpha=0.3)

    im = axes[1].pcolormesh(x, t_ref, U_ref, shading="auto", cmap="viridis")
    axes[1].set_title("Godunov solution u(x,t)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(im, ax=axes[1], label="u")

    plt.suptitle("Sinusoidal IC → many shocks/rarefactions (Godunov)")
    plt.savefig("fluxgnn_sinusoid_truth.png", dpi=200)
    plt.show()

    # --------------------------------------------------------
    # 3-panel comparison
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    im0 = axes[0].pcolormesh(x, t_ref, U_ref, shading="auto", cmap="viridis")
    axes[0].set_title("Truth (Godunov)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(x, t_ref, pred, shading="auto", cmap="viridis")
    axes[1].set_title("FluxGNN-style prediction")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(x, t_ref, np.abs(pred - U_ref), shading="auto", cmap="viridis")
    axes[2].set_title("Absolute error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    fig.colorbar(im2, ax=axes[2])

    plt.savefig("fluxgnn_sinusoid_comparison.png", dpi=200)
    plt.show()

    # --------------------------------------------------------
    # Final-time comparison
    # --------------------------------------------------------
    final_idx = -1
    plt.figure(figsize=(7, 4))
    plt.plot(x, u0_test, label="initial", linewidth=2)
    plt.plot(x, U_ref[final_idx], label="Godunov", linewidth=2)
    plt.plot(x, pred[final_idx], "--", label="FluxGNN-style", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title(f"Final time comparison at t={t_ref[final_idx]:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fluxgnn_sinusoid_final.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()