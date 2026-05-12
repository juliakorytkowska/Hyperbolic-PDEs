
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from gudonov import godunov_flux, add_ghost_cells


# ============================================================
# 1) LWR physics
# ============================================================
def flux_lwr_torch(u: torch.Tensor) -> torch.Tensor:
    return u * (1.0 - u)


def godunov_flux_torch(u_left: torch.Tensor, u_right: torch.Tensor) -> torch.Tensor:
    f_left  = flux_lwr_torch(u_left)
    f_right = flux_lwr_torch(u_right)
    f_min   = torch.minimum(f_left, f_right)
    f_max   = torch.maximum(f_left, f_right)
    u_lo    = torch.minimum(u_left, u_right)
    u_hi    = torch.maximum(u_left, u_right)
    has_mid = (u_lo <= 0.5) & (0.5 <= u_hi)
    f_max   = torch.where(has_mid, torch.maximum(f_max, f_max.new_full((), 0.25)), f_max)
    return torch.where(u_left <= u_right, f_min, f_max)


# ============================================================
# 2) Piecewise-constant IC builder
#
# For CONCAVE flux f(u) = u(1-u):
#   uL < uR  =>  shock
#   uL > uR  =>  rarefaction
#
# Usage:
#   make_pwc_ic(x, jumps=[-0.5, 0.0, 0.5],
#                  values=[0.8, 0.2, 0.7, 0.15])
#   => 3 jumps, 4 regions
# ============================================================
def make_pwc_ic(
    x: np.ndarray,
    jumps: List[float],
    values: List[float],
) -> np.ndarray:
    assert len(values) == len(jumps) + 1, \
        f"Need len(values)==len(jumps)+1, got {len(values)} vs {len(jumps)+1}"
    u = np.full_like(x, values[0], dtype=np.float32)
    for xj, val in zip(jumps, values[1:]):
        u[x >= xj] = val
    return u


# ============================================================
# 3) Truth solver — fixed dt
# ============================================================
def solve_truth_fixed_dt(
    u0: np.ndarray,
    dx: float,
    dt: float,
    t_max: float,
    bc: str = "copy",
) -> Tuple[np.ndarray, np.ndarray]:
    u      = u0.copy().astype(float)
    t      = 0.0
    times  = [0.0]
    states = [u.copy()]
    while t < t_max - 1e-14:
        dt_step = min(dt, t_max - t)
        u_ext   = add_ghost_cells(u, bc=bc)
        fhat    = godunov_flux(u_ext)
        u       = u - (dt_step / dx) * (fhat[1:] - fhat[:-1])
        t      += dt_step
        times.append(t)
        states.append(u.copy())
    return np.array(times), np.stack(states, axis=0)


# ============================================================
# 4) MLP
# ============================================================
def make_mlp(in_dim, hidden, out_dim, depth, activation) -> nn.Sequential:
    act_cls = nn.GELU if activation.lower() == "gelu" else nn.Tanh
    dims    = [in_dim] + [hidden] * (depth - 1) + [out_dim]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


# ============================================================
# 5) FluxGNN1DLatent
# ============================================================
class FluxGNN1DLatent(nn.Module):
    def __init__(self, latent_dim=32, hidden=64, depth=3,
                 activation="gelu", use_base_flux=False,
                 base_flux_weight=0.5, latent_flux_scale=0.25):
        super().__init__()
        self.latent_dim        = latent_dim
        self.use_base_flux     = use_base_flux
        self.base_flux_weight  = base_flux_weight
        self.latent_flux_scale = latent_flux_scale

        e = torch.ones(latent_dim) / np.sqrt(latent_dim)
        self.encoder_vec = nn.Parameter(e)
        self.flux_mlp    = make_mlp(2 * latent_dim, hidden, latent_dim, depth, activation)

    def decoder_vec(self):
        e = self.encoder_vec
        return e / (torch.sum(e * e) + 1e-12)

    def encode(self, u):
        return u.unsqueeze(-1) * self.encoder_vec

    def decode(self, h):
        return torch.sum(h * self.decoder_vec(), dim=-1)

    def latent_neural_flux(self, hL, hR):
        feat = torch.cat([hL + hR, torch.abs(hR - hL)], dim=-1)
        out  = self.flux_mlp(feat.reshape(-1, feat.shape[-1])).reshape_as(hL)
        return torch.tanh(out) * self.latent_flux_scale

    def compute_latent_flux(self, uL, uR):
        hL   = self.encode(uL)
        hR   = self.encode(uR)
        flux = self.latent_neural_flux(hL, hR)
        if self.use_base_flux:
            w    = self.base_flux_weight
            flux = (1 - w) * self.encode(godunov_flux_torch(uL, uR)) + w * flux
        return flux

    def step(self, u, dt, dx, boundary="copy"):
        h = self.encode(u)
        if boundary == "copy":
            u_ext          = torch.empty(u.size(0), u.size(1) + 2, device=u.device, dtype=u.dtype)
            u_ext[:, 1:-1] = u
            u_ext[:, 0]    = u[:, 0]
            u_ext[:, -1]   = u[:, -1]
            flux  = self.compute_latent_flux(u_ext[:, :-1], u_ext[:, 1:])
            h_new = h - (dt / dx) * (flux[:, 1:] - flux[:, :-1])
            return self.decode(h_new)
        raise ValueError(f"Unknown boundary: {boundary}")

    def rollout(self, u0, dt, dx, n_steps, boundary="copy"):
        u    = u0
        outs = [u]
        for _ in range(1, n_steps):
            u = self.step(u, dt, dx, boundary)
            outs.append(u)
        return torch.stack(outs, dim=1)


# ============================================================
# 6) Config
# ============================================================
@dataclass
class Config:
    x_min: float = -1.0
    x_max: float =  1.0
    nx:    int   = 256
    t_max: float = 0.8
    cfl:   float = 0.45
    boundary: str = "copy"

    # ----------------------------------------------------------
    # TRAINING: 2-jump ICs (3 regions) and 3-jump ICs (4 regions)
    # ----------------------------------------------------------
    # Each entry: (jumps, values)
    #   jumps  = interface positions   len = n
    #   values = piecewise states      len = n+1
    #
    # For CONCAVE flux f(u)=u(1-u): uL<uR => shock, uL>uR => rarefaction
    train_ics: Tuple = (
        # --- 2-jump ICs ---
        # shock then rarefaction
        ([-0.3,  0.3], [0.20, 0.75, 0.10]),
        # rarefaction then shock
        ([-0.3,  0.3], [0.80, 0.20, 0.70]),
        # rarefaction then shock (different values)
        ([-0.4,  0.2], [0.70, 0.15, 0.65]),
        # shock then rarefaction (different values)
        ([-0.2,  0.4], [0.10, 0.80, 0.20]),

        # --- 3-jump ICs ---
        # rarefaction, shock, rarefaction
        ([-0.5,  0.0,  0.5], [0.80, 0.20, 0.70, 0.15]),
        # shock, rarefaction, shock
        ([-0.5,  0.0,  0.5], [0.15, 0.70, 0.20, 0.80]),
        # rarefaction, shock, rarefaction (shifted)
        ([-0.6, -0.1,  0.4], [0.75, 0.25, 0.65, 0.10]),
        # shock, rarefaction, shock (shifted)
        ([-0.4,  0.1,  0.6], [0.10, 0.60, 0.15, 0.70]),
    )

    # ----------------------------------------------------------
    # TEST: 4-jump IC (5 regions) — never seen during training
    # ----------------------------------------------------------
    # x=-0.6: 0.80->0.20  rarefaction (uL>uR)
    # x=-0.2: 0.20->0.70  shock       (uL<uR)
    # x= 0.2: 0.70->0.15  rarefaction (uL>uR)
    # x= 0.6: 0.15->0.60  shock       (uL<uR)
    test_jumps:  Tuple = (-0.6, -0.2,  0.2,  0.6)
    test_values: Tuple = ( 0.80, 0.20, 0.70, 0.15, 0.60)

    latent_dim:        int   = 32
    hidden:            int   = 64
    depth:             int   = 3
    activation:        str   = "gelu"
    use_base_flux:     bool  = False
    base_flux_weight:  float = 0.50
    latent_flux_scale: float = 0.25

    epochs:       int   = 1000
    lr:           float = 1e-3
    weight_decay: float = 1e-6
    grad_clip:    float = 1.0
    seed:         int   = 0

    device:   str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "results_fluxgnn_multijump"


# ============================================================
# 7) Grid / dt
# ============================================================
def build_grid(cfg):
    x  = np.linspace(cfg.x_min, cfg.x_max, cfg.nx, dtype=np.float32)
    dx = float(x[1] - x[0])
    return x, dx

def build_fixed_dt(cfg, dx):
    return cfg.cfl * dx / 1.0


# ============================================================
# 8) Dataset — 2-jump + 3-jump ICs
# ============================================================
def build_train_dataset(cfg, x, dx, dt):
    u0_list, y_list = [], []
    times_ref = None
    n2, n3 = 0, 0

    for jumps, values in cfg.train_ics:
        n = len(jumps)
        u0 = make_pwc_ic(x, jumps, values)
        times, U = solve_truth_fixed_dt(u0, dx=dx, dt=dt, t_max=cfg.t_max, bc=cfg.boundary)
        if times_ref is None:
            times_ref = times
        u0_list.append(u0)
        y_list.append(U.astype(np.float32))
        if n == 2: n2 += 1
        if n == 3: n3 += 1

    print(f"  Training ICs: {len(u0_list)} total  ({n2} two-jump, {n3} three-jump)")
    u0_train = torch.from_numpy(np.stack(u0_list, axis=0))
    y_train  = torch.from_numpy(np.stack(y_list,  axis=0))
    return u0_train, y_train, times_ref


# ============================================================
# 9) Training
# ============================================================
def train_model(cfg, model, u0_train, y_train, dt, dx):
    model.train()
    opt      = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    u0_train = u0_train.to(cfg.device)
    y_train  = y_train.to(cfg.device)
    n_steps  = y_train.shape[1]

    loss_hist = []
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
        if epoch == 1 or epoch % 200 == 0:
            print(f"  epoch {epoch:5d} | MSE = {loss.item():.8e}")
    print(f"  Done in {time.time() - t0:.1f}s")
    return loss_hist


# ============================================================
# 10) Evaluation
# ============================================================
def conservation_error(u_hist, dx):
    integrals = u_hist.sum(axis=-1) * dx
    return float(np.max(np.abs(integrals - integrals[0])))


def evaluate(label, u0, x, model, cfg, dx, dt):
    times, U_true = solve_truth_fixed_dt(u0, dx=dx, dt=dt, t_max=cfg.t_max, bc=cfg.boundary)
    U_true = U_true.astype(np.float32)

    model.eval()
    with torch.no_grad():
        u0_t   = torch.from_numpy(u0[None, :]).to(cfg.device)
        U_pred = model.rollout(u0_t, dt=dt, dx=dx, n_steps=len(times),
                               boundary=cfg.boundary)[0].cpu().numpy()

    nt     = min(U_true.shape[0], U_pred.shape[0])
    U_true = U_true[:nt];  U_pred = U_pred[:nt];  times = times[:nt]
    err    = U_pred - U_true

    mse  = float(np.mean(err ** 2))
    mae  = float(np.mean(np.abs(err)))
    linf = float(np.max(np.abs(err)))
    cons_pred = conservation_error(U_pred, dx)
    cons_ref  = conservation_error(U_true, dx)

    print(f"  MSE={mse:.4e}  MAE={mae:.4e}  Linf={linf:.4e}")
    print(f"  Conservation  pred={cons_pred:.4e}  Godunov={cons_ref:.4e}")

    return dict(label=label, x=x, u0=u0, times=times,
                U_true=U_true, U_pred=U_pred, err=err,
                mse=mse, mae=mae, linf=linf,
                cons_pred=cons_pred, cons_ref=cons_ref)


# ============================================================
# 11) Plotting
# ============================================================
def plot_loss(cfg, loss_hist):
    plt.figure(figsize=(7, 4))
    plt.plot(loss_hist); plt.yscale("log")
    plt.xlabel("Epoch"); plt.ylabel("Train MSE"); plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.save_dir, "training_loss.png"), dpi=160); plt.close()


def plot_spacetime(cfg, res, fname):
    x, times   = res["x"], res["times"]
    U_true, U_pred, err = res["U_true"], res["U_pred"], res["err"]
    nt     = U_true.shape[0]
    extent = [float(x[0]), float(x[-1]), times[0], times[nt - 1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(U_true, aspect="auto", extent=extent, origin="lower", cmap="viridis")
    axes[0].set_title("Truth (Godunov)"); axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(U_pred, aspect="auto", extent=extent, origin="lower", cmap="viridis")
    axes[1].set_title("FluxGNN prediction"); axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(np.abs(err), aspect="auto", extent=extent, origin="lower", cmap="hot", vmin=0)
    axes[2].set_title("|Prediction error|"); axes[2].set_xlabel("x"); axes[2].set_ylabel("t")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(res["label"], y=1.01, fontsize=10)
    plt.tight_layout()
    path = os.path.join(cfg.save_dir, fname)
    plt.savefig(path, dpi=160, bbox_inches="tight"); plt.close()
    print(f"Saved: {path}")


def plot_snapshots(cfg, res, fname):
    x, times   = res["x"], res["times"]
    U_true, U_pred = res["U_true"], res["U_pred"]
    nt     = U_true.shape[0]
    chosen = [0, nt // 4, nt // 2, nt - 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8)); axes = axes.ravel()
    for ax, k in zip(axes, chosen):
        ax.plot(x, U_true[k], label="Godunov truth", linewidth=2)
        ax.plot(x, U_pred[k], "--", label="FluxGNN pred", linewidth=2)
        ax.set_title(f"t = {times[k]:.3f}"); ax.set_xlabel("x"); ax.set_ylabel("u")
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.25)
    axes[0].legend()
    plt.suptitle(res["label"], fontsize=10)
    plt.tight_layout()
    path = os.path.join(cfg.save_dir, fname)
    plt.savefig(path, dpi=160); plt.close()
    print(f"Saved: {path}")


def plot_ic_annotated(cfg, x, u0, jumps, values, fname):
    """Plot IC with wave-type annotations at each interface."""
    plt.figure(figsize=(9, 3))
    plt.plot(x, u0, linewidth=2.5, color="steelblue")
    for i, xj in enumerate(jumps):
        wtype = "rarefaction" if values[i] > values[i + 1] else "shock"
        color = "crimson" if wtype == "shock" else "darkgreen"
        plt.axvline(xj, color=color, linestyle="--", alpha=0.6)
        plt.text(xj + 0.02, 0.93, wtype, fontsize=8, color=color,
                 rotation=90, va="top")
    plt.xlabel("x"); plt.ylabel("u(x, 0)")
    plt.title("Test IC: 4-jump (trained on 2- and 3-jump ICs)")
    plt.ylim(-0.05, 1.05); plt.grid(alpha=0.25)
    plt.tight_layout()
    path = os.path.join(cfg.save_dir, fname)
    plt.savefig(path, dpi=160); plt.close()
    print(f"Saved: {path}")


def plot_conservation(cfg, res, dx, fname):
    times, U_true, U_pred = res["times"], res["U_true"], res["U_pred"]
    nt = U_true.shape[0]
    plt.figure(figsize=(6, 3))
    plt.plot(times[:nt], U_true[:nt].sum(axis=-1) * dx, label="Godunov")
    plt.plot(times[:nt], U_pred[:nt].sum(axis=-1) * dx, "--", label="FluxGNN")
    plt.xlabel("t"); plt.ylabel(r"$\int u\,dx$")
    plt.title(f"Conservation — {res['label']}")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    path = os.path.join(cfg.save_dir, fname)
    plt.savefig(path, dpi=160); plt.close()
    print(f"Saved: {path}")


# ============================================================
# 12) Main
# ============================================================
def main():
    cfg = Config()
    os.makedirs(cfg.save_dir, exist_ok=True)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    x, dx = build_grid(cfg)
    dt    = build_fixed_dt(cfg, dx)

    print("=" * 60)
    print("FluxGNN: train on 2-jump + 3-jump ICs")
    print("         evaluate on 4-jump IC (out-of-distribution)")
    print("=" * 60)
    print(f"  device={cfg.device}  nx={cfg.nx}  dx={dx:.5f}  dt={dt:.5f}")
    print(f"  t_max={cfg.t_max}  epochs={cfg.epochs}")

    # ---- Dataset ----
    u0_train, y_train, times = build_train_dataset(cfg, x, dx, dt)
    print(f"  y_train: {tuple(y_train.shape)}  nt={len(times)}")

    # ---- Model ----
    model = FluxGNN1DLatent(
        latent_dim=cfg.latent_dim, hidden=cfg.hidden, depth=cfg.depth,
        activation=cfg.activation, use_base_flux=cfg.use_base_flux,
        base_flux_weight=cfg.base_flux_weight, latent_flux_scale=cfg.latent_flux_scale,
    ).to(cfg.device)

    with torch.no_grad():
        e = model.encoder_vec; d = model.decoder_vec()
        print(f"  Init d^T e = {torch.dot(d, e).item():.8f}  (must be 1.0)")
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # ---- Train ----
    print("\nTraining...")
    loss_hist = train_model(cfg, model, u0_train, y_train, dt, dx)
    with torch.no_grad():
        e = model.encoder_vec; d = model.decoder_vec()
        print(f"  Post-train d^T e = {torch.dot(d, e).item():.8f}  (must be 1.0)")
    torch.save(model.state_dict(), os.path.join(cfg.save_dir, "model.pt"))
    plot_loss(cfg, loss_hist)

    # ---- Test: 4-jump IC ----
    print("\n" + "=" * 60)
    print("Evaluating on 4-jump IC (never seen during training)")
    print("=" * 60)
    jumps  = list(cfg.test_jumps)
    values = list(cfg.test_values)
    for i, xj in enumerate(jumps):
        wtype = "rarefaction" if values[i] > values[i + 1] else "shock"
        print(f"  x={xj:+.2f}: {values[i]:.2f} -> {values[i+1]:.2f}  ({wtype})")

    u0_test = make_pwc_ic(x, jumps, values)
    res     = evaluate(
        "4-jump IC  [trained on 2-jump + 3-jump only]",
        u0_test, x, model, cfg, dx, dt,
    )

    plot_ic_annotated(cfg, x, u0_test, jumps, values, "test_4jump_ic.png")
    plot_spacetime(cfg, res, "test_4jump_spacetime.png")
    plot_snapshots(cfg, res, "test_4jump_snapshots.png")
    plot_conservation(cfg, res, dx, "test_4jump_conservation.png")

    # ---- Metrics ----
    with open(os.path.join(cfg.save_dir, "metrics.txt"), "w") as f:
        f.write("Trained on: 2-jump and 3-jump ICs\n\n")
        f.write(f"Test: {res['label']}\n")
        f.write(f"  MSE  = {res['mse']:.12e}\n")
        f.write(f"  MAE  = {res['mae']:.12e}\n")
        f.write(f"  Linf = {res['linf']:.12e}\n")
        f.write(f"  Conservation error (pred)    = {res['cons_pred']:.12e}\n")
        f.write(f"  Conservation error (Godunov) = {res['cons_ref']:.12e}\n")

    np.savez_compressed(
        os.path.join(cfg.save_dir, "results.npz"),
        x=x, u0=u0_test,
        times=res["times"], U_true=res["U_true"], U_pred=res["U_pred"],
    )

    print(f"\nAll outputs saved to: {cfg.save_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()

