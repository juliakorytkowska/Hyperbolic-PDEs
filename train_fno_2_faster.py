import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import time

from gudonov import solve_fvm  # your solver (Godunov/FVM)

import os

def make_fno_input_from_u0(u0: np.ndarray, x_v: np.ndarray, t_v: np.ndarray) -> np.ndarray:
    """
    u0: (nx,)
    returns: (1, nx, nt, 3) float32
    """
    nx = len(x_v)
    nt = len(t_v)
    X, T = np.meshgrid(x_v, t_v, indexing="ij")          # (nx, nt)
    u0_rep = np.repeat(u0[:, None], nt, axis=1)          # (nx, nt)
    inp = np.stack([u0_rep, X, T], axis=-1)[None, ...]   # (1, nx, nt, 3)
    return inp.astype(np.float32)
def make_sinusoidal_piecewise_u0(x, mean=0.55, amp=0.28, waves=2.0, n_levels=5):
    u = mean + amp * np.sin(2.0 * np.pi * waves * (x - x.min()) / (x.max() - x.min()))
    u = np.clip(u, 0.02, 0.98)

    levels = np.linspace(u.min(), u.max(), n_levels)
    idx = np.abs(u[:, None] - levels[None, :]).argmin(axis=1)
    return levels[idx].astype(np.float32)

def save_dataset_npz(path, x, y, x_v, t_v):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        x=x.numpy(),
        y=y.numpy(),
        x_v=x_v,
        t_v=t_v
    )

def load_dataset_npz(path):
    data = np.load(path)
    x = torch.from_numpy(data["x"]).float()
    y = torch.from_numpy(data["y"]).float()
    x_v = data["x_v"].astype(np.float32)
    t_v = data["t_v"].astype(np.float32)
    return x, y, x_v, t_v

# ============================================================
# 1) Spectral Convolution 2D (paper-style: keep ± low x-modes)
# ============================================================
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # x-modes (two-sided)
        self.modes2 = modes2  # t-modes (one-sided due to rfft)

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale
            * torch.complex(
                torch.randn(in_channels, out_channels, modes1, modes2),
                torch.randn(in_channels, out_channels, modes1, modes2),
            )
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.complex(
                torch.randn(in_channels, out_channels, modes1, modes2),
                torch.randn(in_channels, out_channels, modes1, modes2),
            )
        )

    def forward(self, x):
        # x: (B, C, nx, nt)
        B = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-2)

        out_ft = torch.zeros(
            B,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        # low positive x modes
        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        # low negative x modes
        out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )

        x = torch.fft.irfft(out_ft, n=x.size(-2), dim=-2)
        return x
class SpectralConv1dX(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.complex(
                torch.randn(in_channels, out_channels, modes1),
                torch.randn(in_channels, out_channels, modes1),
            )
        )
        self.weights2 = nn.Parameter(
            scale * torch.complex(
                torch.randn(in_channels, out_channels, modes1),
                torch.randn(in_channels, out_channels, modes1),
            )
        )

    def forward(self, x):
        # x: (B, C, nx, nt)

        B, C, nx, nt = x.shape

        # FFT only in space
        x_ft = torch.fft.fft(x, dim=-2)   # (B, C, nx, nt), complex

        out_ft = torch.zeros(
            B, self.out_channels, nx, nt,
            dtype=torch.cfloat,
            device=x.device
        )

        # low positive spatial modes
        out_ft[:, :, :self.modes1, :] = torch.einsum(
            "bcmt,com->bomt",
            x_ft[:, :, :self.modes1, :],
            self.weights1
        )

        # low negative spatial modes
        out_ft[:, :, -self.modes1:, :] = torch.einsum(
            "bcmt,com->bomt",
            x_ft[:, :, -self.modes1:, :],
            self.weights2
        )

        # inverse FFT only in space
        x = torch.fft.ifft(out_ft, dim=-2).real   # (B, out_channels, nx, nt)
        return x

# ============================================================
# 2) Space–time FNO
# ============================================================
class FNO_SpaceTime(nn.Module):
    def __init__(self, modes1=24, width=64, layers=4):
        super().__init__()
        self.fc0 = nn.Linear(3, width)

        self.convs = nn.ModuleList(
            [SpectralConv1dX(width, width, modes1) for _ in range(layers)]
        )
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.Identity() for _ in range(layers)])

        self.fc1 = nn.Linear(width, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (B, nx, nt, 3)
        x = self.fc0(x)            # (B, nx, nt, width)
        x = x.permute(0, 3, 1, 2)  # (B, width, nx, nt)

        for conv, w, norm in zip(self.convs, self.ws, self.norms):
            x = norm(conv(x) + w(x))
            x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)   # (B, nx, nt, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x).squeeze(-1) # (B, nx, nt)
        return x


# ============================================================
# 3) Data generation: multi-jump piecewise-constant ICs
# ============================================================
def generate_pde_data_multi_jump(
    n_samples=600,
    nx=256,
    nt_out=100,
    t_max=1.0,
    seed=0,
    max_jumps=4,
    cfl=0.3,
    bc="copy",
):
    rng = np.random.default_rng(seed)
    print(f"Generating {n_samples} multi-jump samples (seed={seed}, max_jumps={max_jumps})")

    x_v = np.linspace(-1.0, 1.0, nx).astype(np.float32)
    t_v = np.linspace(0.0, t_max, nt_out).astype(np.float32)
    X, T = np.meshgrid(x_v, t_v, indexing="ij")  # (nx, nt)

    inputs = np.zeros((n_samples, nx, nt_out, 3), dtype=np.float32)
    targets = np.zeros((n_samples, nx, nt_out), dtype=np.float32)

    for i in range(n_samples):
        # number of jumps: 1..max_jumps
        n_jumps = int(rng.integers(1, max_jumps + 1))

        # jump locations (sorted)
        jump_locations = rng.uniform(-0.8, 0.8, size=n_jumps)
        jump_locations.sort()

        # plateau values (n_jumps+1)
        values = rng.uniform(0.1, 0.9, size=n_jumps + 1).astype(np.float32)

        # enforce minimum difference between adjacent plateaus
        for k in range(1, len(values)):
            while abs(values[k] - values[k - 1]) < 0.05:
                values[k] = rng.uniform(0.1, 0.9)

        # build piecewise constant u0
        u0 = np.zeros_like(x_v, dtype=np.float32)
        edges = np.concatenate(([-1.0], jump_locations, [1.0])).astype(np.float32)

        for k in range(len(values)):
            left = edges[k]
            right = edges[k + 1]
            if k < len(values) - 1:
                mask = (x_v >= left) & (x_v < right)
            else:
                mask = (x_v >= left) & (x_v <= right)  # include x=1.0
            u0[mask] = values[k]

        # solve PDE -> u_hist expected shape (nt_out, nx) (then transpose)
        _, _, u_hist = solve_fvm(u0, nt_out=nt_out, t_max=t_max, cfl=cfl, bc=bc)

        # build model input (u0 replicated over time + coords)
        u0_rep = np.repeat(u0[:, None], nt_out, axis=1)  # (nx, nt)
        inputs[i] = np.stack([u0_rep, X, T], axis=-1)    # (nx, nt, 3)
        targets[i] = u_hist.T                             # (nx, nt)

    return torch.from_numpy(inputs), torch.from_numpy(targets), x_v, t_v


# ============================================================
# 4) Train + Evaluate (train on multi-jump, test on unseen multi-jump)
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on", device)
    torch.backends.cudnn.benchmark = True

    # hyperparams
    nx = 256
    nt_out = 100
    t_max = 1.0
    batch_size = 20
    epochs = 300
    lr = 1e-3
    cfl = 0.3
    max_jumps = 4

    # --- DATA (train/test) ---
    train_path = f"cache/lwr_mj_train_N600_nx{nx}_nt{nt_out}_T{t_max}_J{max_jumps}_CFL{cfl}_seed0.npz"
    test_path  = f"cache/lwr_mj_test_N600_nx{nx}_nt{nt_out}_T{t_max}_J{max_jumps}_CFL{cfl}_seed1234.npz"
    if os.path.exists(train_path):
        print("Loading cached TRAIN set...")
        x_train, y_train, x_v, t_v = load_dataset_npz(train_path)
    else:
        x_train, y_train, x_v, t_v = generate_pde_data_multi_jump(
            n_samples=600, nx=256, nt_out=100, seed=0, max_jumps=max_jumps, cfl=cfl
        )
        save_dataset_npz(train_path, x_train, y_train, x_v, t_v)

    if os.path.exists(test_path):
        print("Loading cached TEST set...")
        x_test, y_test, _, _ = load_dataset_npz(test_path)
    else:
        x_test, y_test, _, _ = generate_pde_data_multi_jump(
            n_samples=600, nx=256, nt_out=100, seed=1234, max_jumps=max_jumps, cfl=cfl
        )
        save_dataset_npz(test_path, x_test, y_test, x_v, t_v)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )



    def count_jumps(u0, thr=1e-3):
        return int(np.sum(np.abs(np.diff(u0)) > thr))

    jump_counts = []
    for i in range(x_test.shape[0]):
        u0_tmp = x_test[i, :, 0, 0].numpy()
        jump_counts.append(count_jumps(u0_tmp))

    jump_counts = np.array(jump_counts)

    print("Min jumps:", jump_counts.min())
    print("Max jumps:", jump_counts.max())
    print("Mean jumps:", jump_counts.mean())
    print("Counts:", np.bincount(jump_counts))
    vals, cnts = np.unique(jump_counts, return_counts=True)
    print("Jump histogram:")
    for v, c in zip(vals, cnts):
        print(f"  jumps={v}: {c}")

    # --- MODEL ---
    model = FNO_SpaceTime(modes1=24, width=64, layers=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # --- TRAIN ---
    start_time = time.time()
    for epoch in range(epochs + 1):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)

            # MSE + small L1 helps shock sharpness a bit
            loss = F.mse_loss(pred, yb) + 1e-2 * F.l1_loss(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if epoch % 50 == 0:
            avg = total_loss / len(train_loader)
            print(f"Epoch {epoch:03d} | loss={avg:.6e} | lr={scheduler.get_last_lr()[0]:.2e}")

    print(f"Training Complete in {(time.time()-start_time)/60:.2f} minutes.")

    # --- SINUSOIDAL MANY-SHOCK TEST (zero-shot) ---
    u0_sin = make_sinusoidal_piecewise_u0(x_v, mean=0.55, amp=0.28, waves=2.0, n_levels=5)

    # Godunov truth on same grid/time
    _, _, u_hist = solve_fvm(u0_sin, nt_out=nt_out, t_max=t_max, cfl=cfl, bc="copy")
    truth_sin = u_hist.T  # (nx, nt)

    # FNO prediction
    sin_inp = make_fno_input_from_u0(u0_sin, x_v, t_v)
    with torch.no_grad():
        pred_sin = model(torch.from_numpy(sin_inp).to(device)).cpu().numpy()[0]  # (nx, nt)

    err_sin = np.abs(truth_sin - pred_sin)

    # Plot
    vmin = min(truth_sin.min(), pred_sin.min())
    vmax = max(truth_sin.max(), pred_sin.max())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    im0 = axes[0].pcolormesh(x_v, t_v, pred_sin.T, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[0].set_title("FNO Prediction (sinusoidal steps)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("t")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(x_v, t_v, truth_sin.T, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    axes[1].set_title("Godunov Truth (sinusoidal steps)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("t")
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(x_v, t_v, err_sin.T, shading="auto", cmap="magma")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("t")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig("fno_vs_godunov_sinusoidal_steps2.png", dpi=200)
    plt.show()

    # Metrics
    mse = np.mean((pred_sin - truth_sin)**2)
    rel_l2 = np.linalg.norm((pred_sin - truth_sin).ravel()) / (np.linalg.norm(truth_sin.ravel()) + 1e-12)
    print(f"[SIN TEST] MSE={mse:.3e}  relL2={rel_l2:.2%}")





if __name__ == "__main__":
    main()