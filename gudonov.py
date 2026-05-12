# gudonov.py
# ============================================================
# Godunov finite volume solver for:
#     u_t + (u(1-u))_x = 0
# with concave flux f(u)=u(1-u)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1) Godunov flux for concave f(u)=u(1-u)
# ============================================================
def godunov_flux(u_ext: np.ndarray) -> np.ndarray: 
    """ Compute Godunov numerical flux at interfaces for concave flux f(u)=u(1-u). 
    Input: u_ext : (nx+2,) array with 1 ghost cell on each side Output: fhat : 
    (nx+1,) array of interface fluxes fhat[i] is flux between u_ext[i] and 
    u_ext[i+1] """ 
    uL = u_ext[:-1] 
    uR = u_ext[1:] 
    
    def f(u): 
        return u * (1.0 - u) 
    
    fL = f(uL) 
    fR = f(uR) 
    
    fhat = np.empty_like(uL) 
    # For concave flux: 
    # # uL < uR => shock 
    # # uL > uR => rarefaction 
    shock = (uL < uR) 
    rare = ~shock 
    # Shock: speed s = (fR-fL)/(uR-uL) = 1 - (uL+uR) 
    s = 1.0 - (uL + uR) 
    fhat[shock] = np.where(s[shock] >= 0.0, fL[shock], fR[shock]) 
    # Rarefaction: take max of f on [uR, uL] 
    fhat_r = np.maximum(fL[rare], fR[rare]) 
    # If interval crosses maximizer u=0.5, use f(0.5)=0.25 
    crosses_half = (uR[rare] <= 0.5) & (0.5 <= uL[rare]) 
    fhat_r[crosses_half] = 0.25 
    fhat[rare] = fhat_r 
    return fhat

# ============================================================
# 2) CFL timestep: max |f'(u)| where f'(u)=1-2u
# ============================================================
def compute_time_step(u: np.ndarray, dx: float, cfl: float, t: float, t_max: float) -> float:
    max_speed = max(1e-8, np.max(np.abs(1.0 - 2.0 * u)))
    dt = cfl * dx / max_speed
    return min(dt, t_max - t)


# ============================================================
# 3) Ghost cells
#    Default: copy boundary values (zero-gradient-ish)
#    If you want periodic: set bc="periodic"
# ============================================================
def add_ghost_cells(u: np.ndarray, bc: str = "copy") -> np.ndarray:
    """
    bc:
      - "copy": u_ext[0]=u[0], u_ext[-1]=u[-1]
      - "periodic": u_ext[0]=u[-1], u_ext[-1]=u[0]
    """
    u_ext = np.empty(len(u) + 2, dtype=u.dtype)
    u_ext[1:-1] = u

    if bc == "periodic":
        u_ext[0] = u[-1]
        u_ext[-1] = u[0]
    else:
        u_ext[0] = u[0]
        u_ext[-1] = u[-1]

    return u_ext


# ============================================================
# 4) Finite Volume Solver (IMPORTANT: takes u0!)
# ============================================================
def solve_fvm(
    u0: np.ndarray,
    nt_out: int = 150,
    x_min: float = -1.0,
    x_max: float = 1.0,
    t_max: float = 1.0,
    cfl: float = 0.3,
    bc: str = "copy",
):
    """
    Solve u_t + (u(1-u))_x = 0 using Godunov FVM.

    Inputs:
      u0    : (nx,) initial condition on spatial grid
      nt_out: number of output time snapshots
      x_min,x_max: spatial domain
      t_max : final time
      cfl   : CFL number
      bc    : "copy" or "periodic"

    Returns:
      x     : (nx,)
      t_out : (nt_out,)
      u_hist: (nt_out, nx)  (time-first, like your original code)
    """
    u0 = np.asarray(u0, dtype=float)
    nx = len(u0)

    x = np.linspace(x_min, x_max, nx).astype(float)
    dx = x[1] - x[0]

    t_out = np.linspace(0.0, t_max, nt_out).astype(float)
    u_hist = np.zeros((nt_out, nx), dtype=float)

    u = u0.copy()
    u_hist[0] = u

    t = 0.0
    k = 1

    while k < nt_out:
        dt = compute_time_step(u, dx, cfl, t, t_max)

        u_ext = add_ghost_cells(u, bc=bc)
        fhat = godunov_flux(u_ext)  # (nx+1,)

        # Conservative update for nx cells
        u = u - (dt / dx) * (fhat[1:] - fhat[:-1])

        t += dt
        while k < nt_out and t >= t_out[k] - 1e-12:
            u_hist[k] = u
            k += 1

    return x, t_out, u_hist


# ============================================================
# 5) Demo run (ONLY runs if you execute gudonov.py directly)
# ============================================================
if __name__ == "__main__":
    nx = 200
    x = np.linspace(-1.0, 1.0, nx)

    # Example: single jump IC
    u0 = np.where(x < 0.0, 0.1, 0.3)

    x, t, u_hist = solve_fvm(u0, nt_out=150, t_max=1.0, cfl=0.3, bc="copy")
    print("Grid shapes:", x.shape, t.shape, u_hist.shape)

    plt.figure(figsize=(7, 4))
    plt.pcolormesh(x, t, u_hist, shading="auto", cmap="jet")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.title("u_t + (u(1-u))_x = 0 (Godunov FVM)")
    plt.colorbar(label="u")
    plt.tight_layout()
    plt.savefig("easy2.png", dpi=200)
    plt.show()