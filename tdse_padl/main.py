"""
main.py
=======
Entry point for the TDSE-PADL simulator.

This file is a placeholder that will be expanded as the PADL model training
pipeline is developed.  For now it demonstrates a complete end-to-end run:

    1.  Build a rectangular barrier potential.
    2.  Initialise a Gaussian wavepacket.
    3.  Evolve with the Crank-Nicolson solver.
    4.  Print physics diagnostics (norm, T, R).

Run
---
    python -m tdse_padl.main
    # or from the project root:
    python tdse_padl/main.py
"""

import numpy as np

from tdse_padl.core import (
    CrankNicolsonSolver,
    gaussian_wavepacket,
    rectangular_barrier,
)
from tdse_padl.utils import norm, transmission, reflection, mean_position, mean_momentum


def main() -> None:
    # ------------------------------------------------------------------
    # Grid parameters
    # ------------------------------------------------------------------
    N  = 512
    L  = 1.0
    dt = 1e-5
    dx = L / N
    x  = np.linspace(0.0, L, N, endpoint=False)

    # ------------------------------------------------------------------
    # Potential: single rectangular barrier
    # ------------------------------------------------------------------
    barrier_center = 0.6
    barrier_width  = 0.05
    barrier_height = 200.0
    V = rectangular_barrier(N, L, barrier_center, barrier_width, barrier_height)

    half               = barrier_width / 2.0
    barrier_start_idx  = int((barrier_center - half) / dx)
    barrier_end_idx    = int((barrier_center + half) / dx) + 1

    # ------------------------------------------------------------------
    # Initial wavepacket
    # ------------------------------------------------------------------
    psi = gaussian_wavepacket(N, L, x0=0.25, k0=50.0, sigma=0.05)

    print("=" * 55)
    print("  TDSE-PADL  |  Crank-Nicolson solver demonstration")
    print("=" * 55)
    print(f"  Grid:       N={N}, L={L}, dx={dx:.4e}")
    print(f"  Timestep:   dt={dt:.1e}")
    print(f"  Barrier:    centre={barrier_center}, width={barrier_width}, "
          f"height={barrier_height}")
    print(f"  Wavepacket: x0=0.25, k0=50, sigma=0.05")
    print()

    norm_0 = norm(psi, dx)
    mean_x0 = mean_position(psi, x, dx)
    mean_p0 = mean_momentum(psi, dx)
    print(f"  Initial norm:          {norm_0:.6f}")
    print(f"  Initial <x>:           {mean_x0:.4f}")
    print(f"  Initial <p>:           {mean_p0:.2f}")
    print()

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------
    solver  = CrankNicolsonSolver(N=N, L=L, dt=dt, V=V)
    n_steps = 2_000
    print(f"  Running {n_steps} CN steps ... ", end="", flush=True)
    psi = solver.step_n(psi, n_steps)
    print("done.")
    print()

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    norm_f   = norm(psi, dx)
    T        = transmission(psi, dx, barrier_end_idx)
    R        = reflection(psi, dx, barrier_start_idx)
    mean_xf  = mean_position(psi, x, dx)
    mean_pf  = mean_momentum(psi, dx)

    print(f"  Final norm:            {norm_f:.6f}")
    print(f"  Transmission:          {T * 100:.1f}%")
    print(f"  Reflection:            {R  * 100:.1f}%")
    print(f"  Final <x>:             {mean_xf:.4f}")
    print(f"  Final <p>:             {mean_pf:.2f}")
    print()
    conserved = abs(norm_f - norm_0) < 1e-4
    print(f"  Norm conserved (tol=1e-4): {conserved}")
    print("=" * 55)


if __name__ == "__main__":
    main()