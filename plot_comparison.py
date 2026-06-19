"""
plot_comparison.py
===================
Compare pure Crank-Nicolson and PADL hybrid solver, generate plots of the
wavefunction evolution, and print metrics.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from tdse_padl.core.solver import CrankNicolsonSolver
from tdse_padl.core.wavepacket import gaussian_wavepacket
from tdse_padl.core.potential import rectangular_barrier, double_barrier
from tdse_padl.core.padl_solver import PADLSolver, Benchmarker
from tdse_padl.models.unet1d import UNet1D
from tdse_padl.utils import norm, transmission, reflection

def main():
    # 1. Setup grid and potential
    N = 512
    L = 1.0
    dt = 2e-5
    dx = L / N
    x = np.linspace(0.0, L, N, endpoint=False)

    # Use a double barrier to make the physics interesting (resonant tunnelling / scattering)
    c1, c2 = 0.52, 0.68
    width = 0.04
    height = 180.0
    V = double_barrier(N, L, c1, c2, width, height)

    # Initial Gaussian wavepacket
    psi0 = gaussian_wavepacket(N, L, x0=0.25, k0=50.0, sigma=0.05)

    # 2. Initialize solvers
    cn_solver = CrankNicolsonSolver(N=N, L=L, dt=dt, V=V)

    # Load trained model checkpoint
    ckpt_path = 'checkpoints/best.pt'
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint {ckpt_path} not found. Please train the model or ensure the checkpoint is present.")
        return

    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model = UNet1D()
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    padl_solver = PADLSolver(
        cn_solver=cn_solver,
        model=model,
        device=device,
        physics_interval=5,
        model_skip=10
    )

    # 3. Run evolution and collect snapshots
    n_steps = 750
    skip = padl_solver.model_skip
    n_blocks = n_steps // skip
    
    # Pure CN
    print("Running pure Crank-Nicolson...")
    cn_snapshots = [psi0.copy()]
    psi_cn = psi0.copy()
    t_start = time.perf_counter()
    for i in range(n_steps):
        psi_cn = cn_solver.step(psi_cn)
        if (i + 1) % 150 == 0:  # Save 5 snapshots during evolution
            cn_snapshots.append(psi_cn.copy())
    cn_time = time.perf_counter() - t_start

    # PADL Hybrid
    print("Running PADL Hybrid...")
    padl_solver.cn_calls = 0
    padl_solver.model_calls = 0
    t_start = time.perf_counter()
    # step_n returns snapshots every record_every blocks.
    # We want snapshots at the same times as CN, which is every 150 CN steps (15 blocks).
    padl_snapshots = padl_solver.step_n(
        psi0, V, n_blocks=n_blocks, record_every=15
    )
    padl_time = time.perf_counter() - t_start

    speedup = cn_time / padl_time if padl_time > 0 else 0
    print(f"Pure CN Time: {cn_time:.4f}s")
    print(f"PADL Time:    {padl_time:.4f}s")
    print(f"Speedup:      {speedup:.2f}x")

    # 4. Plot results
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Snapshot times in terms of CN steps
    times = [0, 150, 300, 450, 600, 750]
    
    # Plot Potential Barrier (shaded background)
    for ax in axes:
        ax.fill_between(x, 0, V / V.max() * 0.5, color='gray', alpha=0.15, label='Potential (scaled)')
        ax.set_ylabel('Probability Density $|\\psi(x)|^2$')
        ax.grid(True, linestyle='--', alpha=0.5)

    # Plot initial state
    axes[0].plot(x, np.abs(psi0)**2, 'k-', label='Initial Wavepacket')
    axes[0].set_title('Initial Wavepacket at t=0')
    axes[0].legend()

    # Plot middle state (around t = 450 steps)
    # cn_snapshots index for 450 steps is 3 (0, 150, 300, 450...)
    idx_mid = 3
    axes[1].plot(x, np.abs(cn_snapshots[idx_mid])**2, 'b-', label='Crank-Nicolson')
    axes[1].plot(x, np.abs(padl_snapshots[idx_mid])**2, 'r--', label='PADL Hybrid')
    axes[1].set_title(f'Wavepacket scattering at t = {idx_mid*150} steps')
    axes[1].legend()

    # Plot final state (t = 750 steps)
    idx_final = -1
    axes[2].plot(x, np.abs(cn_snapshots[idx_final])**2, 'b-', label='Crank-Nicolson')
    axes[2].plot(x, np.abs(padl_snapshots[idx_final])**2, 'r--', label='PADL Hybrid')
    axes[2].set_title(f'Final Wavepacket at t = 750 steps')
    axes[2].set_xlabel('Position x')
    axes[2].legend()

    plt.tight_layout()
    plot_path = 'checkpoints/comparison.png'
    plt.savefig(plot_path, dpi=200)
    print(f"Comparison plot saved to {plot_path}")

    # Calculate final transmission & reflection metrics
    barrier_mid = (c1 + c2) / 2.0
    barrier_end_idx = int((c2 + width/2) / dx) + 1
    barrier_start_idx = int((c1 - width/2) / dx)

    cn_final = cn_snapshots[-1]
    padl_final = padl_snapshots[-1]

    cn_T = transmission(cn_final, dx, barrier_end_idx)
    cn_R = reflection(cn_final, dx, barrier_start_idx)
    padl_T = transmission(padl_final, dx, barrier_end_idx)
    padl_R = reflection(padl_final, dx, barrier_start_idx)

    print("\n" + "="*40)
    print("METRICS COMPARISON")
    print("="*40)
    print(f"Norm (CN):         {norm(cn_final, dx):.6f}")
    print(f"Norm (PADL):       {norm(padl_final, dx):.6f}")
    print(f"Transmission (CN): {cn_T*100:.2f}%")
    print(f"Transmission (PADL): {padl_T*100:.2f}%")
    print(f"Reflection (CN):   {cn_R*100:.2f}%")
    print(f"Reflection (PADL): {padl_R*100:.2f}%")
    mae = np.mean(np.abs(cn_final - padl_final))
    print(f"Mean Absolute Error: {mae:.2e}")
    print("="*40)

if __name__ == '__main__':
    main()
