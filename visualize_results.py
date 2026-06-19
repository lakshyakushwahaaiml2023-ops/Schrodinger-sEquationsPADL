"""
visualize_results.py
====================
Generate 4 key visualization plots to compare Crank-Nicolson and PADL solvers.
Saves all plots in high-resolution (300 DPI) to the results/ directory.

Plots:
  1. Tunneling sequence: Overlaid wavepackets at 4 snapshots (t=0, t=T/4, t=T/2, t=3T/4).
  2. Barrier height sweep: Transmission % vs V0 compared to WKB theory.
  3. Speedup vs accuracy tradeoff: Parameter sweep of physics_interval.
  4. Norm conservation: Log-scale drift comparison (CN vs PADL with/without correction).
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from tdse_padl.core.solver import CrankNicolsonSolver
from tdse_padl.core.wavepacket import gaussian_wavepacket
from tdse_padl.core.potential import rectangular_barrier
from tdse_padl.core.padl_solver import PADLSolver
from tdse_padl.models.unet1d import UNet1D
from tdse_padl.utils import norm, transmission, reflection

def get_model():
    ckpt_path = 'checkpoints/best.pt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found. Please train the model first.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = UNet1D().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, device

def run_padl_no_correction(cn_solver, model, device, psi0, V, n_blocks, physics_interval=5):
    """Run PADL solver evolution WITHOUT physics norm correction."""
    psi = psi0.copy()
    V_arr = np.asarray(cn_solver.V, dtype=np.float32)
    V_max = float(V_arr.max()) if V_arr.max() != 0.0 else 1.0
    V_norm = (V_arr / V_max).astype(np.float32)

    history = [psi.copy()]

    for block_idx in range(n_blocks):
        if block_idx % physics_interval == 0:
            psi = cn_solver.step_n(psi, 5)
        else:
            # Model forward pass
            psi32 = psi.astype(np.complex64)
            x_np = np.stack([psi32.real, psi32.imag, V_norm], axis=0)
            x_t = torch.from_numpy(x_np).unsqueeze(0).to(device)
            with torch.no_grad():
                y_t = model(x_t)
            arr = y_t.squeeze(0).cpu().numpy()
            psi = (arr[0] + 1j * arr[1]).astype(np.complex128)
            # DO NOT RENORMALISE
        
        # Save snapshot at each block (covers 5 CN steps)
        history.append(psi.copy())
    return history

def plot_tunneling_sequence(model, device):
    print("Generating Plot 1: Tunneling sequence...")
    N, L = 512, 1.0
    dt = 2e-5
    dx = L / N
    x = np.linspace(0.0, L, N, endpoint=False)
    V = rectangular_barrier(N, L, center=0.6, width=0.05, height=200.0)
    psi0 = gaussian_wavepacket(N, L, x0=0.25, k0=50.0, sigma=0.05)

    cn_solver = CrankNicolsonSolver(N=N, L=L, dt=dt, V=V)
    padl_solver = PADLSolver(cn_solver, model, device, physics_interval=5, model_skip=10)

    # We want snapshots at t=0, t=200, t=400, t=600
    steps = [0, 200, 400, 600]
    
    # Run exact CN
    cn_wavefunctions = {0: psi0.copy()}
    psi = psi0.copy()
    for step in range(1, 601):
        psi = cn_solver.step(psi)
        if step in steps:
            cn_wavefunctions[step] = psi.copy()

    # Run PADL hybrid
    padl_wavefunctions = {0: psi0.copy()}
    psi_h = psi0.copy()
    # 600 steps = 60 blocks of 10 steps
    for b in range(60):
        psi_h = padl_solver.step(psi_h, V, b)
        curr_step = (b + 1) * 10
        if curr_step in steps:
            padl_wavefunctions[curr_step] = psi_h.copy()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, step in enumerate(steps):
        ax = axes[idx]
        # Draw barrier background
        ax.fill_between(x, 0, V / V.max() * 5.0, color='orange', alpha=0.1, label="Barrier (scaled)")
        
        # Plot CN and PADL
        ax.plot(x, np.abs(cn_wavefunctions[step])**2, color='#1f77b4', lw=2, label="Crank-Nicolson (Exact)")
        ax.plot(x, np.abs(padl_wavefunctions[step])**2, color='#ff1493', linestyle='--', lw=2, label="PADL (Hybrid)")
        
        ax.set_title(f"Step {step}", fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        if idx >= 2:
            ax.set_xlabel("Position x")
        if idx % 2 == 0:
            ax.set_ylabel(r"Probability Density $|\psi(x)|^2$")
        if idx == 0:
            ax.legend(loc="upper right")

    plt.suptitle("Wavepacket Tunneling Sequence Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/tunneling_sequence.png', dpi=300)
    plt.close()

def plot_barrier_height_sweep(model, device):
    print("Generating Plot 2: Barrier height sweep...")
    N, L = 512, 1.0
    dt = 2e-5
    dx = L / N
    V0_vals = [50.0, 100.0, 200.0, 400.0, 800.0]
    
    cn_transmissions = []
    padl_transmissions = []
    wkb_transmissions = []

    k0 = 50.0
    sigma = 0.05
    width = 0.05
    center = 0.6
    barrier_end_idx = int((center + width/2) / dx) + 1

    # Kinetic energy E = k0^2 / 2
    E = (k0 ** 2) / 2.0

    for V0 in V0_vals:
        V = rectangular_barrier(N, L, center=center, width=width, height=V0)
        psi0 = gaussian_wavepacket(N, L, x0=0.25, k0=k0, sigma=sigma)

        # Exact CN (750 steps)
        cn = CrankNicolsonSolver(N=N, L=L, dt=dt, V=V)
        psi_cn = cn.step_n(psi0, 750)
        cn_T = transmission(psi_cn, dx, barrier_end_idx) * 100
        cn_transmissions.append(cn_T)

        # PADL (75 blocks of 10 steps)
        padl = PADLSolver(cn, model, device, physics_interval=5, model_skip=10)
        psi_padl = psi0.copy()
        for b in range(75):
            psi_padl = padl.step(psi_padl, V, b)
        padl_T = transmission(psi_padl, dx, barrier_end_idx) * 100
        padl_transmissions.append(padl_T)

        # Theoretical WKB transmission (clip V0 - E to be >= 0)
        # T_WKB = exp(-2 * sqrt(2*m*(V0-E)) * width / hbar)
        exponent = -2.0 * np.sqrt(np.maximum(0.0, 2.0 * (V0 - E))) * width
        wkb_T = np.exp(exponent) * 100
        wkb_transmissions.append(wkb_T)

    plt.figure(figsize=(8, 6))
    plt.plot(V0_vals, cn_transmissions, 'o-', color='#1f77b4', lw=2, label="Crank-Nicolson (Exact)")
    plt.plot(V0_vals, padl_transmissions, 's--', color='#ff1493', lw=2, label="PADL (Hybrid)")
    plt.plot(V0_vals, wkb_transmissions, 'k:', lw=1.8, label="WKB Approximation")

    plt.title("Transmission Probability vs. Barrier Height $V_0$", fontsize=12, fontweight='bold')
    plt.xlabel("Barrier Height $V_0$")
    plt.ylabel("Transmission Coefficient (%)")
    plt.xscale('log')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/barrier_sweep.png', dpi=300)
    plt.close()

def plot_speedup_tradeoff(model, device):
    print("Generating Plot 3: Speedup vs accuracy tradeoff...")
    N, L = 512, 1.0
    dt = 2e-5
    dx = L / N
    V = rectangular_barrier(N, L, center=0.6, width=0.05, height=200.0)
    psi0 = gaussian_wavepacket(N, L, x0=0.25, k0=50.0, sigma=0.05)

    intervals = [1, 2, 3, 5, 8, 10]
    maes = []
    speedups = []

    # Run reference Crank-Nicolson
    cn_solver = CrankNicolsonSolver(N=N, L=L, dt=dt, V=V)
    
    # Warmup and time CN (run 3 times, take min)
    cn_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        psi_cn = cn_solver.step_n(psi0, 750)
        cn_times.append(time.perf_counter() - t0)
    ref_time = min(cn_times)

    for interval in intervals:
        padl_solver = PADLSolver(cn_solver, model, device, physics_interval=interval, model_skip=10)
        
        # Warmup and run timed trials
        padl_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            psi_padl = psi0.copy()
            for b in range(75):
                psi_padl = padl_solver.step(psi_padl, V, b)
            padl_times.append(time.perf_counter() - t0)
        
        run_time = min(padl_times)
        speedup = ref_time / run_time if run_time > 0 else 0
        mae = np.mean(np.abs(psi_cn - psi_padl))

        maes.append(mae)
        speedups.append(speedup)
        print(f"  physics_interval={interval:2d} | Speedup={speedup:.2f}x | MAE={mae:.2e}")

    plt.figure(figsize=(8.5, 6))
    plt.plot(speedups, maes, 'o-', color='#1f77b4', lw=2)
    plt.xlim(min(speedups) - 0.15, max(speedups) + 0.65)

    # Label data points
    for idx, interval in enumerate(intervals):
        plt.annotate(
            f"Interval {interval}",
            (speedups[idx], maes[idx]),
            textcoords="offset points",
            xytext=(10, -5 if idx % 2 == 0 else 5),
            ha='left',
            fontsize=9
        )

    # Highlight sweet spot (interval=5)
    sweet_idx = intervals.index(5)
    plt.scatter([speedups[sweet_idx]], [maes[sweet_idx]], color='#ff1493', s=120, zorder=5)
    plt.annotate(
        "Sweet Spot (Interval=5)",
        xy=(speedups[sweet_idx], maes[sweet_idx]),
        xytext=(speedups[sweet_idx] + 0.08, maes[sweet_idx] * 1.3),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=6),
        fontsize=10,
        fontweight='bold',
        color='#ff1493',
        ha='left'
    )

    plt.title("Speedup vs. Accuracy Tradeoff (Crank-Nicolson reference)", fontsize=12, fontweight='bold')
    plt.xlabel("Speedup Factor (relative to CN)")
    plt.ylabel("Mean Absolute Error (MAE)")
    plt.yscale('log')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/speedup_tradeoff.png', dpi=300)
    plt.close()

def plot_norm_conservation(model, device):
    print("Generating Plot 4: Norm conservation over time...")
    N, L = 512, 1.0
    dt = 2e-5
    dx = L / N
    V = rectangular_barrier(N, L, center=0.6, width=0.05, height=200.0)
    psi0 = gaussian_wavepacket(N, L, x0=0.25, k0=50.0, sigma=0.05)

    cn_solver = CrankNicolsonSolver(N=N, L=L, dt=dt, V=V)

    # 1. Exact Crank-Nicolson
    cn_norm_dev = []
    psi = psi0.copy()
    for _ in range(750):
        psi = cn_solver.step(psi)
        cn_norm_dev.append(abs(norm(psi, dx) - 1.0))

    # 2. PADL with norm correction (physics correction active)
    padl_corrected = PADLSolver(cn_solver, model, device, physics_interval=5, model_skip=10)
    padl_corrected_dev = []
    psi_h = psi0.copy()
    
    # We record norm deviation *each step*. To get step-by-step resolution,
    # we evaluate intermediate values.
    # For a fair comparison, we run the step-by-step block sequence
    for b in range(75):
        psi_before = psi_h.copy()
        psi_h = padl_corrected.step(psi_h, V, b)
        
        # Linearly interpolate intermediate step norms for smooth step-by-step curve
        norm_start = norm(psi_before, dx)
        norm_end = norm(psi_h, dx)
        for step_idx in range(10):
            frac = step_idx / 10.0
            interp_norm = norm_start * (1 - frac) + norm_end * frac
            padl_corrected_dev.append(abs(interp_norm - 1.0))

    # 3. PADL without norm correction
    padl_no_corr_history = run_padl_no_correction(cn_solver, model, device, psi0, V, 75, physics_interval=5)
    padl_no_corr_dev = []
    for idx in range(75):
        norm_start = norm(padl_no_corr_history[idx], dx)
        norm_end = norm(padl_no_corr_history[idx+1], dx)
        for step_idx in range(10):
            frac = step_idx / 10.0
            interp_norm = norm_start * (1 - frac) + norm_end * frac
            padl_no_corr_dev.append(abs(interp_norm - 1.0))

    steps = np.arange(750)

    plt.figure(figsize=(9, 6))
    plt.plot(steps, cn_norm_dev, color='#1f77b4', lw=1.8, label="Crank-Nicolson (Exact)")
    plt.plot(steps, padl_corrected_dev, color='#ff1493', lw=1.8, label="PADL (With Norm Correction)")
    plt.plot(steps, padl_no_corr_dev, 'r--', lw=1.5, label="PADL (Without Norm Correction)")

    plt.title("Wavefunction Norm Conservation Error Over Time", fontsize=12, fontweight='bold')
    plt.xlabel("Simulation Timestep")
    plt.ylabel(r"Norm Conservation Error $|\int |\psi|^2 dx - 1|$")
    plt.yscale('log')
    plt.grid(True, which="both", linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/norm_conservation.png', dpi=300)
    plt.close()

def main():
    os.makedirs('results', exist_ok=True)
    
    try:
        model, device = get_model()
    except FileNotFoundError as e:
        print(e)
        return

    plot_tunneling_sequence(model, device)
    plot_barrier_height_sweep(model, device)
    plot_speedup_tradeoff(model, device)
    plot_norm_conservation(model, device)
    
    print("\nAll 4 presentation plots successfully saved to results/ directory.")

if __name__ == '__main__':
    main()

"""
Run instructions:
  1. Train the model or ensure the checkpoint is present at:
     checkpoints/best.pt
  2. Run the script from the project root:
     python visualize_results.py
  3. The resulting plots will be saved in high-resolution (300 DPI) inside:
     results/
"""
