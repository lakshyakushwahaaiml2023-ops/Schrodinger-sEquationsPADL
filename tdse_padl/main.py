"""
main.py
=======
Interactive, real-time matplotlib animation demonstrating the PADL hybrid solver
versus the exact Crank-Nicolson solver.

Controls:
  - V0 Slider: Adjust potential barrier height (50 to 800)
  - k0 Slider: Adjust wavepacket momentum (20 to 100)
  - Reset Button: Restart the simulation
  - Toggle PADL Button: Enable/disable the PADL deep learning surrogate model

Run:
  python -m tdse_padl.main
  python -m tdse_padl.main --demo   # Presentation mode
"""

import os
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button

from tdse_padl.core.solver import CrankNicolsonSolver
from tdse_padl.core.wavepacket import gaussian_wavepacket
from tdse_padl.core.potential import rectangular_barrier
from tdse_padl.core.padl_solver import PADLSolver
from tdse_padl.models.unet1d import UNet1D
from tdse_padl.utils import norm, transmission, reflection

class TDSEVisualizer:
    def __init__(self, N=512, L=1.0, dt=2e-5, demo_mode=False):
        self.N = N
        self.L = L
        self.dt = dt
        self.dx = L / N
        self.x = np.linspace(0.0, L, N, endpoint=False)

        # Simulation settings
        self.k0 = 50.0
        self.V0 = 200.0
        self.padl_active = True
        self.demo_mode = demo_mode
        self.demo_started = False
        
        # Norm and step trackers
        self.master_step_idx = 0
        self.step_idx = 0
        self.steps_history = []
        self.norm_exact_history = []
        self.norm_padl_history = []
        
        # Timing trackers for live speedup
        self.time_cn_total = 0.0
        self.time_padl_total = 0.0

        # Load trained neural network
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ckpt_path = 'checkpoints/best.pt'
        if os.path.exists(ckpt_path):
            print(f"[Visualizer] Loading model from {ckpt_path} on {self.device}...")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            self.model = UNet1D().to(self.device)
            self.model.load_state_dict(ckpt['model_state'])
            self.model.eval()
        else:
            print(f"[Visualizer] Warning: Checkpoint {ckpt_path} not found. Running CN only.")
            self.model = None
            self.padl_active = False

        # Set up figure layout
        self.fig = plt.figure(figsize=(12, 9))
        
        gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1.5], bottom=0.22, top=0.92, wspace=0.25, hspace=0.35)
        self.ax_exact = self.fig.add_subplot(gs[0, 0])
        self.ax_padl = self.fig.add_subplot(gs[0, 1])
        self.ax_norm = self.fig.add_subplot(gs[1, :])

        # Dual y-axes for potentials (amber/orange)
        self.ax_pot_exact = self.ax_exact.twinx()
        self.ax_pot_padl = self.ax_padl.twinx()

        # Design colors
        self.color_cn = '#1f77b4'       # Deep blue
        self.color_padl = '#ff1493'     # Pink/Magenta
        self.color_pot = '#e69f00'      # Amber

        # Initialize plot lines
        self.setup_axes()
        self.reset_simulation()

        # Create widgets
        self.setup_widgets()

        # Demo overlay and event bindings
        if self.demo_mode:
            # Connect key press to listen for spacebar
            self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
            
            # Figure-level text overlay for demo announcements
            self.demo_overlay = self.fig.text(
                0.5, 0.5,
                "PADL-TDSE: Physics-Accelerated Quantum Simulation\n\nPress SPACE to begin",
                ha='center', va='center', fontsize=16, fontweight='bold', color='white',
                bbox=dict(facecolor='#ff1493', alpha=0.9, boxstyle='round,pad=1.0', edgecolor='none')
            )
            self.demo_overlay.set_visible(True)

    def setup_axes(self):
        # Exact panel setup
        self.ax_exact.set_title("Exact CN solver", fontsize=11, fontweight='bold')
        self.ax_exact.set_ylabel(r"Probability Density $|\psi(x)|^2$", color=self.color_cn)
        self.ax_exact.set_xlim(0, self.L)
        self.ax_exact.set_ylim(-0.5, 12.0)
        self.ax_exact.grid(True, linestyle='--', alpha=0.5)

        # PADL panel setup
        self.ax_padl.set_title("PADL (4/5 predicted)" if self.padl_active else "Exact CN (reference)", fontsize=11, fontweight='bold')
        self.ax_padl.set_ylabel(r"Probability Density $|\psi(x)|^2$", color=self.color_cn)
        self.ax_padl.set_xlim(0, self.L)
        self.ax_padl.set_ylim(-0.5, 12.0)
        self.ax_padl.grid(True, linestyle='--', alpha=0.5)

        # Potential twin axes setup (amber)
        for ax_pot in [self.ax_pot_exact, self.ax_pot_padl]:
            ax_pot.set_ylabel("Potential V(x)", color=self.color_pot)
            ax_pot.tick_params(axis='y', labelcolor=self.color_pot)

        # Norm panel setup
        self.ax_norm.set_title("Norm Conservation History", fontsize=10, fontweight='bold')
        self.ax_norm.set_xlabel("Time step")
        self.ax_norm.set_ylabel(r"Norm $\int |\psi|^2 dx$")
        self.ax_norm.set_xlim(0, 100)
        self.ax_norm.set_ylim(0.98, 1.02)
        self.ax_norm.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
        self.ax_norm.grid(True, linestyle='--', alpha=0.5)

        # Plot artist initializations
        self.line_exact, = self.ax_exact.plot(self.x, np.zeros(self.N), color=self.color_cn, lw=2, label=r"$|\psi|^2$")
        self.line_padl, = self.ax_padl.plot(self.x, np.zeros(self.N), color=self.color_padl if self.padl_active else self.color_cn, lw=2, label=r"$|\psi|^2$")
        
        self.line_pot_exact, = self.ax_pot_exact.plot(self.x, np.zeros(self.N), color=self.color_pot, lw=1.5, alpha=0.8, label="V(x)")
        self.line_pot_padl, = self.ax_pot_padl.plot(self.x, np.zeros(self.N), color=self.color_pot, lw=1.5, alpha=0.8, label="V(x)")

        self.line_norm_exact, = self.ax_norm.plot([], [], color=self.color_cn, lw=1.8, label="CN (Exact)")
        self.line_norm_padl, = self.ax_norm.plot([], [], color=self.color_padl, lw=1.8, label="PADL (Hybrid)")
        self.ax_norm.legend(loc='lower left')

        # Live stats header text (above plots)
        self.stats_text = self.ax_exact.text(0.5, 0.95, "", transform=self.fig.transFigure,
                                             ha='center', fontsize=12, fontweight='bold',
                                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    def update_shading_and_titles(self):
        if self.padl_active:
            self.ax_padl.set_facecolor((1.0, 0.0, 0.5, 0.03)) # light pink
            self.ax_padl.set_title("PADL (4/5 predicted)", fontsize=11, fontweight='bold')
            self.line_padl.set_color(self.color_padl)
            self.line_norm_padl.set_color(self.color_padl)
        else:
            self.ax_padl.set_facecolor('#ffffff') # white
            self.ax_padl.set_title("Exact CN (reference)", fontsize=11, fontweight='bold')
            self.line_padl.set_color(self.color_cn)
            self.line_norm_padl.set_color(self.color_cn)
        self.fig.canvas.draw_idle()

    def reset_simulation(self):
        # Update potential barrier
        self.V = rectangular_barrier(self.N, self.L, center=0.6, width=0.05, height=self.V0)

        # Re-initialize solvers
        self.cn_solver = CrankNicolsonSolver(N=self.N, L=self.L, dt=self.dt, V=self.V)
        if self.model is not None:
            self.padl_solver = PADLSolver(
                cn_solver=self.cn_solver,
                model=self.model,
                device=self.device,
                physics_interval=5,
                model_skip=10
            )
        else:
            self.padl_solver = None
            self.padl_active = False

        # Re-initialize wavefunctions
        self.psi_exact = gaussian_wavepacket(self.N, self.L, x0=0.25, k0=self.k0, sigma=0.05)
        self.psi_padl = self.psi_exact.copy()

        # Clear histories and step counts
        self.master_step_idx = 0
        self.step_idx = 0
        self.steps_history.clear()
        self.norm_exact_history.clear()
        self.norm_padl_history.clear()
        
        self.time_cn_total = 0.0
        self.time_padl_total = 0.0

        # Update potential curves and axis limits
        self.line_pot_exact.set_ydata(self.V)
        self.line_pot_padl.set_ydata(self.V)
        
        self.ax_pot_exact.set_ylim(-0.1 * self.V0, 1.2 * self.V0)
        self.ax_pot_padl.set_ylim(-0.1 * self.V0, 1.2 * self.V0)

        # Set panel styles based on mode
        self.update_shading_and_titles()

        # Update plot line data
        self.line_exact.set_ydata(np.abs(self.psi_exact)**2)
        self.line_padl.set_ydata(np.abs(self.psi_padl)**2)
        
        self.line_norm_exact.set_data([], [])
        self.line_norm_padl.set_data([], [])
        self.ax_norm.set_xlim(0, 100)

        # Force render
        self.fig.canvas.draw_idle()

    def setup_widgets(self):
        # Create axes for widgets in the bottom reserved area
        ax_slide_V0 = self.fig.add_axes([0.15, 0.13, 0.30, 0.03])
        ax_slide_k0 = self.fig.add_axes([0.15, 0.07, 0.30, 0.03])
        ax_btn_reset = self.fig.add_axes([0.55, 0.09, 0.15, 0.05])
        ax_btn_toggle = self.fig.add_axes([0.73, 0.09, 0.20, 0.05])

        # Sliders
        self.slider_V0 = Slider(ax_slide_V0, "V₀ (barrier height)", 50.0, 800.0, valinit=self.V0, valstep=10.0, color=self.color_pot)
        self.slider_k0 = Slider(ax_slide_k0, "k₀ (momentum)", 20.0, 100.0, valinit=self.k0, valstep=1.0, color=self.color_cn)

        # Buttons
        self.btn_reset = Button(ax_btn_reset, "Reset", hovercolor='#e0e0e0')
        self.btn_toggle = Button(ax_btn_toggle, "Toggle PADL on/off", hovercolor='#ffc0cb')

        # Connect events
        self.slider_V0.on_changed(self.on_slider_V0_changed)
        self.slider_k0.on_changed(self.on_slider_k0_changed)
        self.btn_reset.on_clicked(self.on_reset_clicked)
        self.btn_toggle.on_clicked(self.on_toggle_clicked)

    def on_slider_V0_changed(self, val):
        self.V0 = val
        self.reset_simulation()

    def on_slider_k0_changed(self, val):
        self.k0 = val
        self.reset_simulation()

    def on_reset_clicked(self, event):
        self.reset_simulation()

    def on_toggle_clicked(self, event):
        if self.model is None:
            print("[Visualizer] Cannot toggle PADL: checkpoint not loaded.")
            return
        self.padl_active = not self.padl_active
        self.reset_simulation()

    # --- Demo Mode helpers ---
    def start_demo(self):
        if self.demo_started:
            return
        self.demo_started = True
        self.demo_overlay.set_visible(False)
        self.fig.canvas.draw_idle()
        if hasattr(self, 'anim') and self.anim is not None:
            self.anim.event_source.start()

    def on_key_press(self, event):
        if event.key == ' ':
            self.start_demo()

    def show_demo_pause(self, text, duration=2000):
        # Pause animation
        if hasattr(self, 'anim') and self.anim is not None:
            self.anim.event_source.stop()
        
        # Display announcement overlay
        self.demo_overlay.set_text(text)
        self.demo_overlay.set_visible(True)
        self.fig.canvas.draw()
        
        # Set a non-blocking one-shot timer to resume
        timer = self.fig.canvas.new_timer(interval=duration)
        def resume():
            timer.stop()
            self.demo_overlay.set_visible(False)
            self.fig.canvas.draw_idle()
            if hasattr(self, 'anim') and self.anim is not None:
                self.anim.event_source.start()
        
        timer.add_callback(resume)
        timer.start()

    def update_frame(self, frame):
        # Handle start delay in demo mode (pauses on first frame if not started)
        if self.demo_mode and not self.demo_started:
            return (self.line_exact, self.line_padl, self.line_norm_exact, self.line_norm_padl, self.stats_text)

        # 1. Advance exact Crank-Nicolson (10 steps)
        t_start = time.perf_counter()
        self.psi_exact = self.cn_solver.step_n(self.psi_exact, 10)
        self.time_cn_total += (time.perf_counter() - t_start)

        # 2. Advance right panel solver (10 steps)
        t_start = time.perf_counter()
        if self.padl_active and self.padl_solver is not None:
            # Advance 1 block of 10 steps = 10 steps
            block_idx = self.step_idx // 10
            self.psi_padl = self.padl_solver.step(self.psi_padl, self.V, block_idx)
            self.step_idx += 10
        else:
            # Advance using exact CN
            self.psi_padl = self.cn_solver.step_n(self.psi_padl, 10)
            self.step_idx += 10
        self.time_padl_total += (time.perf_counter() - t_start)

        self.master_step_idx += 10

        # Calculate metrics
        dx = self.dx
        barrier_start_idx = int((0.6 - 0.025) / dx)
        barrier_end_idx = int((0.6 + 0.025) / dx) + 1

        cn_norm_val = norm(self.psi_exact, dx)
        padl_norm_val = norm(self.psi_padl, dx)

        cn_T = transmission(self.psi_exact, dx, barrier_end_idx) * 100
        cn_R = reflection(self.psi_exact, dx, barrier_start_idx) * 100
        padl_T = transmission(self.psi_padl, dx, barrier_end_idx) * 100
        padl_R = reflection(self.psi_padl, dx, barrier_start_idx) * 100

        speedup = self.time_cn_total / self.time_padl_total if self.time_padl_total > 0 else 1.0

        # Append to histories
        self.steps_history.append(self.master_step_idx)
        self.norm_exact_history.append(cn_norm_val)
        self.norm_padl_history.append(padl_norm_val)

        # Keep history arrays from growing indefinitely (keep last 500 records)
        if len(self.steps_history) > 500:
            self.steps_history.pop(0)
            self.norm_exact_history.pop(0)
            self.norm_padl_history.pop(0)

        # Update lines
        self.line_exact.set_ydata(np.abs(self.psi_exact)**2)
        self.line_padl.set_ydata(np.abs(self.psi_padl)**2)

        self.line_norm_exact.set_data(self.steps_history, self.norm_exact_history)
        self.line_norm_padl.set_data(self.steps_history, self.norm_padl_history)
        
        # Adjust norm axis x-limits dynamically
        min_x = max(0, self.master_step_idx - 400)
        max_x = max(100, self.master_step_idx + 50)
        self.ax_norm.set_xlim(min_x, max_x)

        # Update live stats header text
        self.stats_text.set_text(
            f"CN: T={cn_T:.1f}% R={cn_R:.1f}% | "
            f"PADL: T={padl_T:.1f}% R={padl_R:.1f}% | "
            f"Speedup: {speedup:.1f}x | Step: {self.master_step_idx}"
        )

        # Handle presentation mode overlay triggers
        if self.demo_mode:
            if self.master_step_idx == 500:
                self.show_demo_pause("Wavepacket hits barrier → partial tunneling", duration=2000)
            elif self.master_step_idx == 1000:
                err = abs(padl_norm_val - 1.0)
                self.show_demo_pause(f"PADL norm error: {err:.3e} (physics conserved)", duration=2000)
            elif self.master_step_idx == 1500:
                self.show_demo_pause("Toggling PADL off: Running exact CN reference on right panel", duration=2000)
                self.padl_active = False
                self.update_shading_and_titles()
            elif self.master_step_idx == 1800:
                self.show_demo_pause("Toggling PADL back on: Resuming deep learning surrogate", duration=2000)
                self.padl_active = True
                self.update_shading_and_titles()

        return (self.line_exact, self.line_padl, self.line_norm_exact, self.line_norm_padl, self.stats_text)

def main():
    parser = argparse.ArgumentParser(description="PADL-TDSE Simulation Visualizer")
    parser.add_argument('--demo', action='store_true', help="Run in presentation/demo mode")
    args = parser.parse_args()

    viz = TDSEVisualizer(demo_mode=args.demo)
    
    # Run animation
    # 30 FPS target = ~33ms interval
    viz.anim = FuncAnimation(
        viz.fig,
        viz.update_frame,
        interval=33,
        blit=True,
        cache_frame_data=False
    )
    
    if args.demo:
        viz.anim.event_source.stop()
        viz.demo_timer = viz.fig.canvas.new_timer(interval=2000)
        viz.demo_timer.add_callback(viz.start_demo)
        viz.demo_timer.start()
    
    plt.show()

if __name__ == '__main__':
    main()