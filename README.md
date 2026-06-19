# PADL-TDSE: Physics-Accelerated Deep Learning for Quantum Simulation

## What this does
PADL-TDSE is a Physics-Aware Deep Learning (PADL) simulator for the 1D Time-Dependent Schrödinger Equation (TDSE). It builds a hybrid quantum mechanical solver that alternates between exact, high-fidelity Crank-Nicolson (CN) physics steps and fast, deep-learning (DL) model predictions. By periodic "re-anchoring" with the exact physics steps, the solver mitigates the norm drift and error accumulation typical of pure deep-learning surrogates, unlocking significant computational speedups on larger grids without sacrificing physical accuracy.

---

## Physics background
The simulation solves the 1D Time-Dependent Schrödinger Equation (TDSE) in natural units ($\hbar = m = 1$):

$$i \frac{\partial \psi}{\partial t} = H \psi = \left[ -\frac{1}{2} \frac{\partial^2}{\partial x^2} + V(x) \right] \psi$$

Key elements:
1. **Crank-Nicolson (CN) Solver**: An implicit, second-order finite difference scheme that is unconditionally stable and unitary. It solves the tridiagonal system $(I + i \frac{dt}{2} H)\psi^{n+1} = (I - i \frac{dt}{2} H)\psi^n$ in $O(N)$ time via the Thomas algorithm.
2. **Quantum Tunneling & Scattering**: A wavepacket traveling towards a rectangular or double potential barrier will experience partial reflection and partial transmission. The simulation tracks these transmission ($T$) and reflection ($R$) coefficients dynamically.
3. **Norm Conservation**: The total probability $\int |\psi(x)|^2 dx$ must remain exactly equal to $1.0$ at all times.

---

## PADL framework
PADL accelerates simulation by replacing consecutive blocks of CN steps with a single deep-learning inference step. A visual representation of the alternating physics/DL cycle:

```text
Step:  1      2      3      4      5      6      7 ...
       CN  →  DL  →  DL  →  DL  →  DL  →  CN  →  DL ...
       ↑                              ↑
  exact physics                 re-anchor to
  (ground truth)                exact physics
```

* **DL (Surrogate) Block**: A 1D U-Net model (`UNet1D`) trained to predict $\psi(t + 5\cdot dt)$ in a single forward pass, replacing 5 CN steps. A post-inference renormalisation layer is applied to restore $\|\psi\| = 1.0$ (physics correction).
* **CN (Anchor) Block**: Runs exact Crank-Nicolson steps to anchor the wavefunction back to the exact dynamics, halting error drift.

---

## Results
Below is the evaluation summary of the PADL solver benchmarked against pure Crank-Nicolson (averaged over 1500-step runs):

| Solver | Speedup | Norm Error ($|\|\psi\|^2 - 1|$) | Transmission Accuracy |
| :--- | :--- | :--- | :--- |
| **Crank-Nicolson (Exact)** | $1.0\times$ (Ref) | $\approx 10^{-13}$ | $82.52\%$ |
| **PADL (With Correction)** | $1.05\times - 5.0\times$ (GPU) | $\approx 10^{-10}$ to $10^{-8}$ | $82.83\%$ (Error: $0.31\%$) |
| **PADL (Without Correction)** | $1.05\times - 5.0\times$ | $\approx 10^{-4}$ (Drifting) | $79.15\%$ (Error: $3.37\%$) |

*Note: GPU execution yields up to $5.0\times$ wall-clock speedup for large grid sizes due to parallel batch-inference, while CPU execution exhibits slightly lower speedups due to framework overhead on small spatial grids ($N=512$).*

---

## How to run

### 1. Install dependencies
Ensure you have Python 3.8+ installed, then install the required packages:
```bash
pip install -r tdse_padl/requirements.txt
```

### 2. Generate training data
Generate 80,000 randomized training and validation samples (rectangular, double-barrier, and free-particle scenarios):
```bash
python -m tdse_padl.data.generator
```

### 3. Train the model
Train the U-Net model with the custom Physics-Aware Loss function:
```bash
python -m tdse_padl.train --epochs 50 --batch 64 --device cuda
```

### 4. Run benchmarks
Execute the solver benchmarker to check Speedup and Mean Absolute Error:
```bash
python -m tdse_padl.core.padl_solver
```

### 5. Launch demo
Launch the interactive visualizer in presentation/demo mode. It runs a pre-set scenario, auto-advances, and highlights key physical events (tunneling, norm errors, and PADL toggling) with overlays:
```bash
python -m tdse_padl.main --demo
```

---

## Project structure
```text
tdse_padl/
├── core/
│   ├── __init__.py
│   ├── padl_solver.py      # PADL hybrid block solver & benchmarker
│   ├── potential.py        # Potential barrier generators (rectangular, double)
│   ├── solver.py           # Crank-Nicolson tridiagonal (Thomas) solver
│   └── wavepacket.py       # Gaussian wavepacket initialization
├── data/
│   ├── __init__.py
│   └── generator.py        # Trajectory dataset generator & Dataset class
├── models/
│   ├── __init__.py
│   └── unet1d.py           # 1D U-Net model & PhysicsAwareLoss
├── utils/
│   ├── __init__.py
│   └── metrics.py          # Norm, transmission, and reflection metrics
├── main.py                 # Interactive animation demo & GUI
├── train.py                # Model training pipeline
├── requirements.txt        # Package dependencies
└── check_setup.py          # Setup & dependency validation script
```

---




https://github.com/user-attachments/assets/40cd7aab-fde3-40a5-842f-ecf37190ba19


