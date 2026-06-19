"""
diag.py
=======
Diagnostic script to test if the model is predicting identity (lazy model)
or true physical dynamics.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from tdse_padl.data import QuantumDataset
from tdse_padl.models.unet1d import UNet1D

def main():
    ckpt_path = 'checkpoints/best.pt'
    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint {ckpt_path} not found.")
        return

    print(f"Loading checkpoint {ckpt_path}...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = UNet1D().to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Load 100 validation samples
    val_path = 'data/val.h5'
    if not os.path.exists(val_path):
        print(f"Error: validation set {val_path} not found.")
        return
        
    dataset = QuantumDataset(val_path, augment=False, preload=True)
    n_samples = min(100, len(dataset))
    print(f"Running diagnostics on {n_samples} validation samples...")

    model_deltas = []
    true_deltas = []

    for idx in range(n_samples):
        x, y = dataset[idx]
        x_t = x.unsqueeze(0).to(device) # (1, 3, N)
        y_t = y.unsqueeze(0).to(device) # (1, 2, N)

        with torch.no_grad():
            pred_t = model(x_t) # (1, 2, N)

        # input_psi is channels 0 & 1 of input
        input_psi = x_t[:, :2, :]

        # Calculate model delta: mean |pred - input_psi|
        model_delta = torch.mean(torch.abs(pred_t - input_psi)).item()
        model_deltas.append(model_delta)

        # Calculate true delta: mean |target - input_psi|
        true_delta = torch.mean(torch.abs(y_t - input_psi)).item()
        true_deltas.append(true_delta)

    mean_model_delta = np.mean(model_deltas)
    mean_true_delta = np.mean(true_deltas)
    ratio = mean_model_delta / mean_true_delta if mean_true_delta > 0 else 0.0

    print("\n" + "="*40)
    print("DIAGNOSTIC RESULTS")
    print("="*40)
    print(f"Mean model delta : {mean_model_delta:.6e}")
    print(f"Mean true delta  : {mean_true_delta:.6e}")
    print(f"Delta Ratio      : {ratio:.4f}")
    if mean_model_delta < 1e-4:
        print("  --> [CONFIRMED LAZY] Model delta is below 1e-4. Model is predicting near-identity!")
    else:
        print("  --> Model delta is above 1e-4. Model is learning dynamics.")
    print("="*40)

    # Plot single example
    os.makedirs('results', exist_ok=True)
    x_ex, y_ex = dataset[0]
    x_ex_t = x_ex.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_ex_t = model(x_ex_t)

    x_grid = np.linspace(0.0, 1.0, len(x_ex[0]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_grid, x_ex[0].numpy(), 'b-', lw=1.8, label="Re(input)")
    plt.plot(x_grid, y_ex[0].numpy(), 'g--', lw=1.8, label="Re(target)")
    plt.plot(x_grid, pred_ex_t[0, 0].cpu().numpy(), 'r:', lw=2, label="Re(prediction)")
    
    plt.title("Wavefunction Prediction Check (Single Sample)", fontsize=12, fontweight='bold')
    plt.xlabel("Spatial Grid x")
    plt.ylabel("Real part Re(psi)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    diag_path = 'results/diag_single.png'
    plt.savefig(diag_path, dpi=200)
    print(f"Diagnostic single plot saved to {diag_path}\n")

if __name__ == '__main__':
    main()
