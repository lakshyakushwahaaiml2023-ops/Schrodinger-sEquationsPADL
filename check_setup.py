"""
check_setup.py
==============
Dependency verification and solver sanity checks for the TDSE-PADL simulator.
Useful for confirming system compatibility before launching a live demo.

Run:
  python check_setup.py
"""

import sys
import os
import importlib
import importlib.metadata
from packaging.version import Version

# Dependencies to check
DEPENDENCIES = {
    "numpy": "1.24",
    "scipy": "1.10",
    "torch": "2.0",
    "matplotlib": "3.7",
    "h5py": "3.8",
    "tqdm": "4.65",
}

def check_dependencies() -> bool:
    print("=" * 60)
    print("  TDSE-PADL  |  Dependency Verification")
    print("=" * 60)
    
    all_passed = True
    for pkg, req_ver in DEPENDENCIES.items():
        try:
            # Check import
            mod = importlib.import_module(pkg)
            # Fetch installed version
            try:
                inst_ver = importlib.metadata.version(pkg)
            except Exception:
                inst_ver = getattr(mod, "__version__", "unknown")
            
            # Verify version condition
            if inst_ver != "unknown" and Version(inst_ver) >= Version(req_ver):
                print(f"  [PASS] {pkg:<12} : installed {inst_ver:<8} (required >= {req_ver})")
            else:
                print(f"  [WARN] {pkg:<12} : installed {inst_ver:<8} (required >= {req_ver}) - Version mismatch")
                all_passed = False
        except ImportError:
            print(f"  [FAIL] {pkg:<12} : NOT INSTALLED (required >= {req_ver})")
            all_passed = False

    print("-" * 60)
    return all_passed

def run_solver_sanity_check() -> bool:
    print("\n" + "=" * 60)
    print("  TDSE-PADL  |  Crank-Nicolson Solver Sanity Check")
    print("=" * 60)
    
    try:
        import numpy as np
        from tdse_padl.core.solver import CrankNicolsonSolver
        from tdse_padl.core.wavepacket import gaussian_wavepacket
        from tdse_padl.core.potential import rectangular_barrier
        from tdse_padl.utils import norm
        
        N, L = 512, 1.0
        dt = 2e-5
        dx = L / N
        V = rectangular_barrier(N, L, center=0.6, width=0.05, height=200.0)
        psi0 = gaussian_wavepacket(N, L, x0=0.25, k0=50.0, sigma=0.05)
        
        solver = CrankNicolsonSolver(N=N, L=L, dt=dt, V=V)
        psi_next = solver.step_n(psi0, 10)
        
        # Verify shape
        assert psi_next.shape == (N,), f"Expected shape ({N},), got {psi_next.shape}"
        
        # Verify norm conservation
        initial_norm = norm(psi0, dx)
        final_norm = norm(psi_next, dx)
        norm_diff = abs(final_norm - initial_norm)
        
        assert norm_diff < 1e-12, f"Norm drift {norm_diff:.2e} exceeded tolerance 1e-12"
        
        print(f"  [PASS] Exact solver evolved 10 steps successfully.")
        print(f"         Final Norm: {final_norm:.10f} (Drift: {norm_diff:.2e})")
        print("-" * 60)
        return True
        
    except Exception as e:
        print(f"  [FAIL] Crank-Nicolson Solver check failed: {e}")
        print("-" * 60)
        return False

def run_padl_sanity_check() -> bool:
    print("\n" + "=" * 60)
    print("  TDSE-PADL  |  PADL Solver Sanity Check")
    print("=" * 60)
    
    ckpt_path = 'checkpoints/best.pt'
    if not os.path.exists(ckpt_path):
        print(f"  [WARN] Model checkpoint {ckpt_path} not found.")
        print("         Please train the model first to run PADL sanity check.")
        print("-" * 60)
        return False

    try:
        import numpy as np
        import torch
        from tdse_padl.core.solver import CrankNicolsonSolver
        from tdse_padl.core.wavepacket import gaussian_wavepacket
        from tdse_padl.core.potential import rectangular_barrier
        from tdse_padl.core.padl_solver import PADLSolver
        from tdse_padl.models.unet1d import UNet1D
        from tdse_padl.utils import norm

        N, L = 512, 1.0
        dt = 2e-5
        dx = L / N
        V = rectangular_barrier(N, L, center=0.6, width=0.05, height=200.0)
        psi0 = gaussian_wavepacket(N, L, x0=0.25, k0=50.0, sigma=0.05)

        # Load U-Net
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        model = UNet1D()
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        cn_solver = CrankNicolsonSolver(N=N, L=L, dt=dt, V=V)
        padl_solver = PADLSolver(cn_solver=cn_solver, model=model, device='cpu', physics_interval=5, model_skip=10)
        
        # Take 1 block step (replaces 5 CN steps)
        psi_next = padl_solver.step(psi0, V, block_idx=1) # block_idx 1 uses DL surrogate model
        
        # Verify shape
        assert psi_next.shape == (N,), f"Expected shape ({N},), got {psi_next.shape}"
        
        # Verify norm correction
        final_norm = norm(psi_next, dx)
        assert abs(final_norm - 1.0) < 1e-5, f"Norm {final_norm:.6f} was not corrected to 1.0"
        
        print(f"  [PASS] U-Net loaded and hybrid PADL solver evolved 1 block successfully.")
        print(f"         Surrogate Norm (corrected): {final_norm:.10f}")
        print("-" * 60)
        return True

    except Exception as e:
        print(f"  [FAIL] PADL Solver check failed: {e}")
        print("-" * 60)
        return False

def main():
    dep_ok = check_dependencies()
    cn_ok = run_solver_sanity_check()
    padl_ok = run_padl_sanity_check()
    
    print("\n" + "=" * 60)
    print("  OVERALL SYSTEM STATUS")
    print("=" * 60)
    if dep_ok and cn_ok:
        if padl_ok:
            print("  [SUCCESS] All dependencies, solvers, and checkpoints passed validation.")
            print("            System is ready for --demo launch!")
        else:
            print("  [PARTIAL] Core solver and dependencies OK, but PADL model check was skipped/failed.")
            print("            You can run CN simulations, but train the model to enable PADL.")
    else:
        print("  [ERROR] Critical dependency or solver check failed. Please resolve failures.")
    print("=" * 60)

if __name__ == "__main__":
    main()
