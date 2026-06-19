"""
padl_solver.py
==============
Physics-Aware Deep Learning (PADL) hybrid solver for the 1D TDSE.

Hybrid stepping rhythm
----------------------
The model was trained with ``skip=5``, meaning each forward pass predicts
ψ(t + 5·dt) from ψ(t) — it acts as a surrogate for **5 consecutive CN steps**.

PADLSolver therefore operates in *blocks* of ``model_skip`` time-steps:

    block 0 :  model  → advances ψ by model_skip steps
    block 1 :  model  → …
    …
    block (physics_interval-1) : CN × model_skip  → exact anchor
    block physics_interval     : model  → …  (cycle repeats)

With physics_interval=5 and model_skip=5 the model handles 25 CN-equivalent
steps for every 5 exact CN steps, giving a theoretical 5× speedup on GPU
(less on CPU due to inference overhead).

After every model block the wavefunction is renormalised (‖ψ‖=1) to
correct the small norm drift introduced by the network.

Classes
-------
PADLSolver
    Hybrid block-stepper mixing model and CN.
Benchmarker
    Wall-time and accuracy comparison between pure CN and PADL hybrid.

Usage
-----
    See __main__ block at the bottom of this file.
"""

from __future__ import annotations

import time
from typing import List

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PADLSolver
# ---------------------------------------------------------------------------

class PADLSolver:
    """
    Hybrid PADL solver: periodic exact CN anchoring + UNet1D block predictions.

    The model replaces ``model_skip`` CN steps per call (matching the skip used
    at training time, default 5).  Every ``physics_interval`` model blocks an
    exact CN anchor block is inserted instead.

    Parameters
    ----------
    cn_solver : CrankNicolsonSolver
        A fully initialised Crank-Nicolson solver (N, L, dt, V already set).
    model : nn.Module
        Trained UNet1D in eval mode.  Accepts input of shape (1, 3, N) and
        returns (1, 2, N).
    device : str or torch.device
        Device for model inference (``'cpu'`` or ``'cuda'``).
    physics_interval : int, optional
        Insert a CN anchor block every ``physics_interval`` blocks (default 5).
        Blocks 0, physics_interval, 2*physics_interval, … use exact CN.
        All other blocks use the model.
    model_skip : int, optional
        Number of CN-equivalent steps each model call replaces (default 5).
        Must match the ``skip`` used when generating the training data.

    Attributes
    ----------
    cn_calls : int
        Cumulative number of CN *steps* taken (each anchor block = model_skip).
    model_calls : int
        Cumulative number of model inference calls made.
    """

    def __init__(
        self,
        cn_solver,
        model: nn.Module,
        device,
        physics_interval: int = 5,
        model_skip: int = 5,
    ) -> None:
        self.cn_solver        = cn_solver
        self.model            = model
        self.device           = torch.device(device)
        self.physics_interval = physics_interval
        self.model_skip       = model_skip

        # Move model to target device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Diagnostics (reset by Benchmarker between runs)
        self.cn_calls    = 0   # total CN *steps* taken
        self.model_calls = 0   # total model inference calls

        # Cache normalised potential as float32 for fast tensor assembly
        V_arr = np.asarray(cn_solver.V, dtype=np.float32)
        V_max = float(V_arr.max()) if V_arr.max() != 0.0 else 1.0
        self._V_norm = (V_arr / V_max).astype(np.float32)   # (N,)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _psi_to_tensor(self, psi: np.ndarray) -> torch.Tensor:
        """
        Convert complex128 numpy array (N,) to model input tensor (1, 3, N).

        Channels:
          0  Re(ψ)
          1  Im(ψ)
          2  V_norm   (pre-computed, broadcast from cached array)
        """
        psi32 = psi.astype(np.complex64)
        x = np.stack([
            psi32.real,      # (N,)
            psi32.imag,      # (N,)
            self._V_norm,    # (N,)
        ], axis=0)           # (3, N)
        return torch.from_numpy(x).unsqueeze(0).to(self.device)  # (1, 3, N)

    @staticmethod
    def _tensor_to_psi(t: torch.Tensor) -> np.ndarray:
        """
        Convert model output tensor (1, 2, N) to complex128 numpy array (N,).

        Channel 0 → Re(ψ),  Channel 1 → Im(ψ).
        """
        arr = t.squeeze(0).cpu().numpy()   # (2, N)
        return (arr[0] + 1j * arr[1]).astype(np.complex128)

    @staticmethod
    def _renormalise(psi: np.ndarray, dx: float) -> np.ndarray:
        """
        Renormalise ψ so that ∫|ψ|²dx = 1 (trapezoidal rule on uniform grid).
        """
        norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, dx=dx))
        if norm < 1e-30:
            return psi  # Avoid division by zero for fully absorbed packet
        return psi / norm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, psi: np.ndarray, V: np.ndarray, block_idx: int) -> np.ndarray:
        """
        Advance ψ by one *block* of ``model_skip`` time-steps.

        Decision rule
        -------------
        * ``block_idx % physics_interval == 0``
              → run ``model_skip`` exact CN steps (anchor block)
        * otherwise
              → single model forward pass (replaces ``model_skip`` CN steps)
                + renormalisation

        Parameters
        ----------
        psi : np.ndarray, shape (N,), dtype complex128
            Current wavefunction.
        V : np.ndarray, shape (N,)
            Potential array (kept for API symmetry; potential is cached at init).
        block_idx : int
            0-based block index.  Each block covers ``model_skip`` real time-steps.

        Returns
        -------
        psi_new : np.ndarray, shape (N,), dtype complex128
            Wavefunction after one hybrid block (= model_skip · dt of real time).
        """
        if block_idx % self.physics_interval == 0:
            # ---- Exact CN anchor block (model_skip individual CN steps) ----
            psi_new = self.cn_solver.step_n(psi, self.model_skip)
            self.cn_calls += self.model_skip
        else:
            # ---- Model surrogate block ----
            x_t = self._psi_to_tensor(psi)     # (1, 3, N)

            with torch.no_grad():
                y_t = self.model(x_t)          # (1, 2, N)

            psi_new = self._tensor_to_psi(y_t) # (N,) complex128

            # Physics correction: restore ‖ψ‖ = 1
            psi_new = self._renormalise(psi_new, self.cn_solver.dx)
            self.model_calls += 1

        return psi_new

    def step_n(
        self,
        psi: np.ndarray,
        V: np.ndarray,
        n_blocks: int,
        record_every: int = 10,
    ) -> List[np.ndarray]:
        """
        Run ``n_blocks`` hybrid blocks and collect snapshots.

        Each block advances real time by ``model_skip · dt``.

        Parameters
        ----------
        psi : np.ndarray, shape (N,), dtype complex128
            Initial wavefunction.
        V : np.ndarray, shape (N,)
            Potential array (passed through to ``step``).
        n_blocks : int
            Total number of hybrid blocks to run.
        record_every : int, optional
            Record a snapshot every ``record_every`` blocks (default 10).

        Returns
        -------
        snapshots : list of np.ndarray, each shape (N,), dtype complex128
            Wavefunction snapshots at blocks 0, record_every, 2·record_every, …
            The initial state is always included as the first snapshot.
        """
        psi = np.asarray(psi, dtype=np.complex128)
        snapshots: List[np.ndarray] = [psi.copy()]

        for i in range(n_blocks):
            psi = self.step(psi, V, block_idx=i)
            if (i + 1) % record_every == 0:
                snapshots.append(psi.copy())

        return snapshots


# ---------------------------------------------------------------------------
# Benchmarker
# ---------------------------------------------------------------------------

class Benchmarker:
    """
    Compare pure Crank-Nicolson against the PADL hybrid solver.

    Methods
    -------
    compare(psi0, V, n_steps=1000) -> dict
        Run both solvers for ``n_steps`` and return timing, norm, transmission,
        and mean-absolute-error statistics.
    """

    @staticmethod
    def _transmission(psi: np.ndarray, cn_solver, V: np.ndarray) -> float:
        """
        Estimate transmission as the fraction of |ψ|² to the right of the
        potential barrier's peak.

        The barrier peak is located at argmax(V); everything to the right of
        that index is counted as transmitted.
        """
        peak_idx = int(np.argmax(V))
        prob = np.abs(psi) ** 2
        total = np.trapezoid(prob, dx=cn_solver.dx)
        if total < 1e-30:
            return 0.0
        transmitted = np.trapezoid(prob[peak_idx:], dx=cn_solver.dx)
        return float(100.0 * transmitted / total)

    @staticmethod
    def _norm(psi: np.ndarray, dx: float) -> float:
        """Return ∫|ψ|²dx (should be ≈1 if normalised)."""
        return float(np.trapezoid(np.abs(psi) ** 2, dx=dx))

    def compare(
        self,
        psi0: np.ndarray,
        V: np.ndarray,
        cn_solver,
        padl_solver: PADLSolver,
        n_steps: int = 1000,
        record_every: int = 10,
    ) -> dict:
        """
        Run both solvers for ``n_steps`` CN-equivalent steps.

        The PADL solver runs ``n_steps // model_skip`` blocks, each covering
        the same real time as ``model_skip`` CN steps, so both solvers advance
        the wavefunction by exactly ``n_steps · dt`` of simulation time.

        Parameters
        ----------
        psi0 : np.ndarray, shape (N,), dtype complex128
            Initial wavefunction.
        V : np.ndarray, shape (N,)
            Potential array.
        cn_solver : CrankNicolsonSolver
            Exact reference solver.
        padl_solver : PADLSolver
            Hybrid PADL solver to benchmark.
        n_steps : int, optional
            Number of *CN-equivalent* steps (default 1000).  Must be divisible
            by ``padl_solver.model_skip``.
        record_every : int, optional
            Snapshot interval in CN steps for MAE computation (default 10).

        Returns
        -------
        results : dict with keys
            'cn_time'   : float  — wall time for pure CN (seconds)
            'padl_time' : float  — wall time for PADL hybrid (seconds)
            'speedup'   : float  — cn_time / padl_time  (>1 means PADL faster)
            'cn_norm'   : float  — ‖ψ_CN‖² at end
            'padl_norm' : float  — ‖ψ_PADL‖² at end
            'mae'       : float  — mean |ψ_PADL - ψ_CN| averaged over snapshots
            'cn_T'      : float  — CN transmission %
            'padl_T'    : float  — PADL transmission %
        """
        dx         = cn_solver.dx
        skip       = padl_solver.model_skip
        n_blocks   = n_steps // skip   # PADL block count (each = skip CN steps)
        rec_blocks = max(1, record_every // skip)  # snapshot interval in blocks
        psi0 = np.asarray(psi0, dtype=np.complex128)

        # ------------------------------------------------------------------
        # 1. Pure Crank-Nicolson  (n_steps individual steps)
        # ------------------------------------------------------------------
        print("  [Benchmarker] Running pure CN ...", flush=True)
        cn_snapshots: List[np.ndarray] = [psi0.copy()]
        psi_cn = psi0.copy()

        t0 = time.perf_counter()
        for i in range(n_steps):
            psi_cn = cn_solver.step(psi_cn)
            if (i + 1) % record_every == 0:
                cn_snapshots.append(psi_cn.copy())
        cn_time = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # 2. PADL hybrid  (n_blocks block steps, each = skip CN-equivalent)
        # ------------------------------------------------------------------
        print("  [Benchmarker] Running PADL hybrid ...", flush=True)
        padl_solver.cn_calls    = 0
        padl_solver.model_calls = 0

        t0 = time.perf_counter()
        padl_snapshots = padl_solver.step_n(
            psi0, V, n_blocks=n_blocks, record_every=rec_blocks
        )
        padl_time = time.perf_counter() - t0

        # ------------------------------------------------------------------
        # 3. Metrics
        # ------------------------------------------------------------------
        cn_norm   = self._norm(psi_cn,               dx)
        padl_norm = self._norm(padl_snapshots[-1],   dx)
        cn_T      = self._transmission(psi_cn,             cn_solver, V)
        padl_T    = self._transmission(padl_snapshots[-1], cn_solver, V)
        speedup   = cn_time / padl_time if padl_time > 0 else float('inf')

        # MAE averaged over all snapshot pairs (aligned by record_every)
        n_snaps = min(len(cn_snapshots), len(padl_snapshots))
        mae = float(np.mean([
            np.mean(np.abs(padl_snapshots[k] - cn_snapshots[k]))
            for k in range(n_snaps)
        ]))

        print(
            f"  [Benchmarker] CN calls: {padl_solver.cn_calls}, "
            f"model calls: {padl_solver.model_calls}",
            flush=True,
        )

        return {
            'cn_time':   cn_time,
            'padl_time': padl_time,
            'speedup':   speedup,
            'cn_norm':   cn_norm,
            'padl_norm': padl_norm,
            'mae':       mae,
            'cn_T':      cn_T,
            'padl_T':    padl_T,
        }


# ---------------------------------------------------------------------------
# Standalone demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from tdse_padl.core.solver import CrankNicolsonSolver
    from tdse_padl.core.wavepacket import gaussian_wavepacket
    from tdse_padl.core.potential import rectangular_barrier
    from tdse_padl.models.unet1d import UNet1D
    import torch

    N, L = 512, 1.0
    V = rectangular_barrier(N, L, center=0.6, width=0.05, height=200)
    psi0 = gaussian_wavepacket(N, L, x0=0.25, k0=50, sigma=0.05)

    # dt=2e-5 matches the new training data (skip=10, dt=2e-5 => effective dt_eff=2e-4)
    cn = CrankNicolsonSolver(N=N, L=L, dt=2e-5, V=V)

    # Load trained model
    ckpt = torch.load('checkpoints/best.pt', map_location='cpu', weights_only=True)
    model = UNet1D()
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # model_skip=10 matches the skip=10 used during training data generation
    padl = PADLSolver(cn, model, device='cpu', physics_interval=5, model_skip=10)
    bench = Benchmarker()
    # n_steps=1000 CN-equivalent steps; PADL runs 100 blocks of 10 steps each
    results = bench.compare(psi0, V, cn_solver=cn, padl_solver=padl, n_steps=1000)

    print(f"\n{'='*40}")
    print(f"Speedup:           {results['speedup']:.2f}x faster")
    print(f"Norm (CN):         {results['cn_norm']:.6f}")
    print(f"Norm (PADL):       {results['padl_norm']:.6f}")
    print(f"Transmission CN:   {results['cn_T']:.1f}%")
    print(f"Transmission PADL: {results['padl_T']:.1f}%")
    print(f"Mean error:        {results['mae']:.2e}")
    print(f"{'='*40}")
