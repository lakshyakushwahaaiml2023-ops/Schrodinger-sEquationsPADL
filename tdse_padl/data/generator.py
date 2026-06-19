"""
generator.py
============
TDSE trajectory dataset generator for PADL model training.

Overview
--------
The module defines two public classes:

TrajectoryGenerator
    Runs the exact Crank-Nicolson solver to produce (input, target) pairs
    for supervised surrogate-model training.

    Input  – shape (3, N) float32:
        ch 0 : Re(psi(x, t))
        ch 1 : Im(psi(x, t))
        ch 2 : V(x) / V_max    (normalised static potential context)

    Target – shape (2, N) float32:
        ch 0 : Re(psi(x, t + skip*dt))
        ch 1 : Im(psi(x, t + skip*dt))

    Potentials are drawn from three regimes each trajectory:
        80 % – rectangular barrier  (tunnelling + classical over-barrier)
        10 % – double barrier        (resonant tunnelling)
        10 % – zero potential        (free-particle reference)

QuantumDataset (torch.utils.data.Dataset)
    Lazy HDF5 reader that wraps a file produced by
    TrajectoryGenerator.generate_dataset().  Supports spatial-flip
    augmentation (x -> L - x), which is a valid symmetry of the TDSE
    when the potential is simultaneously flipped.

Usage
-----
    gen = TrajectoryGenerator()
    gen.generate_dataset(n_trajectories=500, save_path='data/train.h5')

    ds = QuantumDataset('data/train.h5', augment=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)
"""

from __future__ import annotations

import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from ..core.solver import CrankNicolsonSolver
from ..core.wavepacket import gaussian_wavepacket
from ..core.potential import (
    rectangular_barrier,
    double_barrier,
    zero_potential,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_potential(V: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Normalise a potential array to [0, 1] by its maximum absolute value.

    Returns the normalised array and the scale factor V_max.
    If V is identically zero (free particle), V_max is set to 1.0 to avoid
    division-by-zero; the returned array is all zeros.

    Parameters
    ----------
    V : np.ndarray, shape (N,)
        Raw potential in natural units.

    Returns
    -------
    V_norm : np.ndarray, shape (N,), float32
        V / V_max, values in [0, 1].
    V_max : float
        Maximum absolute value of V (used to reconstruct V = V_norm * V_max).
    """
    V_max = float(np.max(np.abs(V)))
    if V_max == 0.0:
        V_max = 1.0
    return (V / V_max).astype(np.float32), V_max


# ---------------------------------------------------------------------------
# TrajectoryGenerator
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """
    Generate (input, target) TDSE trajectory datasets using the CN solver.

    Each call to :meth:`generate_trajectory` runs the CN solver for a single
    wavepacket / potential configuration and extracts overlapping
    (psi_t, psi_{t+skip}) pairs at every `skip` steps.

    :meth:`generate_dataset` orchestrates many trajectories with randomised
    physical parameters and writes the concatenated dataset to HDF5.

    Parameters
    ----------
    N : int, optional
        Number of spatial grid points (default 512).
    L : float, optional
        Spatial domain length in natural units (default 1.0).
    dt : float, optional
        CN timestep (default 1e-5).
    skip : int, optional
        Number of CN steps between saved (input, target) pairs (default 5).
        The model learns to predict `skip` steps ahead in one network pass.
    device : str, optional
        Torch device hint stored for future GPU-based extensions (default 'cpu').
        Currently the solver runs on CPU regardless of this setting.
    seed : int or None, optional
        NumPy random seed for reproducibility (default None = non-deterministic).
    """

    # Wavepacket parameter ranges  (sampled uniformly each trajectory)
    _K0_RANGE      = (30.0, 80.0)
    _SIGMA_RANGE   = (0.03, 0.08)
    _X0_RANGE      = (0.15, 0.35)

    # Barrier parameter ranges
    _BC_RANGE      = (0.55, 0.75)   # barrier centre
    _BW_RANGE      = (0.03, 0.10)   # barrier width
    _BH_RANGE      = (50.0, 500.0)  # barrier height

    # Potential type probabilities: [rectangular, double, zero]
    _POT_PROBS = [0.80, 0.10, 0.10]

    def __init__(
        self,
        N: int   = 512,
        L: float = 1.0,
        dt: float = 2e-5,
        skip: int = 10,
        device: str = 'cpu',
        seed: int | None = None,
    ) -> None:
        self.N      = N
        self.L      = L
        self.dt     = dt
        self.dx     = L / N
        self.skip   = skip
        self.device = device
        self.rng    = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core trajectory generation
    # ------------------------------------------------------------------

    def generate_trajectory(
        self,
        psi0: np.ndarray,
        V: np.ndarray,
        n_steps: int = 2000,
    ) -> dict:
        """
        Evolve `psi0` under potential `V` for `n_steps` CN steps and extract
        (input, target) pairs every `skip` steps.

        Pair extraction
        ---------------
        At step indices  t = 0, skip, 2*skip, …, n_steps - skip:
            input  = [Re(psi_t), Im(psi_t), V_norm]  shape (3, N)
            target = [Re(psi_{t+skip}), Im(psi_{t+skip})]  shape (2, N)

        This yields  n_steps // skip  pairs per trajectory.

        Parameters
        ----------
        psi0 : np.ndarray, shape (N,), complex128
            Normalised initial wavefunction.
        V : np.ndarray, shape (N,), float64
            Potential energy profile (physical units, not normalised).
        n_steps : int, optional
            Total number of CN steps to run (default 2000).

        Returns
        -------
        dict with keys:
            ``'inputs'``  – float32 array of shape (n_steps//skip, 3, N)
            ``'targets'`` – float32 array of shape (n_steps//skip, 2, N)
            ``'V'``       – float32 array of shape (N,), the raw potential
        """
        n_pairs  = n_steps // self.skip
        inputs   = np.empty((n_pairs, 3, self.N), dtype=np.float32)
        targets  = np.empty((n_pairs, 2, self.N), dtype=np.float32)

        V_norm, _ = _normalise_potential(V)

        # Build solver for this potential
        solver = CrankNicolsonSolver(N=self.N, L=self.L, dt=self.dt, V=V)

        psi = psi0.astype(np.complex128)

        for pair_idx in range(n_pairs):
            # --- record input at current time ---
            inputs[pair_idx, 0] = psi.real.astype(np.float32)
            inputs[pair_idx, 1] = psi.imag.astype(np.float32)
            inputs[pair_idx, 2] = V_norm                          # static context

            # --- advance `skip` CN steps ---
            psi = solver.step_n(psi, self.skip)

            # --- record target at t + skip*dt ---
            targets[pair_idx, 0] = psi.real.astype(np.float32)
            targets[pair_idx, 1] = psi.imag.astype(np.float32)

        return {
            'inputs':  inputs,
            'targets': targets,
            'V':       V.astype(np.float32),
        }

    # ------------------------------------------------------------------
    # Dataset generation
    # ------------------------------------------------------------------

    def generate_dataset(
        self,
        n_trajectories: int = 500,
        save_path: str = 'data/train.h5',
        n_steps: int = 2000,
    ) -> None:
        """
        Generate `n_trajectories` randomised trajectories and save to HDF5.

        Randomisation per trajectory
        ----------------------------
        Wavepacket:
            k0     ~ Uniform(30, 80)
            sigma  ~ Uniform(0.03, 0.08)
            x0     ~ Uniform(0.15, 0.35)

        Potential (drawn with probabilities 0.80 / 0.10 / 0.10):
            Rectangular barrier:
                centre ~ Uniform(0.55, 0.75)
                width  ~ Uniform(0.03, 0.10)
                height ~ Uniform(50, 500)
            Double barrier:
                c1     ~ Uniform(0.50, 0.60)
                c2     ~ Uniform(0.65, 0.75)
                width  ~ Uniform(0.02, 0.05)
                height ~ Uniform(50, 300)
            Zero potential: V = 0 everywhere

        HDF5 layout
        -----------
        /inputs    (total_samples, 3, N)  float32
        /targets   (total_samples, 2, N)  float32
        Attributes: N, L, dt, skip, n_steps, n_trajectories

        Parameters
        ----------
        n_trajectories : int, optional
            Number of independent wavepacket trajectories (default 500).
        save_path : str, optional
            Output HDF5 file path (default 'data/train.h5').  Parent
            directories are created automatically.
        n_steps : int, optional
            CN steps per trajectory (default 2000).
        """
        n_pairs_per_traj = n_steps // self.skip
        total_samples    = n_trajectories * n_pairs_per_traj

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        # Pre-allocate HDF5 datasets so we can write trajectory by trajectory
        with h5py.File(save_path, 'w') as hf:
            ds_inputs  = hf.create_dataset(
                'inputs',
                shape=(total_samples, 3, self.N),
                dtype=np.float32,
                chunks=(min(n_pairs_per_traj, total_samples), 3, self.N),
                compression='gzip',
                compression_opts=4,
            )
            ds_targets = hf.create_dataset(
                'targets',
                shape=(total_samples, 2, self.N),
                dtype=np.float32,
                chunks=(min(n_pairs_per_traj, total_samples), 2, self.N),
                compression='gzip',
                compression_opts=4,
            )

            # Metadata attributes
            hf.attrs['N']               = self.N
            hf.attrs['L']               = self.L
            hf.attrs['dt']              = self.dt
            hf.attrs['skip']            = self.skip
            hf.attrs['n_steps']         = n_steps
            hf.attrs['n_trajectories']  = n_trajectories
            hf.attrs['n_pairs_per_traj']= n_pairs_per_traj

            write_ptr = 0  # running HDF5 row index

            desc = os.path.basename(save_path)
            for traj_idx in tqdm(
                range(n_trajectories),
                desc=f'Generating {desc}',
                unit='traj',
                dynamic_ncols=True,
            ):
                # ---- sample wavepacket parameters ----
                k0    = float(self.rng.uniform(*self._K0_RANGE))
                sigma = float(self.rng.uniform(*self._SIGMA_RANGE))
                x0    = float(self.rng.uniform(*self._X0_RANGE))

                # ---- sample potential type ----
                pot_type = self.rng.choice(
                    ['rect', 'double', 'zero'],
                    p=self._POT_PROBS,
                )
                V = self._sample_potential(pot_type)

                # ---- initial wavepacket ----
                psi0 = gaussian_wavepacket(
                    self.N, self.L, x0=x0, k0=k0, sigma=sigma
                )

                # ---- run trajectory ----
                traj = self.generate_trajectory(psi0, V, n_steps=n_steps)

                # ---- write to HDF5 ----
                end_ptr = write_ptr + n_pairs_per_traj
                ds_inputs [write_ptr:end_ptr] = traj['inputs']
                ds_targets[write_ptr:end_ptr] = traj['targets']
                write_ptr = end_ptr

        print(
            f"[generator] Saved {total_samples:,} samples "
            f"({n_trajectories} trajectories x {n_pairs_per_traj} pairs) "
            f"-> {save_path}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sample_potential(self, pot_type: str) -> np.ndarray:
        """
        Draw a random potential array for the given potential type.

        Parameters
        ----------
        pot_type : {'rect', 'double', 'zero'}

        Returns
        -------
        V : np.ndarray, shape (N,), float64
        """
        if pot_type == 'rect':
            centre = float(self.rng.uniform(*self._BC_RANGE))
            width  = float(self.rng.uniform(*self._BW_RANGE))
            height = float(self.rng.uniform(*self._BH_RANGE))
            return rectangular_barrier(self.N, self.L, centre, width, height)

        elif pot_type == 'double':
            # Two barriers on either side of centre region
            c1     = float(self.rng.uniform(0.50, 0.60))
            c2     = float(self.rng.uniform(0.65, 0.75))
            width  = float(self.rng.uniform(0.02, 0.05))
            height = float(self.rng.uniform(50.0, 300.0))
            return double_barrier(self.N, self.L, c1, c2, width, height)

        else:  # 'zero'
            return zero_potential(self.N, self.L)


# ---------------------------------------------------------------------------
# QuantumDataset
# ---------------------------------------------------------------------------

class QuantumDataset(Dataset):
    """
    PyTorch Dataset for lazy-loading TDSE (input, target) pairs from HDF5.

    The HDF5 file is kept open for the lifetime of the dataset object so that
    individual samples are loaded on demand without reading the whole file.
    This is memory-efficient for large datasets that do not fit in RAM.

    Augmentation
    ------------
    When `augment=True`, each sample is spatially flipped (x -> L - x) with
    probability 0.5.  Because the 1D TDSE is symmetric under x -> L - x when
    both psi and V are flipped simultaneously, the flipped pair is an equally
    valid (input, target) pair.

    Concretely, flipping reverses the spatial axis of all channels:
        input  flipped: [Re(psi)[::−1], Im(psi)[::−1], V_norm[::−1]]
        target flipped: [Re(psi')[::−1], Im(psi')[::−1]]

    Parameters
    ----------
    h5_path : str
        Path to an HDF5 file produced by TrajectoryGenerator.generate_dataset().
    augment : bool, optional
        Enable random spatial-flip augmentation (default True).
    preload : bool, optional
        If True (default), load the entire dataset into RAM as contiguous
        numpy arrays at construction time.  This eliminates per-sample HDF5
        disk seeks and reduces epoch time from minutes to seconds.  Set to
        False only when RAM is insufficient (>1 GB needed for 80 k samples).

    Attributes
    ----------
    n_samples : int
        Total number of (input, target) pairs in the dataset.
    N : int
        Number of spatial grid points (read from HDF5 attributes).
    """

    def __init__(self, h5_path: str, augment: bool = True, preload: bool = True) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.augment = augment
        self._preloaded = preload

        # Open file to read metadata (and optionally bulk-load data)
        self._file   = h5py.File(h5_path, 'r')
        self._inputs_ds  = self._file['inputs']   # HDF5 dataset handle
        self._targets_ds = self._file['targets']  # HDF5 dataset handle

        self.n_samples = self._inputs_ds.shape[0]
        self.N         = int(self._file.attrs.get('N', self._inputs_ds.shape[2]))

        if preload:
            # Bulk read: single contiguous HDF5 read is ~100x faster than
            # n_samples individual reads.  80k × 5 × 512 × float32 ≈ 820 MB.
            print(f"[QuantumDataset] Preloading {self.n_samples:,} samples into RAM "
                  f"from {h5_path} ...", flush=True)
            self._inputs_mem  = self._inputs_ds[:]   .astype(np.float32)  # (N,3,512)
            self._targets_mem = self._targets_ds[:].astype(np.float32)  # (N,2,512)
            mb = (self._inputs_mem.nbytes + self._targets_mem.nbytes) / 1024**2
            print(f"[QuantumDataset] Preloaded {mb:.0f} MB into RAM.", flush=True)
            # Close HDF5 — no longer needed
            self._file.close()
            self._file = None
        else:
            # Keep HDF5 open for lazy per-sample reads
            self._inputs  = self._inputs_ds
            self._targets = self._targets_ds

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch the idx-th (input, target) pair as torch FloatTensors.

        Parameters
        ----------
        idx : int
            Sample index in [0, len(self)).

        Returns
        -------
        x : torch.FloatTensor, shape (3, N)
            [Re(psi_t), Im(psi_t), V_norm]
        y : torch.FloatTensor, shape (2, N)
            [Re(psi_{t+skip}), Im(psi_{t+skip})]
        """
        # Fetch from RAM cache or HDF5
        if self._preloaded:
            x = self._inputs_mem [idx]   # already float32, shape (3, N)
            y = self._targets_mem[idx]   # already float32, shape (2, N)
        else:
            x = np.array(self._inputs [idx], dtype=np.float32)  # (3, N)
            y = np.array(self._targets[idx], dtype=np.float32)  # (2, N)

        # Warn on near-zero delta (don't hard-assert to avoid DataLoader crashes)
        delta = float(np.mean(np.abs(y - x[:2])))
        if delta < 1e-6:
            import warnings
            warnings.warn(f"Sample {idx} has near-zero delta: {delta:.2e} — may be lazy.")

        # Spatial-flip augmentation with p=0.5
        if self.augment and np.random.random() < 0.5:
            x = x[:, ::-1].copy()   # flip all 3 channels along spatial axis
            y = y[:, ::-1].copy()   # flip both target channels
        else:
            # Ensure we return writable copies (not HDF5 views)
            x = x.copy()
            y = y.copy()

        return torch.from_numpy(x), torch.from_numpy(y)

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Explicitly close the underlying HDF5 file handle (no-op if preloaded)."""
        try:
            if self._file is not None and self._file.id.valid:
                self._file.close()
        except Exception:
            pass  # Ignore errors during interpreter shutdown

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            f"QuantumDataset("
            f"n_samples={self.n_samples:,}, "
            f"N={self.N}, "
            f"augment={self.augment}, "
            f"path='{self.h5_path}')"
        )


# ---------------------------------------------------------------------------
# Standalone dataset generation script
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    print("=" * 60)
    print("  TDSE-PADL  |  Dataset Generator")
    print("=" * 60)

    gen = TrajectoryGenerator(
        N=512,
        L=1.0,
        dt=2e-5,
        skip=10,
        seed=42,        # reproducible run when called directly
    )

    # Generate training set (200 trajectories * 400 pairs = 80,000 samples)
    gen.generate_dataset(
        n_trajectories=200,
        save_path='data/train.h5',
        n_steps=2000,
    )

    # Generate validation set (40 trajectories * 400 pairs = 16,000 samples)
    gen.generate_dataset(
        n_trajectories=40,
        save_path='data/val.h5',
        n_steps=2000,
    )

    print()
    print("Dataset sizes:")
    ds_train = QuantumDataset('data/train.h5', augment=False)
    print(f"  Train: {len(ds_train):,} samples")
    ds_val   = QuantumDataset('data/val.h5',   augment=False)
    print(f"  Val:   {len(ds_val):,} samples")

    # Quick shape sanity check
    x0, y0 = ds_train[0]
    print()
    print(f"  Sample input  shape : {tuple(x0.shape)}  dtype={x0.dtype}")
    print(f"  Sample target shape : {tuple(y0.shape)}  dtype={y0.dtype}")

    # Confirm input channels look reasonable
    assert x0.shape == (3, 512), f"Unexpected input shape: {x0.shape}"
    assert y0.shape == (2, 512), f"Unexpected target shape: {y0.shape}"

    # V channel (ch 2) should be in [0, 1] after normalisation
    v_ch = x0[2]
    assert float(v_ch.min()) >= -1e-6, "V channel below 0"
    assert float(v_ch.max()) <= 1.0 + 1e-6, "V channel above 1"

    print()
    print("  Shape assertions passed.")
    print("=" * 60)

    ds_train.close()
    ds_val.close()
    sys.exit(0)