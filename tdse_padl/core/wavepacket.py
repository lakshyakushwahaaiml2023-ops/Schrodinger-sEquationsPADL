"""
wavepacket.py
=============
Initial state generation for 1D TDSE simulations.

Provides Gaussian wavepacket initialisation on a uniform spatial grid.
All wavefunctions are returned as complex128 NumPy arrays normalised so that
the discrete approximation to ∫|ψ|² dx = 1.
"""

import numpy as np


def gaussian_wavepacket(
    N: int,
    L: float,
    x0: float = 0.25,
    k0: float = 50.0,
    sigma: float = 0.05,
) -> np.ndarray:
    """
    Construct a normalised Gaussian wavepacket on a uniform grid [0, L].

    The analytic form before normalisation is:

        ψ(x) = exp(-(x - x0)² / (2σ²)) · exp(i·k0·x)

    A real Gaussian envelope (width σ, centred at x0) modulates a plane
    wave of momentum k0.  After construction the discrete norm is set to 1
    via

        ψ ← ψ / sqrt(∫|ψ|² dx)  ≈  ψ / sqrt(Σ|ψ|²·dx)

    Parameters
    ----------
    N : int
        Number of spatial grid points.
    L : float
        Length of the spatial domain (grid spans [0, L]).
    x0 : float, optional
        Centre of the Gaussian envelope (default 0.25).
    k0 : float, optional
        Central wavenumber / momentum in natural units ℏ = m = 1
        (default 50.0).
    sigma : float, optional
        Standard deviation of the Gaussian envelope (default 0.05).

    Returns
    -------
    psi : np.ndarray, shape (N,), dtype complex128
        Normalised Gaussian wavepacket evaluated on the grid
        x = [dx, 2·dx, …, (N-1)·dx]  (interior points only; boundary
        values are kept zero to match Dirichlet conditions).
    """
    dx = L / N
    x = np.linspace(0.0, L, N, endpoint=False)  # x[0]=0, x[N-1]=L-dx

    # Build un-normalised wavepacket
    envelope = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))
    plane_wave = np.exp(1j * k0 * x)
    psi = (envelope * plane_wave).astype(np.complex128)

    # Enforce Dirichlet boundary conditions explicitly
    psi[0] = 0.0
    psi[-1] = 0.0

    # Normalise: ∫|ψ|² dx ≈ Σ|ψ[i]|² · dx = 1
    norm_sq = np.sum(np.abs(psi) ** 2) * dx
    psi /= np.sqrt(norm_sq)

    return psi