"""
metrics.py
==========
Wavefunction diagnostics for 1D TDSE simulations.

All functions accept the wavefunction ψ as a complex128 NumPy array of
shape (N,) and the spatial step dx = L / N.  They return Python floats
(or real-valued NumPy scalars).

Functions
---------
norm(psi, dx)
    Total probability: ∫|ψ(x)|² dx.

transmission(psi, dx, barrier_end_idx)
    Transmission coefficient T = ∫_{x > barrier} |ψ|² dx.

reflection(psi, dx, barrier_start_idx)
    Reflection coefficient R = ∫_{x < barrier} |ψ|² dx.

mean_position(psi, x_grid, dx)
    Expectation value ⟨x⟩ = ∫ x |ψ(x)|² dx.

mean_momentum(psi, dx)
    Expectation value ⟨p⟩ estimated via finite-difference phase gradient.
"""

import numpy as np


def norm(psi: np.ndarray, dx: float) -> float:
    """
    Compute the total probability norm of ψ.

    Approximates the continuous integral

        ‖ψ‖² = ∫₀ᴸ |ψ(x)|² dx

    using the rectangle rule:

        ‖ψ‖² ≈ Σᵢ |ψ[i]|² · dx

    Parameters
    ----------
    psi : np.ndarray, shape (N,), dtype complex128
        Wavefunction evaluated on the spatial grid.
    dx : float
        Spatial step size.

    Returns
    -------
    float
        Norm squared (total probability).  For a correctly initialised and
        evolved wavefunction this should remain close to 1.
    """
    return float(np.sum(np.abs(psi) ** 2) * dx)


def transmission(
    psi: np.ndarray,
    dx: float,
    barrier_end_idx: int,
) -> float:
    """
    Transmission coefficient: probability density to the *right* of the barrier.

    T = ∫_{x > x_barrier_end} |ψ(x)|² dx
      ≈ Σ_{i > barrier_end_idx} |ψ[i]|² · dx

    Parameters
    ----------
    psi : np.ndarray, shape (N,), dtype complex128
        Wavefunction on the spatial grid.
    dx : float
        Spatial step size.
    barrier_end_idx : int
        Grid index of the right edge of the barrier.  All grid points with
        index > barrier_end_idx are considered "transmitted".

    Returns
    -------
    float
        Transmission coefficient in [0, 1].
    """
    return float(np.sum(np.abs(psi[barrier_end_idx:]) ** 2) * dx)


def reflection(
    psi: np.ndarray,
    dx: float,
    barrier_start_idx: int,
) -> float:
    """
    Reflection coefficient: probability density to the *left* of the barrier.

    R = ∫_{x < x_barrier_start} |ψ(x)|² dx
      ≈ Σ_{i < barrier_start_idx} |ψ[i]|² · dx

    Parameters
    ----------
    psi : np.ndarray, shape (N,), dtype complex128
        Wavefunction on the spatial grid.
    dx : float
        Spatial step size.
    barrier_start_idx : int
        Grid index of the left edge of the barrier.  All grid points with
        index < barrier_start_idx are considered "reflected".

    Returns
    -------
    float
        Reflection coefficient in [0, 1].
    """
    return float(np.sum(np.abs(psi[:barrier_start_idx]) ** 2) * dx)


def mean_position(
    psi: np.ndarray,
    x_grid: np.ndarray,
    dx: float,
) -> float:
    """
    Compute the expectation value of position ⟨x⟩.

    ⟨x⟩ = ∫ x |ψ(x)|² dx / ∫ |ψ(x)|² dx
         ≈ (Σᵢ x[i] |ψ[i]|² · dx) / ‖ψ‖²

    The result is normalised by the current norm so that the answer remains
    meaningful even if ‖ψ‖ has drifted slightly from 1.

    Parameters
    ----------
    psi : np.ndarray, shape (N,), dtype complex128
        Wavefunction on the spatial grid.
    x_grid : np.ndarray, shape (N,), dtype float64
        Spatial grid array x[i] = i · dx.
    dx : float
        Spatial step size.

    Returns
    -------
    float
        ⟨x⟩ in the same units as x_grid.
    """
    prob_density = np.abs(psi) ** 2
    norm_sq = np.sum(prob_density) * dx
    if norm_sq == 0.0:
        return 0.0
    return float(np.sum(x_grid * prob_density) * dx / norm_sq)


def mean_momentum(psi: np.ndarray, dx: float) -> float:
    """
    Estimate the expectation value of momentum ⟨p⟩ via the phase gradient.

    In natural units (ℏ = 1) the momentum operator is  p̂ = -i ∂/∂x.
    The local momentum is approximated by the finite-difference phase gradient:

        p_local(x) ≈ Im[ ψ*(x) · (ψ(x+dx) - ψ(x-dx)) / (2·dx) ] / |ψ(x)|²

    Integrating over the grid gives:

        ⟨p⟩ = ∫ Im[ψ* · (-i ∂ψ/∂x)] dx
             ≈ -Σᵢ Im[ψ*[i] · (ψ[i+1]-ψ[i-1])/(2·dx)] · dx

    Boundary points (i=0, i=N-1) are excluded as their values are fixed at 0.

    Parameters
    ----------
    psi : np.ndarray, shape (N,), dtype complex128
        Wavefunction on the spatial grid.
    dx : float
        Spatial step size.

    Returns
    -------
    float
        ⟨p⟩ in natural units.  For a Gaussian wavepacket initialised with
        central wavenumber k0 this should be close to k0.
    """
    # Central finite difference: (ψ[i+1] - ψ[i-1]) / (2·dx)
    d_psi = (psi[2:] - psi[:-2]) / (2.0 * dx)   # shape (N-2,)
    psi_conj_mid = np.conj(psi[1:-1])             # shape (N-2,)

    # ⟨p⟩ = Σ Im[ψ*(x) · (-i ∂ψ/∂x)] · dx
    #      = -Σ Im[ψ*(x) · ∂ψ/∂x] · dx   (factor of -i·i = +1 in Im)
    integrand = np.imag(psi_conj_mid * d_psi)
    return float(-np.sum(integrand) * dx)