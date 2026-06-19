"""
potential.py
============
Potential energy functions for 1D TDSE simulations.

All functions operate on a uniform grid with N points spanning [0, L] and
return a real-valued NumPy array V of shape (N,).  V[i] is the potential at
grid point x_i = i · dx, where dx = L / N.

Available potentials
--------------------
* zero_potential       – V(x) = 0 everywhere (free particle)
* rectangular_barrier  – single rectangular potential barrier
* rectangular_well     – single rectangular potential well (V < 0)
* double_barrier       – two rectangular barriers (double-slit / resonance)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_grid(N: int, L: float) -> np.ndarray:
    """Return the N-point uniform spatial grid on [0, L)."""
    return np.linspace(0.0, L, N, endpoint=False)


# ---------------------------------------------------------------------------
# Potentials
# ---------------------------------------------------------------------------

def zero_potential(N: int, L: float) -> np.ndarray:
    """
    Free-particle potential: V(x) = 0 for all x.

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain length.

    Returns
    -------
    V : np.ndarray, shape (N,), dtype float64
        Zero potential array.
    """
    return np.zeros(N, dtype=np.float64)


def rectangular_barrier(
    N: int,
    L: float,
    center: float,
    width: float,
    height: float,
) -> np.ndarray:
    """
    Single rectangular (step) potential barrier.

    V(x) = height   if  |x - center| ≤ width / 2
    V(x) = 0        otherwise

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain length.
    center : float
        Position of the barrier centre within [0, L].
    width : float
        Full width of the barrier.
    height : float
        Peak value of the potential (> 0 for a barrier).

    Returns
    -------
    V : np.ndarray, shape (N,), dtype float64
    """
    x = _make_grid(N, L)
    V = np.zeros(N, dtype=np.float64)
    half = width / 2.0
    V[(x >= center - half) & (x <= center + half)] = height
    return V


def rectangular_well(
    N: int,
    L: float,
    center: float,
    width: float,
    depth: float,
) -> np.ndarray:
    """
    Single rectangular potential well (attractive region).

    V(x) = -|depth|  if  |x - center| ≤ width / 2
    V(x) = 0         otherwise

    The *depth* parameter is taken as a positive number representing how far
    below zero the well extends; the returned potential values are negative
    inside the well.

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain length.
    center : float
        Position of the well centre.
    width : float
        Full width of the well.
    depth : float
        Depth of the well (positive number; V = -depth inside).

    Returns
    -------
    V : np.ndarray, shape (N,), dtype float64
    """
    x = _make_grid(N, L)
    V = np.zeros(N, dtype=np.float64)
    half = width / 2.0
    V[(x >= center - half) & (x <= center + half)] = -abs(depth)
    return V


def double_barrier(
    N: int,
    L: float,
    c1: float,
    c2: float,
    width: float,
    height: float,
) -> np.ndarray:
    """
    Two identical rectangular barriers (double-barrier / resonant tunnelling).

    Each barrier has the same *width* and *height*; they are centred at
    positions *c1* and *c2* respectively.

        V(x) = height   if  |x - c1| ≤ width/2  OR  |x - c2| ≤ width/2
        V(x) = 0        otherwise

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain length.
    c1 : float
        Centre of the first barrier.
    c2 : float
        Centre of the second barrier.
    width : float
        Full width of each barrier.
    height : float
        Peak value of both barriers.

    Returns
    -------
    V : np.ndarray, shape (N,), dtype float64
    """
    V = rectangular_barrier(N, L, c1, width, height)
    V += rectangular_barrier(N, L, c2, width, height)
    return V