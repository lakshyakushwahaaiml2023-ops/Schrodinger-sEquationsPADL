"""
tdse_padl.core
==============
Core physics engine: Crank-Nicolson solver, wavepacket generation, and
potential definitions for the 1D Time-Dependent Schrödinger Equation.
"""

from .solver import CrankNicolsonSolver
from .wavepacket import gaussian_wavepacket
from .potential import (
    rectangular_barrier,
    rectangular_well,
    double_barrier,
    zero_potential,
)

__all__ = [
    "CrankNicolsonSolver",
    "gaussian_wavepacket",
    "rectangular_barrier",
    "rectangular_well",
    "double_barrier",
    "zero_potential",
]