"""
tdse_padl.core
==============
Core physics engine: Crank-Nicolson solver, wavepacket generation,
potential definitions, and the PADL hybrid solver.
"""

from .solver import CrankNicolsonSolver
from .wavepacket import gaussian_wavepacket
from .potential import (
    rectangular_barrier,
    rectangular_well,
    double_barrier,
    zero_potential,
)
from .padl_solver import PADLSolver, Benchmarker

__all__ = [
    "CrankNicolsonSolver",
    "gaussian_wavepacket",
    "rectangular_barrier",
    "rectangular_well",
    "double_barrier",
    "zero_potential",
    "PADLSolver",
    "Benchmarker",
]