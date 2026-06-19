"""
tdse_padl.data
==============
Dataset generation and loading utilities for TDSE trajectory data.
"""

from .generator import TrajectoryGenerator, QuantumDataset

__all__ = ["TrajectoryGenerator", "QuantumDataset"]