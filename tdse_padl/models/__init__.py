"""
tdse_padl.models
================
PADL (Physics-Accelerated Deep Learning) surrogate models for TDSE evolution.
"""

from .unet1d import UNet1D, PhysicsAwareLoss

__all__ = ["UNet1D", "PhysicsAwareLoss"]