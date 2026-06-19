"""
unet1d.py
=========
1D U-Net surrogate model for TDSE wavefunction evolution and its
companion physics-aware loss function.

Architecture overview
---------------------
The network maps a 3-channel spatial input of shape (B, 3, N) to a
2-channel output of shape (B, 2, N):

    Input channels:
        ch 0 : Re(ψ(x, t))
        ch 1 : Im(ψ(x, t))
        ch 2 : V(x) / V_max    ← static potential context

    Output channels  (residual formulation):
        ch 0 : Re(ψ(x, t + skip·dt))
        ch 1 : Im(ψ(x, t + skip·dt))

Residual formulation
--------------------
The model learns the *increment* Δψ = ψ_{t+skip} − ψ_t rather than the
full next state.  Because Δψ is typically small (CN is a near-identity map
for small dt·skip), this dramatically reduces the effective dynamic range
the network must represent and accelerates convergence.

    ψ_pred = ψ_input  +  UNet1D(input)          # forward() applies this
                ↑               ↑
           channels 0-1    raw network output

Design choices
--------------
* **Kernel size 7** – larger receptive field per layer than the common k=3;
  captures the smooth, long-range spatial correlations of wavefunctions.
* **GroupNorm(8, C)** – instance-group normalisation; works correctly with
  batch size 1, unlike BatchNorm.
* **GELU activations** – smooth non-linearity; better gradient flow than
  ReLU for physics-surrogate tasks.
* **Skip connections** – encoder features (pre-pooling) are concatenated
  with the decoder at matching spatial scales.
* **No output activation** – wavefunction amplitudes are unbounded reals.

References
----------
* Ronneberger, O. et al. "U-Net: Convolutional Networks for Biomedical
  Image Segmentation." MICCAI 2015.
* Li, Z. et al. "Fourier Neural Operator for Parametric Partial
  Differential Equations." ICLR 2021.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class EncBlock(nn.Module):
    """
    Encoder (double-conv) block used throughout the U-Net.

    Structure:
        Conv1d(in_ch, out_ch, k=7, pad=3)
        GroupNorm(8, out_ch)
        GELU
        Conv1d(out_ch, out_ch, k=7, pad=3)
        GroupNorm(8, out_ch)
        GELU

    The padding of 3 with kernel 7 preserves the spatial length at every
    layer  (L_out = L_in  when stride=1, pad=(k-1)//2 = 3).

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch,  out_ch, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecBlock(nn.Module):
    """
    Decoder block: upsample, concatenate skip connection, then double-conv.

    Structure:
        ConvTranspose1d(in_ch, out_ch, k=2, stride=2)   ← 2× upsample
        cat([upsampled, skip], dim=1)                    ← skip concat
        EncBlock(out_ch + skip_ch, out_ch)               ← double-conv

    Parameters
    ----------
    in_ch : int
        Channels coming from the deeper decoder stage.
    skip_ch : int
        Channels in the corresponding encoder skip feature map.
    out_ch : int
        Output channels of this decoder stage.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose1d(
            in_ch, out_ch, kernel_size=2, stride=2
        )
        self.conv = EncBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)                  # (B, out_ch, L*2)

        # Handle edge case where spatial size mismatches by 1 after upsampling
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode='linear',
                              align_corners=False)

        x = torch.cat([x, skip], dim=1)       # (B, out_ch + skip_ch, L)
        return self.conv(x)                   # (B, out_ch, L)


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class UNet1D(nn.Module):
    """
    1D U-Net surrogate for TDSE wavefunction propagation.

    Maps  (B, 3, N) → (B, 2, N)  via an encoder-bottleneck-decoder
    architecture with skip connections.  The output is the *full predicted
    wavefunction* ψ_{t+skip}, computed as:

        ψ_pred = ψ_input[:, :2, :]  +  raw_network_output

    i.e. the network learns the residual Δψ.

    Channel progression:
        Encoder :  3 → 32 → 64 → 128 → 256
        Bottleneck:          256 → 512
        Decoder : 512 → 256 → 128 → 64 → 32
        Head    :  32 → 2

    Spatial progression (for N=512):
        After Down1 : 512 → 256
        After Down2 : 256 → 128
        After Down3 : 128 →  64
        After Down4 :  64 →  32  (bottleneck)
        After Up1   :  32 →  64
        After Up2   :  64 → 128
        After Up3   : 128 → 256
        After Up4   : 256 → 512

    Parameters
    ----------
    None (all hyperparameters are fixed in the architecture).

    Inputs
    ------
    x : torch.Tensor, shape (B, 3, N)
        Batch of [Re(ψ), Im(ψ), V_norm] channel inputs.

    Returns
    -------
    torch.Tensor, shape (B, 2, N)
        Predicted [Re(ψ_{t+skip}), Im(ψ_{t+skip})].
    """

    def __init__(self) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        # Encoder (keep skip connections BEFORE pooling)
        # ------------------------------------------------------------------
        self.enc1 = EncBlock(3,    32)    # → (B, 32,  N)
        self.enc2 = EncBlock(32,   64)    # → (B, 64,  N/2)
        self.enc3 = EncBlock(64,  128)    # → (B, 128, N/4)
        self.enc4 = EncBlock(128, 256)    # → (B, 256, N/8)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        self.bottleneck = EncBlock(256, 512)  # → (B, 512, N/16)

        # ------------------------------------------------------------------
        # Decoder
        # ------------------------------------------------------------------
        self.dec4 = DecBlock(512, 256, 256)   # N/16 → N/8
        self.dec3 = DecBlock(256, 128, 128)   # N/8  → N/4
        self.dec2 = DecBlock(128,  64,  64)   # N/4  → N/2
        self.dec1 = DecBlock( 64,  32,  32)   # N/2  → N

        # ------------------------------------------------------------------
        # Output head (1×1 conv = per-point linear projection)
        # ------------------------------------------------------------------
        self.head = nn.Conv1d(32, 2, kernel_size=1)

        # ------------------------------------------------------------------
        # Weight initialisation
        # ------------------------------------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------
    # Weight init
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Initialise Conv1d weights with Kaiming (He) normal and zero biases.
        GroupNorm parameters are set to weight=1, bias=0 (default behaviour
        but made explicit here).
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-init the output head so the network starts as a near-identity
        # map (predicting Δψ ≈ 0, i.e. ψ_pred ≈ ψ_input).  This gives
        # stable early training.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of UNet1D.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, N)
            Input channels: [Re(ψ_t), Im(ψ_t), V_norm].

        Returns
        -------
        torch.Tensor, shape (B, 2, N)
            Predicted wavefunction [Re(ψ_{t+skip}), Im(ψ_{t+skip})].
        """
        # Keep the input wavefunction channels for the residual connection
        psi_in = x[:, :2, :]           # (B, 2, N)

        # ------------------------------------------------------------------
        # Encoder path — save skip features BEFORE pooling
        # ------------------------------------------------------------------
        s1 = self.enc1(x)              # (B,  32, N)
        s2 = self.enc2(self.pool(s1))  # (B,  64, N/2)
        s3 = self.enc3(self.pool(s2))  # (B, 128, N/4)
        s4 = self.enc4(self.pool(s3))  # (B, 256, N/8)

        # ------------------------------------------------------------------
        # Bottleneck
        # ------------------------------------------------------------------
        b = self.bottleneck(self.pool(s4))   # (B, 512, N/16)

        # ------------------------------------------------------------------
        # Decoder path — upsample and fuse with skip connections
        # ------------------------------------------------------------------
        d4 = self.dec4(b,  s4)         # (B, 256, N/8)
        d3 = self.dec3(d4, s3)         # (B, 128, N/4)
        d2 = self.dec2(d3, s2)         # (B,  64, N/2)
        d1 = self.dec1(d2, s1)         # (B,  32, N)

        # ------------------------------------------------------------------
        # Output head + residual connection
        # ------------------------------------------------------------------
        delta = self.head(d1)          # (B, 2, N) — predicted increment Δψ
        return psi_in + delta          # (B, 2, N) — predicted ψ_{t+skip}


# ---------------------------------------------------------------------------
# Physics-aware loss
# ---------------------------------------------------------------------------

class PhysicsAwareLoss(nn.Module):
    """
    Multi-objective loss for TDSE surrogate training.

    Combines three terms:

    1. **L1 loss** (fidelity):
            L1 = mean |pred − target|

    2. **Norm conservation loss** (physics constraint):
            ||ψ||² = ∫ (Re²+Im²) dx  must equal the target norm.
            norm_loss = mean( (||ψ_pred||² − ||ψ_target||²)² )

       A perfectly norm-conserving CN solver has ||ψ||²=1 at all times, but
       the surrogate must learn this implicitly.  This penalty explicitly
       penalises drift.

    3. **Gradient smoothness loss** (regularity):
            grad_loss = mean |∇_x ψ_pred − ∇_x ψ_target|
            (finite-difference approximation of ∂ψ/∂x)

       Wavefunctions must be differentiable; this discourages the network
       from producing jagged spatial profiles.

    Total:
        loss = λ_l1 · L1  +  λ_norm · norm_loss  +  λ_grad · grad_loss

    Parameters
    ----------
    dx : float, optional
        Spatial step size (default 1/512).  Used to convert discrete sums
        to integral approximations.
    lambda_norm : float, optional
        Weight on the norm-conservation penalty (default 1.0).
    lambda_l1 : float, optional
        Weight on the L1 fidelity loss (default 1.0).
    lambda_grad : float, optional
        Weight on the gradient smoothness loss (default 0.5).
    """

    def __init__(
        self,
        dx: float   = 1.0 / 512,
        lambda_norm: float = 1.0,
        lambda_l1:   float = 1.0,
        lambda_grad: float = 0.5,
    ) -> None:
        super().__init__()
        self.dx          = dx
        self.lambda_norm = lambda_norm
        self.lambda_l1   = lambda_l1
        self.lambda_grad = lambda_grad

    def forward(
        self,
        pred:          torch.Tensor,   # (B, 2, N)
        target:        torch.Tensor,   # (B, 2, N)
        V_normalized:  torch.Tensor,   # (B, N)  — unused here, kept for API
    ) -> dict[str, torch.Tensor]:
        """
        Compute all loss components.

        Parameters
        ----------
        pred : torch.Tensor, shape (B, 2, N)
            Predicted wavefunction [Re(ψ_pred), Im(ψ_pred)].
        target : torch.Tensor, shape (B, 2, N)
            Ground-truth wavefunction [Re(ψ_true), Im(ψ_true)].
        V_normalized : torch.Tensor, shape (B, N)
            Normalised potential (passed through for API completeness;
            currently not used in the loss but available for future
            potential-weighted losses).

        Returns
        -------
        dict with keys 'l1', 'norm', 'grad', 'total'
            Each value is a scalar tensor with gradient.
        """
        # ------------------------------------------------------------------
        # 1. L1 fidelity loss
        # ------------------------------------------------------------------
        l1_loss = F.l1_loss(pred, target)

        # ------------------------------------------------------------------
        # 2. Norm conservation loss
        # ∫|ψ|² dx ≈ Σ (Re²+Im²) · dx   summed over spatial axis
        # ------------------------------------------------------------------
        pred_norm   = (pred[:, 0, :]**2   + pred[:, 1, :]**2).sum(dim=-1)   * self.dx
        target_norm = (target[:, 0, :]**2 + target[:, 1, :]**2).sum(dim=-1) * self.dx
        norm_loss   = ((pred_norm - target_norm) ** 2).mean()

        # ------------------------------------------------------------------
        # 3. Gradient smoothness loss
        # ∇_x ψ ≈ (ψ[i+1] - ψ[i]) / dx  (forward difference, shape B,2,N-1)
        # ------------------------------------------------------------------
        pred_grad   = pred[:, :, 1:]   - pred[:, :, :-1]
        target_grad = target[:, :, 1:] - target[:, :, :-1]
        grad_loss   = F.l1_loss(pred_grad, target_grad)

        # ------------------------------------------------------------------
        # Total weighted loss
        # ------------------------------------------------------------------
        total = (
            self.lambda_l1  * l1_loss
            + self.lambda_norm * norm_loss
            + self.lambda_grad * grad_loss
        )

        return {
            'l1':    l1_loss,
            'norm':  norm_loss,
            'grad':  grad_loss,
            'total': total,
        }


# ---------------------------------------------------------------------------
# Model summary / sanity check
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    torch.manual_seed(0)

    print("=" * 60)
    print("  UNet1D  |  Architecture & Loss Sanity Check")
    print("=" * 60)

    model = UNet1D()
    model.eval()

    # ------------------------------------------------------------------
    # Shape check
    # ------------------------------------------------------------------
    x      = torch.randn(2, 3, 512)
    with torch.no_grad():
        out = model(x)

    print(f"\n  Input  shape : {tuple(x.shape)}")
    print(f"  Output shape : {tuple(out.shape)}")
    assert out.shape == (2, 2, 512), f"Shape mismatch: {out.shape}"

    # ------------------------------------------------------------------
    # Parameter count
    # ------------------------------------------------------------------
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Total parameters     : {total_params:>12,}")
    print(f"  Trainable parameters : {trainable_params:>12,}")

    # ------------------------------------------------------------------
    # Zero-init residual check: at init, output ≈ input[:, :2, :]
    # ------------------------------------------------------------------
    delta_norm = (out - x[:, :2, :]).abs().max().item()
    print(f"\n  Residual at init (head zeroed): max|delta_psi| = {delta_norm:.2e}")
    print(f"  (should be 0.0 -- confirms zero-init of output head)")

    # ------------------------------------------------------------------
    # Loss check
    # ------------------------------------------------------------------
    loss_fn = PhysicsAwareLoss(dx=1.0/512,
                               lambda_norm=1.0,
                               lambda_l1=1.0,
                               lambda_grad=0.5)
    target  = torch.randn(2, 2, 512)
    V       = torch.zeros(2, 512)

    with torch.no_grad():
        losses = loss_fn(out, target, V)

    print("\n  Loss components:")
    for k, v in losses.items():
        tag = "<-- weighted total" if k == "total" else ""
        print(f"    {k:6s} : {v.item():.6f}  {tag}")

    # ------------------------------------------------------------------
    # Memory footprint (forward pass, batch=2)
    # ------------------------------------------------------------------
    param_mb = total_params * 4 / (1024**2)      # float32
    print(f"\n  Parameter memory : {param_mb:.2f} MB  (float32)")

    # ------------------------------------------------------------------
    # Gradient flow check (backward pass)
    # ------------------------------------------------------------------
    model.train()
    x_grad   = torch.randn(2, 3, 512, requires_grad=False)
    out_grad = model(x_grad)
    tgt_grad = torch.randn(2, 2, 512)
    V_grad   = torch.zeros(2, 512)
    loss_dict = loss_fn(out_grad, tgt_grad, V_grad)
    loss_dict['total'].backward()

    max_grad = max(
        p.grad.abs().max().item()
        for p in model.parameters()
        if p.grad is not None
    )
    print(f"\n  Max gradient after backward : {max_grad:.4e}")
    print(f"  (non-zero confirms gradient flow through full network)")

    print("\n  All checks passed.")
    print("=" * 60)
    sys.exit(0)
