"""
train.py
========
Training pipeline for the UNet1D TDSE surrogate model.

This script trains UNet1D to predict psi(t + skip*dt) from psi(t) using the
PhysicsAwareLoss (L1 fidelity + norm conservation + gradient smoothness).

Features
--------
* Mixed-precision training (torch.cuda.amp) when CUDA is available
* Cosine annealing LR schedule with linear warm-up
* Best-checkpoint saving based on validation total loss
* Per-epoch CSV logging to checkpoints/train_log.csv
* Early stopping (patience configurable via --patience)
* Full CLI via argparse — run with --help for all options

Usage
-----
    # default: uses data/train.h5 and data/val.h5
    python train.py

    # custom paths / hyperparameters
    python train.py --train data/train.h5 --val data/val.h5 \\
                    --epochs 100 --batch 64 --lr 3e-4 \\
                    --ckpt checkpoints/best.pt

    # CPU-only explicit
    python train.py --device cpu
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from tdse_padl.data import QuantumDataset
from tdse_padl.models import UNet1D, PhysicsAwareLoss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(requested: str) -> torch.device:
    """Resolve device string, falling back gracefully."""
    if requested == 'cuda' and not torch.cuda.is_available():
        print("[train] CUDA requested but not available — falling back to CPU.")
        return torch.device('cpu')
    return torch.device(requested)


def make_dataloader(
    h5_path: str,
    batch_size: int,
    augment: bool,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader from an HDF5 dataset file."""
    dataset = QuantumDataset(h5_path, augment=augment)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
) -> LambdaLR:
    """
    Linear warm-up then cosine annealing learning-rate schedule.

    During the first `warmup_epochs` epochs the LR increases linearly from
    0 to its base value.  Afterwards it decays following a cosine curve down
    to 0 at `total_epochs`.
    """
    import math

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs + 1)
        progress = float(epoch - warmup_epochs) / float(
            max(1, total_epochs - warmup_epochs)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def count_params(model: nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Training / validation loops
# ---------------------------------------------------------------------------

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn:   PhysicsAwareLoss,
    device:    torch.device,
    scaler:    GradScaler,
    use_amp:   bool,
) -> dict[str, float]:
    """
    Run one full training epoch.

    Returns averaged loss components as a plain dict of floats.
    """
    model.train()
    totals: dict[str, float] = {'l1': 0., 'norm': 0., 'grad': 0., 'total': 0.}
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)   # (B, 3, N)
        y = y.to(device, non_blocking=True)   # (B, 2, N)
        V = x[:, 2, :]                        # (B, N) — potential channel

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            pred   = model(x)                 # (B, 2, N)
            losses = loss_fn(pred, y, V)

        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def val_epoch(
    model:   nn.Module,
    loader:  DataLoader,
    loss_fn: PhysicsAwareLoss,
    device:  torch.device,
    use_amp: bool,
) -> dict[str, float]:
    """
    Run one full validation epoch (no gradients).

    Returns averaged loss components.
    """
    model.eval()
    totals: dict[str, float] = {'l1': 0., 'norm': 0., 'grad': 0., 'total': 0.}
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        V = x[:, 2, :]

        with autocast(device_type=device.type, enabled=use_amp):
            pred   = model(x)
            losses = loss_fn(pred, y, V)

        for k in totals:
            totals[k] += losses[k].item()
        n_batches += 1

    return {k: v / n_batches for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    device  = get_device(args.device)
    use_amp = (device.type == 'cuda') and args.amp
    ckpt_dir = Path(args.ckpt).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  TDSE-PADL  |  UNet1D Training")
    print("=" * 65)
    print(f"  Device      : {device}")
    print(f"  AMP         : {use_amp}")
    print(f"  Train file  : {args.train}")
    print(f"  Val file    : {args.val}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch}")
    print(f"  Base LR     : {args.lr}")
    print(f"  Checkpoint  : {args.ckpt}")
    print(f"  Patience    : {args.patience}")
    print()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_loader = make_dataloader(
        args.train, args.batch,
        augment=True, num_workers=args.workers, shuffle=True,
    )
    val_loader = make_dataloader(
        args.val, args.batch,
        augment=False, num_workers=args.workers, shuffle=False,
    )
    print(f"  Train batches : {len(train_loader):,}  "
          f"({len(train_loader.dataset):,} samples)")
    print(f"  Val   batches : {len(val_loader):,}  "
          f"({len(val_loader.dataset):,} samples)")
    print()

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = UNet1D().to(device)
    total_p, train_p = count_params(model)
    print(f"  Parameters : {total_p:,} total  |  {train_p:,} trainable")
    print(f"  Model size : {total_p * 4 / 1024**2:.1f} MB  (float32)")
    print()

    # ------------------------------------------------------------------
    # Optimiser, scheduler, loss, scaler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = cosine_warmup_schedule(
        optimizer,
        warmup_epochs=args.warmup,
        total_epochs=args.epochs,
    )
    dx      = 1.0 / 512
    loss_fn = PhysicsAwareLoss(
        dx=dx,
        lambda_l1=args.lambda_l1,
        lambda_norm=args.lambda_norm,
        lambda_grad=args.lambda_grad,
    )
    scaler = GradScaler(device.type, enabled=use_amp)

    # ------------------------------------------------------------------
    # CSV log
    # ------------------------------------------------------------------
    log_path = ckpt_dir / "train_log.csv"
    log_fields = [
        'epoch', 'lr',
        'train_total', 'train_l1', 'train_norm', 'train_grad',
        'val_total',   'val_l1',   'val_norm',   'val_grad',
        'epoch_time_s',
    ]
    log_file = open(log_path, 'w', newline='')
    log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
    log_writer.writeheader()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val_loss   = float('inf')
    patience_count  = 0
    best_epoch      = 0

    print(f"  {'Ep':>4}  {'LR':>8}  "
          f"{'Tr-total':>9}  {'Tr-L1':>8}  {'Tr-norm':>8}  {'Tr-grad':>8}  | "
          f"{'Va-total':>9}  {'Va-L1':>8}  {'Va-norm':>8}  {'Va-grad':>8}  "
          f"{'Time':>6}")
    print("  " + "-" * 117)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, device, scaler, use_amp
        )
        val_losses = val_epoch(
            model, val_loader, loss_fn, device, use_amp
        )

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        # --- console row ---
        print(
            f"  {epoch:>4}  {lr_now:>8.2e}  "
            f"{train_losses['total']:>9.5f}  "
            f"{train_losses['l1']:>8.5f}  "
            f"{train_losses['norm']:>8.5f}  "
            f"{train_losses['grad']:>8.5f}  | "
            f"{val_losses['total']:>9.5f}  "
            f"{val_losses['l1']:>8.5f}  "
            f"{val_losses['norm']:>8.5f}  "
            f"{val_losses['grad']:>8.5f}  "
            f"{elapsed:>5.1f}s"
        )

        # --- CSV log ---
        log_writer.writerow({
            'epoch':         epoch,
            'lr':            f"{lr_now:.6e}",
            'train_total':   f"{train_losses['total']:.6f}",
            'train_l1':      f"{train_losses['l1']:.6f}",
            'train_norm':    f"{train_losses['norm']:.6f}",
            'train_grad':    f"{train_losses['grad']:.6f}",
            'val_total':     f"{val_losses['total']:.6f}",
            'val_l1':        f"{val_losses['l1']:.6f}",
            'val_norm':      f"{val_losses['norm']:.6f}",
            'val_grad':      f"{val_losses['grad']:.6f}",
            'epoch_time_s':  f"{elapsed:.1f}",
        })
        log_file.flush()

        # --- checkpoint ---
        val_total = val_losses['total']
        if val_total < best_val_loss:
            best_val_loss  = val_total
            best_epoch     = epoch
            patience_count = 0
            torch.save({
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_loss':        best_val_loss,
                'args':            vars(args),
            }, args.ckpt)
            print(f"  [*] New best val loss {best_val_loss:.5f} — saved to {args.ckpt}")
        else:
            patience_count += 1

        # --- early stopping ---
        if args.patience > 0 and patience_count >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs).")
            break

    log_file.close()

    print("  " + "-" * 117)
    print(f"\n  Training complete.")
    print(f"  Best val loss : {best_val_loss:.5f}  (epoch {best_epoch})")
    print(f"  Checkpoint    : {args.ckpt}")
    print(f"  Log           : {log_path}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train UNet1D TDSE surrogate model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    p.add_argument('--train',   default='data/train.h5',          help='Training HDF5 file')
    p.add_argument('--val',     default='data/val.h5',             help='Validation HDF5 file')
    p.add_argument('--workers', default=0,   type=int,             help='DataLoader worker processes')
    # training
    p.add_argument('--epochs',  default=100, type=int,             help='Number of training epochs')
    p.add_argument('--batch',   default=64,  type=int,             help='Batch size')
    p.add_argument('--lr',      default=3e-4, type=float,          help='Base learning rate (AdamW)')
    p.add_argument('--weight-decay', default=1e-4, type=float,     help='AdamW weight decay')
    p.add_argument('--warmup',  default=5,   type=int,             help='Linear warm-up epochs')
    p.add_argument('--patience',default=20,  type=int,             help='Early-stop patience (0=off)')
    # loss weights
    p.add_argument('--lambda-l1',   default=1.0, type=float,       help='L1 loss weight')
    p.add_argument('--lambda-norm', default=1.0, type=float,       help='Norm conservation weight')
    p.add_argument('--lambda-grad', default=0.5, type=float,       help='Gradient smoothness weight')
    # hardware
    p.add_argument('--device',  default='cuda',                    help='Device: cuda or cpu')
    p.add_argument('--amp',     action='store_true', default=True,  help='Use mixed-precision (AMP)')
    p.add_argument('--no-amp',  dest='amp', action='store_false',  help='Disable AMP')
    # output
    p.add_argument('--ckpt',    default='checkpoints/best.pt',     help='Best checkpoint save path')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
