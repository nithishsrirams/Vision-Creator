import os
import sys
import time
import math

import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# Make sure we can import DnCNN whether this script is run directly
# or as part of the package (python -m denoiser.train_dncnn)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from stage1_ocr.denoiser.dncnn import DnCNN


# ----------------------------------------------------------------------------
# Device setup
#
# We check for CUDA first (Nvidia GPU), then MPS (Apple Silicon), then
# fall back to CPU. Training on CPU is totally fine but will take longer —
# roughly 3-4x more time than a mid-range GPU.
# ----------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        print(f"  Using GPU: {name}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("  No GPU found — training on CPU (slower but works fine)")
    return device


# ----------------------------------------------------------------------------
# NoisyMNIST Dataset
#
# This dataset wrapper does one simple thing: takes a clean MNIST image
# and returns it alongside a noisy version of itself.
#
# The key design choice is that sigma (noise strength) is sampled
# randomly per image from [sigma_min, sigma_max]. This trains the network
# to handle a range of noise levels rather than a single fixed one —
# called "blind denoising". It's more useful in practice because real
# scanned documents don't all have the same noise level.
#
# Sigma values are in [5, 55] / 255, following the original DnCNN paper.
# sigma=5/255 is barely perceptible noise; sigma=55/255 is quite heavy.
# ----------------------------------------------------------------------------

class NoisyMNIST(Dataset):
    """
    Wraps MNIST and returns (noisy_image, clean_image) pairs.

    Every time __getitem__ is called, a fresh noise sample is drawn —
    so each epoch the model sees the same image with different noise.
    This is free data augmentation and helps the model generalize.
    """

    def __init__(
        self,
        root="./data",
        train=True,
        sigma_min=5,
        sigma_max=55,
        download=True,
    ):
        # Convert sigma from the [0, 255] scale to [0.0, 1.0] scale.
        # We keep images in float [0,1] throughout, so noise std must match.
        self.sigma_min = sigma_min / 255.0
        self.sigma_max = sigma_max / 255.0

        # Load MNIST with just ToTensor — no normalization.
        # DnCNN works on raw [0, 1] pixel values, not zero-centered ones.
        # Adding mean/std normalization here would mess up the noise scale.
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=transforms.ToTensor(),
        )

        self.train = train

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        clean, _ = self.mnist[idx]      # clean shape: (1, 28, 28), values in [0, 1]
                                        # we discard the label — denoising is unsupervised

        # Sample a random noise level for this specific image.
        # During training we want variation. During validation we fix
        # sigma to get consistent, comparable metrics across epochs.
        if self.train:
            sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        else:
            # Validation: use sigma=25/255 — a moderate, representative level.
            # This matches what the original DnCNN paper uses for AWGN evaluation.
            sigma = 25.0 / 255.0

        # Generate Gaussian noise and add it to the clean image.
        # torch.randn_like gives us a tensor of the same shape as clean,
        # filled with standard normal samples (mean=0, std=1).
        # Multiplying by sigma scales that to our desired noise level.
        noise = torch.randn_like(clean) * sigma
        noisy = torch.clamp(clean + noise, 0.0, 1.0)

        # We return both noisy and clean so the loss can compare them.
        # The training loop computes: loss = MSE(model(noisy), clean)
        return noisy, clean


# ----------------------------------------------------------------------------
# Training metrics helpers
#
# MSE loss is what we optimize, but PSNR (Peak Signal-to-Noise Ratio)
# is what we report — it's a more interpretable number.
#
# PSNR = 10 * log10(1 / MSE)
# Higher is better. Values in dB:
#   < 25 dB — noticeable degradation
#   25-30 dB — acceptable
#   > 30 dB — good, often visually indistinguishable
#
# We're aiming for PSNR > 28 dB on the validation set before we trust
# this denoiser enough to pass its output into OCRNet.
# ----------------------------------------------------------------------------

def mse_to_psnr(mse_value):
    """Convert mean squared error to PSNR in decibels."""
    if mse_value == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse_value)


# ----------------------------------------------------------------------------
# One training epoch
# ----------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    num_batches = len(loader)

    for batch_idx, (noisy, clean) in enumerate(loader):
        noisy = noisy.to(device)
        clean = clean.to(device)

        # Standard forward → loss → backward → step
        optimizer.zero_grad()
        denoised = model(noisy)
        loss = criterion(denoised, clean)
        loss.backward()

        # Gradient clipping helps avoid occasional large gradient spikes,
        # especially in early training when weights are still far from optimal.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

        # Print progress every 100 batches so we can see training is alive
        if (batch_idx + 1) % 100 == 0:
            avg_so_far = total_loss / (batch_idx + 1)
            psnr = mse_to_psnr(avg_so_far)
            print(
                f"    batch {batch_idx+1:>4}/{num_batches}"
                f"  loss={avg_so_far:.6f}"
                f"  PSNR={psnr:.2f} dB"
            )

    avg_loss = total_loss / num_batches
    return avg_loss


# ----------------------------------------------------------------------------
# Validation pass — no gradients, fixed sigma=25/255
# ----------------------------------------------------------------------------

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    for noisy, clean in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        denoised = model(noisy)
        loss = criterion(denoised, clean)
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    return avg_loss


# ----------------------------------------------------------------------------
# Main training function
#
# Hyperparameter choices explained:
#
# epochs=50      — DnCNN converges slowly. 30 epochs is usable, 50 is solid.
#                  The LR drops at 30 and 40 to do fine-tuning in the last stretch.
#
# batch_size=128 — Larger batches = more stable gradients for BN layers.
#                  128 is the sweet spot for MNIST-sized images on most GPUs.
#                  Drop to 64 if you run out of memory.
#
# lr=1e-3        — Adam's default. DnCNN trains well with this.
#                  The scheduler cuts it to 1e-4 at epoch 30, then 1e-5 at 40.
#
# MSELoss        — Mean squared error on pixel values. This is the standard
#                  loss for image restoration. It doesn't directly optimize
#                  PSNR but MSE and PSNR are mathematically equivalent
#                  (PSNR is just a log transform of MSE), so minimizing one
#                  maximizes the other.
# ----------------------------------------------------------------------------

def train(
    data_root="./data",
    save_dir="./weights",
    epochs=50,
    batch_size=128,
    lr=1e-3,
    num_workers=2,
    sigma_min=5,
    sigma_max=55,
):
    print("\n" + "="*60)
    print("  DnCNN Training")
    print("="*60)

    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\nLoading MNIST...")
    train_dataset = NoisyMNIST(
        root=data_root,
        train=True,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )
    val_dataset = NoisyMNIST(
        root=data_root,
        train=False,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    print(f"  Train samples : {len(train_dataset):,}")
    print(f"  Val samples   : {len(val_dataset):,}")
    print(f"  Noise sigma   : [{sigma_min}, {sigma_max}] / 255")
    print(f"  Val sigma     : fixed at 25/255")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        # drop_last=True keeps batch sizes consistent for BatchNorm,
        # which behaves poorly on very small final batches
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = DnCNN(num_layers=17, num_channels=64).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: DnCNN  ({total_params:,} trainable parameters)")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = nn.MSELoss(reduction="mean")
    optimizer = Adam(model.parameters(), lr=lr)

    # Drop LR by 10x at epoch 30 and again at epoch 40.
    # This is the schedule from the original paper — coarse learning early,
    # fine-tuning in the last stretch.
    scheduler = MultiStepLR(optimizer, milestones=[30, 40], gamma=0.1)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nTraining for {epochs} epochs  (LR drops at epoch 30 and 40)")
    print("-" * 60)

    best_val_loss = float("inf")
    best_psnr     = 0.0
    save_path     = os.path.join(save_dir, "dncnn.pth")

    history = []   # store per-epoch stats for the final summary table

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        current_lr  = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch:02d}/{epochs}  (lr={current_lr:.2e})")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = validate(model, val_loader, criterion, device)

        train_psnr = mse_to_psnr(train_loss)
        val_psnr   = mse_to_psnr(val_loss)
        epoch_time = time.time() - epoch_start

        # Step the scheduler after each epoch (not after each batch)
        scheduler.step()

        # Save the best checkpoint based on validation loss.
        # We compare on loss rather than PSNR — they're equivalent but
        # loss is more numerically stable to compare.
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_psnr     = val_psnr
            torch.save(model.state_dict(), save_path)

        marker = " <-- best" if improved else ""
        print(
            f"  train  loss={train_loss:.6f}  PSNR={train_psnr:.2f} dB\n"
            f"  val    loss={val_loss:.6f}  PSNR={val_psnr:.2f} dB"
            f"  [{epoch_time:.0f}s]{marker}"
        )

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "val_loss":   val_loss,
            "val_psnr":   val_psnr,
        })

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"  Training complete")
    print(f"  Best val loss : {best_val_loss:.6f}")
    print(f"  Best val PSNR : {best_psnr:.2f} dB")
    print(f"  Weights saved : {save_path}")

    # Print a clean epoch table for the README
    print("\n  Epoch summary (every 5 epochs):")
    print(f"  {'Epoch':>5}  {'Train PSNR':>12}  {'Val PSNR':>10}  {'Val Loss':>12}")
    print(f"  {'-'*45}")
    for row in history:
        if row["epoch"] % 5 == 0 or row["epoch"] == 1:
            print(
                f"  {row['epoch']:>5}"
                f"  {row['train_psnr']:>10.2f} dB"
                f"  {row['val_psnr']:>8.2f} dB"
                f"  {row['val_loss']:>12.6f}"
            )

    # Warn if we didn't hit the quality bar we need for OCRNet
    if best_psnr < 28.0:
        print(
            f"\n  WARNING: Best PSNR ({best_psnr:.2f} dB) is below the 28 dB target."
            f"\n  Consider training for more epochs or lowering sigma_max."
        )
    else:
        print(f"\n  PSNR target (>28 dB) passed — denoiser is ready for OCRNet.")

    return save_path


# ----------------------------------------------------------------------------
# Quick visual sanity check — run after training to see if the denoiser
# is actually doing something useful before plugging it into OCRNet.
#
# This loads the saved weights, runs one batch through, and prints pixel
# statistics. It doesn't display images (no GUI assumed), but the numbers
# tell you what you need to know.
# ----------------------------------------------------------------------------

@torch.no_grad()
def quick_visual_check(weights_path, data_root="./data"):
    print("\n" + "="*60)
    print("  Post-training sanity check")
    print("="*60)

    device = get_device()
    model  = DnCNN(num_layers=17, num_channels=64).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Grab a single validation batch
    val_dataset = NoisyMNIST(root=data_root, train=False)
    val_loader  = DataLoader(val_dataset, batch_size=8, shuffle=True)
    noisy_batch, clean_batch = next(iter(val_loader))

    noisy_batch = noisy_batch.to(device)
    clean_batch = clean_batch.to(device)
    denoised_batch = model(noisy_batch)

    # Compute per-image PSNR
    psnr_noisy    = []
    psnr_denoised = []

    for i in range(noisy_batch.size(0)):
        mse_before = torch.mean((noisy_batch[i] - clean_batch[i]) ** 2).item()
        mse_after  = torch.mean((denoised_batch[i] - clean_batch[i]) ** 2).item()
        psnr_noisy.append(mse_to_psnr(mse_before))
        psnr_denoised.append(mse_to_psnr(mse_after))

    avg_before = sum(psnr_noisy)    / len(psnr_noisy)
    avg_after  = sum(psnr_denoised) / len(psnr_denoised)
    improvement = avg_after - avg_before

    print(f"\n  Avg PSNR before denoising : {avg_before:.2f} dB")
    print(f"  Avg PSNR after  denoising : {avg_after:.2f} dB")
    print(f"  Improvement               : +{improvement:.2f} dB")

    if improvement > 3.0:
        print("\n  Denoiser is working well.")
    elif improvement > 1.0:
        print("\n  Denoiser is helping, but consider more training.")
    else:
        print("\n  WARNING: Very small improvement. Check training ran correctly.")

    # Also check that output values stay in valid [0,1] range
    out_min = denoised_batch.min().item()
    out_max = denoised_batch.max().item()
    print(f"\n  Output pixel range: [{out_min:.4f}, {out_max:.4f}]  (should be within [0, 1])")


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DnCNN denoiser on MNIST")
    parser.add_argument("--data_root",  type=str,   default="./data",    help="Path to dataset root")
    parser.add_argument("--save_dir",   type=str,   default="./weights", help="Where to save weights")
    parser.add_argument("--epochs",     type=int,   default=50,          help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=128,         help="Training batch size")
    parser.add_argument("--lr",         type=float, default=1e-3,        help="Initial learning rate")
    parser.add_argument("--workers",    type=int,   default=2,           help="DataLoader worker processes")
    parser.add_argument("--sigma_min",  type=int,   default=5,           help="Min noise sigma (out of 255)")
    parser.add_argument("--sigma_max",  type=int,   default=55,          help="Max noise sigma (out of 255)")
    parser.add_argument("--check_only", action="store_true",             help="Skip training, just run sanity check on saved weights")
    args = parser.parse_args()

    weights_path = os.path.join(args.save_dir, "dncnn.pth")

    if args.check_only:
        if not os.path.exists(weights_path):
            print(f"ERROR: No weights found at {weights_path}. Train first.")
            sys.exit(1)
        quick_visual_check(weights_path, data_root=args.data_root)
    else:
        saved_path = train(
            data_root=args.data_root,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.workers,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
        # Always run the sanity check after training completes
        quick_visual_check(saved_path, data_root=args.data_root)
