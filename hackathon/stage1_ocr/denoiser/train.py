import argparse
import os
import sys
import time

import torch
import torch.nn as nn

from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

HACKATHON_ROOT = "/content/drive/MyDrive/hackathon"
if HACKATHON_ROOT not in sys.path:
    sys.path.insert(0, HACKATHON_ROOT)

from stage1_ocr.denoiser.model   import OCRNet
from stage1_ocr.denoiser.dataset import get_dataloaders


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Returns fraction of correct predictions in this batch."""
    return (logits.argmax(dim=1) == labels).float().mean().item()


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ----------------------------------------------------------------------------
# One epoch of training
# ----------------------------------------------------------------------------

def train_one_epoch(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    epoch:     int,
    total_epochs: int,
) -> tuple[float, float]:
    """
    Runs one full pass over the training set.

    Returns:
        avg_loss : mean CrossEntropyLoss over all batches
        avg_acc  : mean accuracy over all batches
    """
    model.train()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

        # Progress print every 50 batches
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == n_batches:
            print(
                f"    Epoch {epoch}/{total_epochs} "
                f"[{batch_idx+1:>3}/{n_batches}]  "
                f"loss={total_loss/(batch_idx+1):.4f}  "
                f"acc={total_acc/(batch_idx+1)*100:.2f}%",
                end="\r",
            )

    print()  # newline after \r progress
    return total_loss / n_batches, total_acc / n_batches


# ----------------------------------------------------------------------------
# Validation pass
# ----------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model:     nn.Module,
    loader:    torch.utils.data.DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """
    Runs one full pass over the validation set with no gradient updates.

    Returns:
        avg_loss : mean CrossEntropyLoss
        avg_acc  : mean accuracy
    """
    model.eval()
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = len(loader)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

    return total_loss / n_batches, total_acc / n_batches


# ----------------------------------------------------------------------------
# Main training loop
# ----------------------------------------------------------------------------

def train(args):
    # ── Device ───────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 60)
    print("  OCRNet Training")
    print("=" * 60)
    print(f"  Device      : {device}")
    print(f"  Noise type  : {args.noise}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  LR (init)   : {args.lr}")
    print(f"  Dropout     : {args.dropout}")
    print(f"  Checkpoint  : {args.save_path}")
    print()

    # ── Data ─────────────────────────────────────────────────────────────────
    print("  Loading datasets...")
    train_loader, val_loader = get_dataloaders(
        root=args.data,
        weights_path=args.dncnn_weights,
        noise_type=args.noise,
        snp_prob=args.snp_prob,
        gauss_std=args.gauss_std,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=str(device),
        seed=args.seed,
    )
    print()

    # ── Model ────────────────────────────────────────────────────────────────
    model = OCRNet(dropout=args.dropout, num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters  : {total_params:,}")
    print()

    # ── Loss / Optimizer / Scheduler ─────────────────────────────────────────
    # CrossEntropyLoss = LogSoftmax + NLLLoss — do NOT add Softmax to model forward()
    criterion = nn.CrossEntropyLoss()

    # Adam with default betas (0.9, 0.999) — weight_decay adds mild L2 regularization
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # CosineAnnealingLR: decays LR from args.lr → eta_min over T_max epochs
    # Keeps LR from plateauing early while avoiding abrupt drops
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,   # floor at 1% of initial LR
    )

    # ── Checkpoint setup ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    best_val_acc  = 0.0
    best_epoch    = 0
    history       = []

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"  {'Ep':>3}  {'Train Loss':>10}  {'Train Acc':>9}  {'Val Loss':>9}  {'Val Acc':>8}  {'LR':>9}  {'Time':>7}  {'Note'}")
    print(f"  {'-'*80}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        elapsed = time.time() - t0
        note    = ""

        # Save checkpoint whenever val accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            torch.save(
                {
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "val_acc":      val_acc,
                    "val_loss":     val_loss,
                    "args":         vars(args),
                },
                args.save_path,
            )
            note = "✓ saved"

        history.append({
            "epoch":      epoch,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "lr":         current_lr,
        })

        print(
            f"  {epoch:>3}  {train_loss:>10.4f}  {train_acc*100:>8.2f}%  "
            f"{val_loss:>9.4f}  {val_acc*100:>7.2f}%  "
            f"{current_lr:>9.2e}  {format_time(elapsed):>7}  {note}"
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Training complete.")
    print(f"  Best val accuracy : {best_val_acc*100:.2f}%  (epoch {best_epoch})")
    print(f"  Checkpoint saved  : {args.save_path}")

    # Warn if still below the 95% accuracy gate
    if best_val_acc < 0.95:
        print()
        print("  ⚠  WARNING — best val acc is below the 95% gate.")
        print("     Try: more epochs, higher dropout, or noise_type='both'.")
    else:
        print()
        print("  ✓  95% accuracy gate passed — ready to run evaluate.py.")

    return history


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train OCRNet on denoised MNIST",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--data",         default="./data",
                        help="MNIST data root")
    parser.add_argument("--dncnn-weights",default="./weights/dncnn.pth",
                        help="Path to trained dncnn.pth")
    parser.add_argument("--save-path",    default="./weights/ocr_snp.pth",
                        help="Where to save the best checkpoint")

    # Noise
    parser.add_argument("--noise",        default="snp",
                        choices=["snp", "gaussian", "both", "none"],
                        help="Noise profile to train on")
    parser.add_argument("--snp-prob",     type=float, default=0.05,
                        help="Salt-and-pepper corruption fraction")
    parser.add_argument("--gauss-std",    type=float, default=25.0,
                        help="Gaussian noise std (0–255 scale)")

    # Training hyperparameters
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch-size",   type=int,   default=64)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--dropout",      type=float, default=0.5)
    parser.add_argument("--num-workers",  type=int,   default=0)
    parser.add_argument("--seed",         type=int,   default=42)

    # Device
    parser.add_argument("--device",       default=None,
                        help="Force device: cuda | cpu | mps")

    args = parser.parse_args()

    train(args)
