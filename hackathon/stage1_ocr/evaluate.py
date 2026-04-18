"""
evaluate.py — OCRNet per-noise-profile accuracy report

Runs inference on the MNIST val split under three noise conditions:
  • none        — clean images, no noise
  • snp         — salt-and-pepper (prob=0.05)
  • gaussian    — additive Gaussian (std=25)

Prints a formatted accuracy table and flags whether each profile
clears the 95% submission gate.

Usage (from hackathon root):
    python stage1_ocr/evaluate.py \
        --weights-snp   weights/ocr_snp.pth \
        --weights-clean weights/ocr_clean.pth \
        --data          ./data

Or from anywhere in Colab:
    python /content/drive/MyDrive/hackathon/stage1_ocr/evaluate.py \
        --weights-snp   /content/drive/MyDrive/hackathon/weights/ocr_snp.pth \
        --weights-clean /content/drive/MyDrive/hackathon/weights/ocr_clean.pth \
        --data          /content/drive/MyDrive/hackathon/data
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# ── Path setup ───────────────────────────────────────────────────────────────
HACKATHON_ROOT = "/content/drive/MyDrive/hackathon"
_dynamic_root  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for _root in [HACKATHON_ROOT, _dynamic_root]:
    if _root not in sys.path:
        sys.path.insert(0, _root)

from stage1_ocr.denoiser.model   import OCRNet
from stage1_ocr.denoiser.pipeline import MedianDnCNNPipeline

# ── Constants ─────────────────────────────────────────────────────────────────
MNIST_MEAN   = 0.1307
MNIST_STD    = 0.3081
ACCURACY_GATE = 0.95   # 95% — submission requirement


# ----------------------------------------------------------------------------
# Noise injection (mirrors dataset.py exactly — same RNG, same seed logic)
# ----------------------------------------------------------------------------

def inject_snp(img_pil: Image.Image, prob: float, rng: np.random.Generator) -> Image.Image:
    arr  = np.array(img_pil, dtype=np.uint8).flatten()
    n    = len(arr)
    k    = int(n * prob)
    arr[rng.choice(n, k // 2, replace=False)] = 255
    arr[rng.choice(n, k // 2, replace=False)] = 0
    return Image.fromarray(arr.reshape(28, 28), mode="L")


def inject_gaussian(img_pil: Image.Image, std: float, rng: np.random.Generator) -> Image.Image:
    arr   = np.array(img_pil, dtype=np.float32)
    noisy = np.clip(arr + rng.normal(0, std, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="L")


# ----------------------------------------------------------------------------
# Single-profile evaluator
# ----------------------------------------------------------------------------

@torch.no_grad()
def evaluate_profile(
    model:        nn.Module,
    pipeline,                     # MedianDnCNNPipeline or None
    noise_type:   str,            # "none" | "snp" | "gaussian"
    snp_prob:     float,
    gauss_std:    float,
    data_root:    str,
    device:       torch.device,
    num_samples:  int,
    seed:         int,
) -> dict:
    """
    Runs inference on `num_samples` val images under one noise profile.

    Returns dict with keys: noise_type, correct, total, accuracy, per_class
    """
    model.eval()

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(MNIST_MEAN,), std=(MNIST_STD,)),
    ])

    mnist_val = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=None,
    )

    n         = min(num_samples, len(mnist_val))
    correct   = 0
    per_class = {i: {"correct": 0, "total": 0} for i in range(10)}

    for idx in range(n):
        img_pil, label = mnist_val[idx]
        img_pil = img_pil.convert("L")

        # ── Noise injection ──────────────────────────────────────────────────
        rng = np.random.default_rng(seed=seed + idx)

        if noise_type == "snp":
            noisy_pil = inject_snp(img_pil, snp_prob, rng)
        elif noise_type == "gaussian":
            noisy_pil = inject_gaussian(img_pil, gauss_std, rng)
        else:
            noisy_pil = img_pil   # clean — no noise

        # ── Denoising ────────────────────────────────────────────────────────
        if pipeline is not None and noise_type != "none":
            denoised = pipeline(noisy_pil)            # (1, H, W) float [0,1]
            input_pil = transforms.ToPILImage()(denoised)
        else:
            input_pil = noisy_pil

        # ── Inference ────────────────────────────────────────────────────────
        tensor = normalize(input_pil).unsqueeze(0).to(device)   # (1,1,28,28)
        logits = model(tensor)
        pred   = logits.argmax(dim=1).item()

        if pred == label:
            correct += 1
            per_class[label]["correct"] += 1
        per_class[label]["total"] += 1

    return {
        "noise_type": noise_type,
        "correct":    correct,
        "total":      n,
        "accuracy":   correct / n,
        "per_class":  per_class,
    }


# ----------------------------------------------------------------------------
# Load checkpoint helper
# ----------------------------------------------------------------------------

def load_model(weights_path: str, device: torch.device) -> nn.Module:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            f"Run train.py first to generate them."
        )
    ckpt  = torch.load(weights_path, map_location=device)
    model = OCRNet(dropout=0.0, num_classes=10).to(device)  # dropout=0 for eval
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    trained_acc = ckpt.get("val_acc", "unknown")
    trained_ep  = ckpt.get("epoch",   "unknown")
    print(f"    Loaded  : {weights_path}")
    print(f"    Trained : epoch {trained_ep},  val_acc={float(trained_acc)*100:.2f}%")
    return model


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(args):
    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("\n" + "=" * 65)
    print("  OCRNet — Noise Profile Accuracy Report")
    print("=" * 65)
    print(f"  Device   : {device}")
    print(f"  Samples  : {args.num_samples} val images per profile")
    print(f"  Gate     : {ACCURACY_GATE*100:.0f}% accuracy required on all profiles")
    print()

    # ── Load denoising pipeline ───────────────────────────────────────────────
    print("  Loading denoiser pipeline...")
    pipeline = MedianDnCNNPipeline(
        weights_path=args.dncnn_weights,
        device=str(device),
    )
    print()

    # ── Define profiles to evaluate ───────────────────────────────────────────
    # Each profile specifies which model weights to use and which noise to apply.
    # Fall back to snp weights if clean weights aren't provided.
    profiles = [
        {
            "label":       "Clean (no noise)",
            "noise_type":  "none",
            # Use clean weights if provided AND the file exists, else fall back to snp
            "weights":     args.weights_clean
                           if args.weights_clean and os.path.exists(args.weights_clean)
                           else args.weights_snp,
        },
        {
            "label":       "Salt & Pepper (prob=0.05)",
            "noise_type":  "snp",
            "weights":     args.weights_snp,
        },
        {
            "label":       "Gaussian (std=25)",
            "noise_type":  "gaussian",
            "weights":     args.weights_snp,   # same model handles both
        },
    ]

    results = []

    for profile in profiles:
        print(f"  ── {profile['label']} ──")
        model = load_model(profile["weights"], device)

        result = evaluate_profile(
            model       = model,
            pipeline    = pipeline,
            noise_type  = profile["noise_type"],
            snp_prob    = args.snp_prob,
            gauss_std   = args.gauss_std,
            data_root   = args.data,
            device      = device,
            num_samples = args.num_samples,
            seed        = args.seed,
        )
        result["label"] = profile["label"]
        results.append(result)
        print(f"    Accuracy : {result['accuracy']*100:.2f}%  ({result['correct']}/{result['total']})")
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  {'Noise Profile':<30}  {'Accuracy':>9}  {'Correct':>8}  {'Gate'}")
    print(f"  {'-' * 61}")

    all_passed = True
    for r in results:
        passed  = r["accuracy"] >= ACCURACY_GATE
        gate    = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False
        print(
            f"  {r['label']:<30}  {r['accuracy']*100:>8.2f}%  "
            f"{r['correct']:>4}/{r['total']:<4}  {gate}"
        )

    print(f"  {'-' * 61}")

    if all_passed:
        print(f"\n  ✓  All profiles cleared {ACCURACY_GATE*100:.0f}% gate.")
        print("     Paste this table into README.md and move on to Stage 2.")
    else:
        print(f"\n  ✗  One or more profiles below {ACCURACY_GATE*100:.0f}% gate.")
        print("     Fix: re-train with --noise both --epochs 50 --dropout 0.3")

    # ── Per-class breakdown ───────────────────────────────────────────────────
    if args.per_class:
        print("\n" + "=" * 65)
        print("  Per-class accuracy breakdown")
        print("=" * 65)
        header = f"  {'Digit':<6}" + "".join(f"  {r['noise_type']:>10}" for r in results)
        print(header)
        print(f"  {'-' * 55}")
        for digit in range(10):
            row = f"  {digit:<6}"
            for r in results:
                pc  = r["per_class"][digit]
                acc = pc["correct"] / pc["total"] if pc["total"] > 0 else 0.0
                row += f"  {acc*100:>9.1f}%"
            print(row)

    print()
    return all_passed


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate OCRNet across all noise profiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Weights
    parser.add_argument("--weights-snp",   required=True,
                        help="Path to ocr_snp.pth (used for S&P and Gaussian profiles)")
    parser.add_argument("--weights-clean", default=None,
                        help="Path to ocr_clean.pth (used for clean profile; falls back to --weights-snp)")
    parser.add_argument("--dncnn-weights", default="./weights/dncnn.pth",
                        help="Path to dncnn.pth")

    # Data
    parser.add_argument("--data",          default="./data",
                        help="MNIST data root")
    parser.add_argument("--num-samples",   type=int, default=10000,
                        help="Val images to evaluate per profile (max 10000)")

    # Noise params
    parser.add_argument("--snp-prob",      type=float, default=0.05)
    parser.add_argument("--gauss-std",     type=float, default=25.0)

    # Output
    parser.add_argument("--per-class",     action="store_true",
                        help="Also print per-digit accuracy breakdown")

    # Misc
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--device",        default=None)

    args = parser.parse_args()
    passed = main(args)
    sys.exit(0 if passed else 1)
