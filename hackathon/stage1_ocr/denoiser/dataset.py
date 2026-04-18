import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from .pipeline import MedianDnCNNPipeline


# ----------------------------------------------------------------------------
# Normalization constants — computed from clean MNIST training split.
# Applied AFTER denoising so the values match what OCRNet expects.
# ----------------------------------------------------------------------------
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081


# ----------------------------------------------------------------------------
# Noise injection helpers
# ----------------------------------------------------------------------------

def inject_snp(img_pil: Image.Image, prob: float, rng: np.random.Generator) -> Image.Image:
    """
    Inject salt-and-pepper noise into a grayscale PIL Image.

    Args:
        img_pil : Input PIL Image (mode "L")
        prob    : Fraction of pixels to corrupt (e.g. 0.05 = 5%)
        rng     : NumPy Generator for reproducibility

    Returns:
        Noisy PIL Image (mode "L")
    """
    arr  = np.array(img_pil, dtype=np.uint8).flatten()
    n    = len(arr)
    k    = int(n * prob)

    salt_idx   = rng.choice(n, k // 2, replace=False)
    pepper_idx = rng.choice(n, k // 2, replace=False)
    arr[salt_idx]   = 255
    arr[pepper_idx] = 0

    return Image.fromarray(arr.reshape(28, 28), mode="L")


def inject_gaussian(img_pil: Image.Image, std: float, rng: np.random.Generator) -> Image.Image:
    """
    Inject additive Gaussian noise into a grayscale PIL Image.

    Args:
        img_pil : Input PIL Image (mode "L")
        std     : Standard deviation of the Gaussian noise (0–255 scale)
        rng     : NumPy Generator for reproducibility

    Returns:
        Noisy PIL Image (mode "L"), clipped to [0, 255]
    """
    arr   = np.array(img_pil, dtype=np.float32)
    noise = rng.normal(loc=0.0, scale=std, size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="L")


# ----------------------------------------------------------------------------
# DenoisedMNIST Dataset
# ----------------------------------------------------------------------------

class DenoisedMNIST(Dataset):
    """
    MNIST dataset with on-the-fly noise injection and denoising.

    Pipeline per sample
    -------------------
    1. Load clean MNIST image (PIL, 28×28, mode "L")
    2. Inject noise (S&P or Gaussian, controlled by `noise_type`)
    3. Denoise through MedianDnCNNPipeline → float tensor (1, 28, 28) in [0, 1]
    4. Normalize with MNIST mean/std → ready for OCRNet

    Args:
        root         : Directory where MNIST data is stored / will be downloaded
        train        : True for training split, False for validation
        weights_path : Path to trained dncnn.pth weights
        noise_type   : "snp" | "gaussian" | "both" | "none"
                         "both"  → randomly picks S&P or Gaussian per sample
                         "none"  → skips noise injection (clean baseline)
        snp_prob     : Fraction of pixels corrupted for S&P  (default 0.05)
        gauss_std    : Std dev of Gaussian noise on 0–255 scale (default 25.0)
        device       : "cuda" | "cpu" | "mps" | None (auto-detect)
        seed         : RNG seed for reproducible noise injection
        download     : Download MNIST if not already present

    Returns (per __getitem__)
    -------------------------
        image : (1, 28, 28) float tensor, normalized
        label : int scalar (0–9)
    """

    NOISE_TYPES = {"snp", "gaussian", "both", "none"}

    def __init__(
        self,
        root:          str,
        train:         bool,
        weights_path:  str,
        noise_type:    str   = "snp",
        snp_prob:      float = 0.05,
        gauss_std:     float = 25.0,
        device:        str   = None,
        seed:          int   = 42,
        download:      bool  = True,
    ):
        if noise_type not in self.NOISE_TYPES:
            raise ValueError(
                f"noise_type must be one of {self.NOISE_TYPES}, got '{noise_type}'"
            )

        self.noise_type  = noise_type
        self.snp_prob    = snp_prob
        self.gauss_std   = gauss_std
        self.seed        = seed

        # Load base MNIST (clean, PIL images via ToTensor → we'll undo for noise injection)
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=download,
            transform=None,       # We handle transforms manually
        )

        # Denoising pipeline (loads DnCNN weights once, shared across all samples)
        self.pipeline = MedianDnCNNPipeline(
            weights_path=weights_path,
            device=device,
        )

        # Final normalization applied after denoising
        self.normalize = transforms.Compose([
            transforms.ToTensor(),           # PIL → (1, H, W) float [0, 1]
            transforms.Normalize(
                mean=(MNIST_MEAN,),
                std=(MNIST_STD,),
            ),
        ])

        # Per-sample RNG: seeded deterministically per index so noise is
        # reproducible across runs but different for every sample.
        # We store the base seed and derive per-index seeds in __getitem__.
        self._base_seed = seed

    def __len__(self) -> int:
        return len(self.mnist)

    def __getitem__(self, idx: int):
        """
        Returns:
            image : (1, 28, 28) normalized float tensor
            label : int
        """
        img_pil, label = self.mnist[idx]   # img_pil is PIL Image mode "L"

        if self.noise_type != "none":
            # Derive a per-sample seed so noise is deterministic and unique per index
            rng = np.random.default_rng(seed=self._base_seed + idx)

            chosen = self.noise_type
            if chosen == "both":
                chosen = rng.choice(["snp", "gaussian"])

            if chosen == "snp":
                noisy_pil = inject_snp(img_pil, prob=self.snp_prob, rng=rng)
            else:
                noisy_pil = inject_gaussian(img_pil, std=self.gauss_std, rng=rng)

            # Denoise: PIL → (1, 28, 28) float tensor in [0, 1]
            denoised_tensor = self.pipeline(noisy_pil)   # (1, H, W) cpu float
            denoised_pil    = transforms.ToPILImage()(denoised_tensor)
        else:
            # Clean baseline — skip noise and denoising entirely
            denoised_pil = img_pil

        # Normalize → ready for OCRNet
        image = self.normalize(denoised_pil)   # (1, 28, 28)

        return image, label


# ----------------------------------------------------------------------------
# DataLoader factory — convenience wrapper used by train.py and evaluate.py
# ----------------------------------------------------------------------------

def get_dataloaders(
    root:          str,
    weights_path:  str,
    noise_type:    str   = "snp",
    snp_prob:      float = 0.05,
    gauss_std:     float = 25.0,
    batch_size:    int   = 64,
    num_workers:   int   = 0,
    device:        str   = None,
    seed:          int   = 42,
):
    """
    Returns (train_loader, val_loader) for OCRNet training.

    Args:
        root         : MNIST data directory
        weights_path : Path to dncnn.pth
        noise_type   : "snp" | "gaussian" | "both" | "none"
        snp_prob     : S&P corruption fraction
        gauss_std    : Gaussian noise std on 0–255 scale
        batch_size   : Samples per batch
        num_workers  : DataLoader worker processes (0 = main process only)
        device       : Device string passed to pipeline
        seed         : RNG seed

    Returns:
        train_loader : DataLoader (shuffled)
        val_loader   : DataLoader (not shuffled)
    """
    shared = dict(
        root=root,
        weights_path=weights_path,
        noise_type=noise_type,
        snp_prob=snp_prob,
        gauss_std=gauss_std,
        device=device,
        seed=seed,
    )

    train_ds = DenoisedMNIST(train=True,  **shared)
    val_ds   = DenoisedMNIST(train=False, **shared)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"  Train samples : {len(train_ds):,}  ({len(train_loader)} batches)")
    print(f"  Val samples   : {len(val_ds):,}  ({len(val_loader)} batches)")

    return train_loader, val_loader


# ----------------------------------------------------------------------------
# Sanity check — run directly to verify pipeline wiring is correct
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    default="./weights/dncnn.pth")
    parser.add_argument("--data",       default="./data")
    parser.add_argument("--noise",      default="snp",
                        choices=["snp", "gaussian", "both", "none"])
    parser.add_argument("--samples",    type=int, default=5)
    args = parser.parse_args()

    print("\nDenoisedMNIST sanity check")
    print("=" * 40)

    ds = DenoisedMNIST(
        root=args.data,
        train=False,
        weights_path=args.weights,
        noise_type=args.noise,
    )

    print(f"Dataset length : {len(ds):,}")

    for i in range(args.samples):
        img, label = ds[i]
        assert img.shape == (1, 28, 28), f"Bad shape: {img.shape}"
        assert img.dtype == torch.float32
        print(f"  Sample {i} | label={label} | shape={img.shape} "
              f"| mean={img.mean():.4f} | std={img.std():.4f}")

    print("\nAll checks passed — dataset.py is ready.")
