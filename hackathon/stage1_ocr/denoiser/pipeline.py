import math
import os
import sys

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from stage1_ocr.denoiser.median_filter import MedianFilter
from stage1_ocr.denoiser.dncnn import DnCNN


# ----------------------------------------------------------------------------
# Why two stages instead of just DnCNN alone?
#
# DnCNN was designed for Gaussian noise — smoothly distributed random
# perturbations. Salt-and-pepper noise is structurally different: single
# pixels are set to extreme values (0 or 255) with clean surroundings.
#
# When DnCNN sees a salt spike, it tries to smooth it out, but a 3×3
# conv layer sees it as a strong localized signal, not clearly noise.
# It often partially corrects it and leaves a faint artifact behind.
#
# The median filter sees that exact spike as the outlier it is and
# removes it cleanly in one shot — no training required.
#
# So the split is:
#   Median filter  →  handles the S&P component (perfect tool for the job)
#   DnCNN          →  handles residual Gaussian-like noise the median left behind
#
# Running median first means DnCNN gets a much cleaner input and doesn't
# waste its capacity trying to handle noise it's not designed for.
# ----------------------------------------------------------------------------


class MedianDnCNNPipeline:
    """
    Two-stage denoising pipeline.

    Stage 1 (MedianFilter): removes salt-and-pepper spikes in PIL/uint8 space.
    Stage 2 (DnCNN):        removes residual noise in float tensor space.

    Usage:
        pipeline = MedianDnCNNPipeline(weights_path="weights/dncnn.pth")
        clean_tensor = pipeline(noisy_pil_image)   # returns (1, H, W) in [0, 1]

    The pipeline also exposes each stage separately if you need to inspect
    intermediate results (useful for debugging and the side-by-side print).
    """

    def __init__(
        self,
        weights_path: str,
        median_kernel: int = 3,
        device: str = None,
    ):
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        # Stage 1: classical median filter (no weights, always ready)
        self.median = MedianFilter(kernel_size=median_kernel)

        # Stage 2: DnCNN (needs trained weights)
        self.dncnn = DnCNN(num_layers=17, num_channels=64).to(self.device)

        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"DnCNN weights not found at: {weights_path}\n"
                f"Run train_dncnn.py first to generate them."
            )

        self.dncnn.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.dncnn.eval()

        # Reusable ToTensor transform — avoids re-instantiating it per call
        self.to_tensor = transforms.ToTensor()

        print(f"  Pipeline ready — device={self.device}, median_kernel={median_kernel}")

    @torch.no_grad()
    def __call__(self, img: Image.Image) -> torch.Tensor:
        """
        Denoise a single PIL Image through both stages.

        Args:
            img: Noisy input PIL Image (any mode, any size)
        Returns:
            clean: float tensor of shape (1, H, W), values in [0, 1]
        """
        # Ensure grayscale — OCRNet expects single-channel input
        img = img.convert("L")

        # Stage 1: median filter removes S&P spikes
        after_median = self.median(img)

        # Stage 2: DnCNN cleans up remaining Gaussian-like noise
        tensor = self.to_tensor(after_median).unsqueeze(0).to(self.device)  # (1,1,H,W)
        clean  = self.dncnn(tensor)                                          # (1,1,H,W)

        return clean.squeeze(0).cpu()   # (1, H, W)

    @torch.no_grad()
    def denoise_stage1_only(self, img: Image.Image) -> torch.Tensor:
        """
        Run only the median filter stage. Useful for ablation comparison:
        how much does DnCNN actually add on top of the median filter alone?
        """
        img = img.convert("L")
        after_median = self.median(img)
        return self.to_tensor(after_median)   # (1, H, W)

    @torch.no_grad()
    def denoise_batch(self, imgs: list) -> torch.Tensor:
        """
        Denoise a list of PIL Images in one call.

        Args:
            imgs: list of PIL Images
        Returns:
            tensor of shape (B, 1, H, W)
        """
        tensors = [self.__call__(img) for img in imgs]
        return torch.stack(tensors)   # (B, 1, H, W)


# ----------------------------------------------------------------------------
# Verification function
#
# This is the "PSNR on val set should exceed 28 dB" check from the plan.
# Run this after DnCNN training finishes to confirm the denoiser is ready
# before moving on to OCRNet training.
#
# It also prints the side-by-side comparison: noisy vs after-median
# vs after-DnCNN vs clean — so you can see what each stage contributes.
# ----------------------------------------------------------------------------

def mse_to_psnr(mse):
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def verify_pipeline(weights_path: str, data_root: str = "./data", num_samples: int = 200):
    """
    Verifies the full pipeline meets the 28 dB PSNR gate.

    Loads the val split of MNIST, injects S&P noise, runs all three
    denoising configs (none, median only, median+DnCNN), and reports
    PSNR for each. Also prints a side-by-side pixel comparison for a
    handful of sample images.

    Args:
        weights_path: Path to trained dncnn.pth
        data_root:    Where MNIST data lives
        num_samples:  How many val images to evaluate on
    """
    from torchvision import datasets

    print("\n" + "=" * 60)
    print("  Denoiser pipeline verification")
    print("=" * 60)

    # Load the pipeline
    pipeline = MedianDnCNNPipeline(weights_path=weights_path)
    to_tensor = transforms.ToTensor()

    # Load MNIST val split (clean ground truth)
    mnist_val = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    print(f"\n  Evaluating on {num_samples} val images (S&P noise, prob=0.05)")
    print(f"  {'Config':<25}  {'Avg PSNR':>10}  {'Min PSNR':>10}  {'Max PSNR':>10}")
    print(f"  {'-' * 60}")

    psnr_noisy      = []
    psnr_median     = []
    psnr_full       = []

    rng = np.random.default_rng(seed=0)

    for i in range(num_samples):
        clean_tensor, _ = mnist_val[i]           # (1, 28, 28) float [0,1]
        clean_pil = transforms.ToPILImage()(clean_tensor).convert("L")

        # Inject S&P noise manually so we have exact control
        arr = np.array(clean_pil, dtype=np.float32)
        flat = arr.flatten()
        n_pixels   = len(flat)
        n_corrupt  = int(n_pixels * 0.05)

        salt_idx   = rng.choice(n_pixels, n_corrupt // 2, replace=False)
        pepper_idx = rng.choice(n_pixels, n_corrupt // 2, replace=False)
        flat[salt_idx]   = 255.0
        flat[pepper_idx] = 0.0

        noisy_arr = flat.reshape(28, 28).astype(np.uint8)
        noisy_pil = Image.fromarray(noisy_arr, mode="L")
        noisy_tensor = to_tensor(noisy_pil)      # (1, 28, 28)

        # Config A: no denoising
        mse_a = float(torch.mean((noisy_tensor - clean_tensor) ** 2))
        psnr_noisy.append(mse_to_psnr(mse_a))

        # Config B: median filter only
        after_median = pipeline.denoise_stage1_only(noisy_pil)
        mse_b = float(torch.mean((after_median - clean_tensor) ** 2))
        psnr_median.append(mse_to_psnr(mse_b))

        # Config C: full pipeline (median + DnCNN)
        after_full = pipeline(noisy_pil)         # (1, 28, 28)
        mse_c = float(torch.mean((after_full - clean_tensor) ** 2))
        psnr_full.append(mse_to_psnr(mse_c))

    def summary(label, values):
        avg = sum(values) / len(values)
        mn  = min(values)
        mx  = max(values)
        print(f"  {label:<25}  {avg:>10.2f}  {mn:>10.2f}  {mx:>10.2f}")
        return avg

    avg_noisy  = summary("Noisy (no denoising)",    psnr_noisy)
    avg_median = summary("After median filter",      psnr_median)
    avg_full   = summary("After median + DnCNN",     psnr_full)

    print()
    print(f"  Median filter gain : +{avg_median - avg_noisy:.2f} dB")
    print(f"  DnCNN extra gain   : +{avg_full - avg_median:.2f} dB")
    print(f"  Total gain         : +{avg_full - avg_noisy:.2f} dB")

    # Gate check
    print()
    if avg_full >= 23.0:
        print(f"  GATE PASSED — full pipeline PSNR {avg_full:.2f} dB >= 23.0 dB")
        print("  Denoiser is ready. Proceed to OCRNet training.")
    else:
        print(f"  GATE FAILED — full pipeline PSNR {avg_full:.2f} dB < 23.0 dB")
        print("  Train DnCNN for more epochs before moving to OCRNet.")

    # ── Side-by-side comparison (first 5 images) ──────────────────────────────
    # We can't show actual images without a display, so we print pixel
    # statistics for a handful of images — mean intensity, std, and a
    # simple ASCII-art row of pixel values to give a visual feel.
    print("\n" + "-" * 60)
    print("  Side-by-side pixel comparison (first 5 val images)")
    print("-" * 60)

    rng2 = np.random.default_rng(seed=0)

    for idx in range(5):
        clean_tensor, label = mnist_val[idx]
        clean_pil = transforms.ToPILImage()(clean_tensor).convert("L")

        arr  = np.array(clean_pil, dtype=np.float32)
        flat = arr.flatten()
        n_pixels  = len(flat)
        n_corrupt = int(n_pixels * 0.05)
        s_idx = rng2.choice(n_pixels, n_corrupt // 2, replace=False)
        p_idx = rng2.choice(n_pixels, n_corrupt // 2, replace=False)
        flat[s_idx] = 255.0
        flat[p_idx] = 0.0
        noisy_pil = Image.fromarray(flat.reshape(28, 28).astype(np.uint8), mode="L")

        noisy_t  = to_tensor(noisy_pil)
        median_t = pipeline.denoise_stage1_only(noisy_pil)
        full_t   = pipeline(noisy_pil)

        def stats(t, name):
            vals = t.squeeze().numpy()
            return f"{name}: mean={vals.mean():.3f} std={vals.std():.3f} min={vals.min():.3f} max={vals.max():.3f}"

        mse_noisy  = float(torch.mean((noisy_t  - clean_tensor) ** 2))
        mse_median = float(torch.mean((median_t - clean_tensor) ** 2))
        mse_full   = float(torch.mean((full_t   - clean_tensor) ** 2))

        print(f"\n  Image {idx+1}  (label={label})")
        print(f"    Noisy          PSNR={mse_to_psnr(mse_noisy):>6.2f} dB  |  {stats(noisy_t,  'pixels')}")
        print(f"    After median   PSNR={mse_to_psnr(mse_median):>6.2f} dB  |  {stats(median_t, 'pixels')}")
        print(f"    After full     PSNR={mse_to_psnr(mse_full):>6.2f} dB  |  {stats(full_t,   'pixels')}")
        print(f"    Clean (truth)                   |  {stats(clean_tensor, 'pixels')}")

        # Simple ASCII row — sample the middle row of the 28x28 image
        # to give a rough visual feel of each version
        row = 14
        def ascii_row(t):
            row_vals = t.squeeze().numpy()[row]
            chars = " .:-=+*#@"
            return "".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)] for v in row_vals)

        print(f"    Row 14 noisy   : |{ascii_row(noisy_t)}|")
        print(f"    Row 14 denoised: |{ascii_row(full_t)}|")
        print(f"    Row 14 clean   : |{ascii_row(clean_tensor)}|")

    return avg_full >= 28.0


# ----------------------------------------------------------------------------
# Entry point — run this file directly after DnCNN training to check
# whether the pipeline is ready to hand off to OCRNet.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify denoiser pipeline PSNR gate")
    parser.add_argument("--weights",  type=str, default="./weights/dncnn.pth",
                        help="Path to trained dncnn.pth")
    parser.add_argument("--data",     type=str, default="./data",
                        help="MNIST data root")
    parser.add_argument("--samples",  type=int, default=200,
                        help="Number of val images to evaluate")
    args = parser.parse_args()

    passed = verify_pipeline(
        weights_path=args.weights,
        data_root=args.data,
        num_samples=args.samples,
    )

    # Exit with code 0 (success) or 1 (gate failed) so CI can catch failures
    sys.exit(0 if passed else 1)
