import cv2
import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------
# The median filter is the right tool specifically for salt-and-pepper noise.
#
# Salt-and-pepper noise means individual pixels have been randomly set to
# either pure white (salt) or pure black (pepper). They're outliers — their
# values have nothing to do with their neighbors.
#
# The median filter replaces each pixel with the median of its neighborhood.
# Because the corrupted pixels are extreme outliers, they almost never win
# the median vote. The clean surrounding pixels do. Result: spikes gone,
# edges preserved.
#
# Contrast with Gaussian blur, which takes a weighted average. Averages are
# pulled toward outliers — a pure-white salt pixel drags the average up and
# leaves a faint bright smear. The median is immune to that.
#
# Why kernel_size=3?
# A 3×3 kernel looks at the 8 neighbors of each pixel plus the pixel itself.
# That's usually enough to outvote isolated noise spikes. A 5×5 kernel is
# more aggressive but starts to soften fine strokes in digit images, which
# hurts OCR accuracy downstream. 3×3 is the sweet spot for MNIST-style content.
# ----------------------------------------------------------------------------


class MedianFilter:
    """
    Classical median filter for salt-and-pepper denoising.

    Operates directly on PIL Images in uint8 [0, 255] space — before
    any tensor conversion happens. This keeps it fast and lets us chain
    it with DnCNN (which works in float tensor space) in the pipeline.

    Args:
        kernel_size: Side length of the square filter window. Must be odd.
                     3 works well for MNIST. Use 5 for heavier noise.
    """

    def __init__(self, kernel_size: int = 3):
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd, got {kernel_size}. "
                "Even kernels don't have a well-defined center pixel."
            )
        if kernel_size < 3:
            raise ValueError(
                f"kernel_size must be at least 3, got {kernel_size}."
            )
        self.kernel_size = kernel_size

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply median filter to a PIL Image.

        Args:
            img: Input PIL Image. Can be any mode (L, RGB, etc.).
                 Internally converted to uint8 numpy for OpenCV.
        Returns:
            Denoised PIL Image in the same mode as input.
        """
        original_mode = img.mode

        # Convert to numpy uint8 — this is what cv2 expects
        arr = np.array(img, dtype=np.uint8)

        # cv2.medianBlur handles both grayscale (H, W) and
        # color (H, W, C) arrays automatically
        denoised_arr = cv2.medianBlur(arr, self.kernel_size)

        # Convert back to PIL in the original mode
        result = Image.fromarray(denoised_arr)
        if result.mode != original_mode:
            result = result.convert(original_mode)

        return result

    def __repr__(self):
        return f"MedianFilter(kernel_size={self.kernel_size})"


# ----------------------------------------------------------------------------
# Quick test — run this file directly to confirm the filter is working.
# Creates a synthetic noisy image, applies the filter, and checks that
# the corrupted pixels were actually fixed.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from torchvision import transforms

    print("MedianFilter sanity check")
    print("-" * 40)

    # Create a clean synthetic grayscale image — a simple gradient
    width, height = 28, 28
    gradient = np.linspace(0, 255, width * height, dtype=np.uint8).reshape(height, width)
    clean_img = Image.fromarray(gradient, mode="L")

    # Manually inject salt-and-pepper noise at 10% rate
    noisy_arr = gradient.copy().astype(np.float32)
    rng = np.random.default_rng(seed=42)

    num_pixels = width * height
    num_salt   = int(num_pixels * 0.05)   # 5% salt
    num_pepper = int(num_pixels * 0.05)   # 5% pepper

    salt_idx   = rng.choice(num_pixels, num_salt,   replace=False)
    pepper_idx = rng.choice(num_pixels, num_pepper, replace=False)

    flat = noisy_arr.flatten()
    flat[salt_idx]   = 255.0
    flat[pepper_idx] = 0.0
    noisy_arr = flat.reshape(height, width).astype(np.uint8)
    noisy_img = Image.fromarray(noisy_arr, mode="L")

    # Count corrupted pixels before filtering
    corrupted_before = int(np.sum(np.abs(noisy_arr.astype(int) - gradient.astype(int)) > 50))
    print(f"  Corrupted pixels before : {corrupted_before}")

    # Apply filter
    filt = MedianFilter(kernel_size=3)
    denoised_img = filt(noisy_img)
    denoised_arr = np.array(denoised_img)

    # Count remaining corrupted pixels after filtering
    corrupted_after = int(np.sum(np.abs(denoised_arr.astype(int) - gradient.astype(int)) > 50))
    print(f"  Corrupted pixels after  : {corrupted_after}")

    recovery_rate = (1 - corrupted_after / max(corrupted_before, 1)) * 100
    print(f"  Recovery rate           : {recovery_rate:.1f}%")

    # Confirm PIL round-trip works
    assert isinstance(denoised_img, Image.Image), "Output is not a PIL Image"
    assert denoised_img.mode == "L", f"Mode changed: expected L, got {denoised_img.mode}"
    assert denoised_img.size == clean_img.size, "Image size changed after filtering"

    # Confirm repr works
    print(f"  repr                    : {filt}")

    print("\n  All checks passed.")
