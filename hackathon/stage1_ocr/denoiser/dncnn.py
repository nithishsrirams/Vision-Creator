import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# A quick note on why we do "residual" denoising instead of direct mapping:
#
# The naive approach would be: train a network to map noisy → clean.
# That works, but it forces the network to learn an identity-like function
# for most pixels (which are already fine), and also learn to fix the noisy
# ones. That's a hard, combined task.
#
# The smarter approach (residual learning): train the network to predict
# ONLY the noise. Then subtract it. The network now has a much simpler job —
# it just needs to find and return what doesn't belong.
#
#     clean = noisy - network(noisy)
#
# This is the core idea from Zhang et al. (2017), "Beyond a Gaussian Denoiser".
# Works remarkably well in practice.
# ----------------------------------------------------------------------------


class DnCNN(nn.Module):
    """
    17-layer residual denoising CNN.

    Input:  (B, 1, H, W)  — noisy grayscale image, pixel values in [0, 1]
    Output: (B, 1, H, W)  — denoised image, same range

    Internally the network predicts the noise map, then we subtract it
    from the input. The subtraction step happens inside forward(), so the
    caller just gets the clean image back directly.
    """

    def __init__(self, num_layers=17, num_channels=64):
        super(DnCNN, self).__init__()

        # We'll build the layers as a plain list first,
        # then wrap in nn.Sequential at the end.
        layers = []

        # ------------------------------------------------------------------
        # Layer 1 — entry conv, no BatchNorm
        #
        # Why no BN here? BatchNorm normalizes across the batch, which is
        # great for middle layers. But on the very first layer, we're working
        # directly on raw pixel values. Normalizing those tends to wash out
        # the subtle intensity differences that indicate noise vs. signal.
        # So we skip BN only on this first layer.
        # ------------------------------------------------------------------
        layers.append(nn.Conv2d(
            in_channels=1,
            out_channels=num_channels,
            kernel_size=3,
            padding=1,      # padding=1 keeps spatial dimensions the same (28x28 stays 28x28)
            bias=False,     # bias=False because BN (on later layers) already handles shift
        ))
        layers.append(nn.ReLU(inplace=True))

        # ------------------------------------------------------------------
        # Layers 2 through 16 — the main body (15 identical blocks)
        #
        # Each block is: Conv → BatchNorm → ReLU
        # The order matters: BN before ReLU is the standard DnCNN recipe.
        # BN normalizes the conv output so ReLU always gets a well-scaled
        # input, which keeps gradients healthy during backprop.
        #
        # We use 64 feature channels throughout. Going wider (128) helps
        # marginally but roughly doubles memory and compute. 64 is the
        # sweet spot for this task.
        #
        # Receptive field: each 3x3 conv adds 2 pixels on each side.
        # After 17 layers: receptive field = 1 + 16*2 = 33 pixels.
        # That means each output pixel "sees" a 33x33 neighborhood of the
        # input — enough context to distinguish noise from real structure.
        # ------------------------------------------------------------------
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ))
            layers.append(nn.BatchNorm2d(num_channels))
            layers.append(nn.ReLU(inplace=True))

        # ------------------------------------------------------------------
        # Layer 17 — exit conv, produces the noise map
        #
        # No BN, no ReLU here. We want raw, unconstrained values because
        # the noise map can have both positive and negative values
        # (some pixels need brightening, others darkening).
        # Adding ReLU would kill the negative corrections.
        # ------------------------------------------------------------------
        layers.append(nn.Conv2d(
            in_channels=num_channels,
            out_channels=1,     # back to single-channel: one noise value per pixel
            kernel_size=3,
            padding=1,
            bias=False,
        ))

        # Wrap everything into a single sequential module.
        # This is cleaner than storing layers individually — forward()
        # just calls self.net(x) and we're done.
        self.net = nn.Sequential(*layers)

        # Initialize weights properly before any training happens
        self._init_weights()


    def _init_weights(self):
        """
        Weight initialization matters more than people think.

        For conv layers: orthogonal initialization keeps the initial
        gradient magnitudes stable across all 17 layers. Random normal
        (the default) can cause gradients to shrink or explode before
        training even gets going.

        For BN layers: start with weight=1 and bias=0 so that initially
        BN is a no-op. The network can then learn to deviate from that
        baseline as needed.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)


    def forward(self, x):
        """
        x: noisy image tensor, shape (B, 1, H, W), values in [0.0, 1.0]

        We pass x through the full 17-layer network to get the noise_map,
        then subtract: clean = noisy - noise_map.

        The clamp at the end just makes sure we don't accidentally produce
        pixel values outside [0, 1] due to floating-point imprecision.
        """
        noise_map = self.net(x)
        clean = x - noise_map
        return torch.clamp(clean, min=0.0, max=1.0)


# ----------------------------------------------------------------------------
# Quick sanity check — run this file directly to confirm the architecture
# is wired up correctly and produces the right output shape.
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running DnCNN shape check...")

    model = DnCNN(num_layers=17, num_channels=64)

    # Count parameters so we know roughly how big this model is
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Simulate a batch of 4 noisy MNIST images (28x28 grayscale)
    dummy_batch = torch.rand(4, 1, 28, 28)   # random values in [0, 1]
    print(f"\n  Input shape  : {tuple(dummy_batch.shape)}")

    model.eval()
    with torch.no_grad():
        output = model(dummy_batch)

    print(f"  Output shape : {tuple(output.shape)}")
    print(f"  Output min   : {output.min().item():.4f}")
    print(f"  Output max   : {output.max().item():.4f}")

    # Confirm output values stay in valid pixel range
    assert output.shape == dummy_batch.shape, "Shape mismatch — something is wrong"
    assert output.min() >= 0.0 and output.max() <= 1.0, "Output out of [0,1] range"

    print("\n  All checks passed.")

    # Print a layer-by-layer summary so you can see the full architecture
    print("\n  Layer breakdown:")
    print(f"  {'Index':<6} {'Type':<20} {'Detail'}")
    print(f"  {'-'*55}")
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Conv2d):
            detail = (f"in={layer.in_channels}, out={layer.out_channels}, "
                      f"k={layer.kernel_size[0]}x{layer.kernel_size[1]}, "
                      f"pad={layer.padding[0]}")
        elif isinstance(layer, nn.BatchNorm2d):
            detail = f"num_features={layer.num_features}"
        elif isinstance(layer, nn.ReLU):
            detail = "inplace=True"
        else:
            detail = ""
        print(f"  {i:<6} {type(layer).__name__:<20} {detail}")
