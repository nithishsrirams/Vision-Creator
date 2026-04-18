import torch
import torch.nn as nn


class OCRNet(nn.Module):
    """
    OCRNet — lightweight CNN for denoised MNIST digit classification.

    Architecture
    ------------
    Block 1 : Conv2d(1→32, 3×3, pad=1) → BatchNorm2d(32) → ReLU → MaxPool2d(2×2)
              Output: (B, 32, 14, 14)

    Block 2 : Conv2d(32→64, 3×3, pad=1) → BatchNorm2d(64) → ReLU → MaxPool2d(2×2)
              Output: (B, 64, 7, 7)

    Classifier: Flatten → Linear(64×7×7 → 128) → ReLU → Dropout(0.5) → Linear(128 → 10)

    Notes
    -----
    - No Softmax in forward() — use CrossEntropyLoss during training (handles it internally).
    - Call .predict(x) for inference — returns class index via argmax.
    - Call .predict_proba(x) if you need the full softmax probability vector
      (e.g. for the FastAPI response payload).
    """

    def __init__(self, dropout: float = 0.5, num_classes: int = 10):
        super().__init__()

        # ── Block 1 ──────────────────────────────────────────────────────────
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 28×28 → 14×14
        )

        # ── Block 2 ──────────────────────────────────────────────────────────
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # 14×14 → 7×7
        )

        # ── Classifier head ──────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                             # (B, 64, 7, 7) → (B, 3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),             # logits — no Softmax here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, 28, 28) float tensor, values in [0, 1]
        Returns:
            logits: (B, 10) — raw scores, NOT probabilities
        """
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns predicted class index (0–9) for each image in the batch.

        Args:
            x: (B, 1, 28, 28)
        Returns:
            preds: (B,) int64 tensor
        """
        self.eval()
        logits = self.forward(x)
        return logits.argmax(dim=1)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns softmax probability distribution over 10 classes.
        Use this for the FastAPI response payload.

        Args:
            x: (B, 1, 28, 28)
        Returns:
            probs: (B, 10) float tensor summing to 1 per row
        """
        self.eval()
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


# ----------------------------------------------------------------------------
# Quick sanity check — run this file directly to verify shapes are correct
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    model = OCRNet()
    print(model)
    print()

    dummy = torch.zeros(8, 1, 28, 28)           # batch of 8 blank images
    logits = model(dummy)
    preds  = model.predict(dummy)
    probs  = model.predict_proba(dummy)

    assert logits.shape == (8, 10),  f"Expected (8,10), got {logits.shape}"
    assert preds.shape  == (8,),     f"Expected (8,),   got {preds.shape}"
    assert probs.shape  == (8, 10),  f"Expected (8,10), got {probs.shape}"
    assert torch.allclose(probs.sum(dim=1), torch.ones(8), atol=1e-5), \
        "Probabilities don't sum to 1"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Output shapes  — logits: {logits.shape}  preds: {preds.shape}  probs: {probs.shape}")
    print(f"Total params   — {total_params:,}")
    print("All assertions passed — model.py is ready.")
