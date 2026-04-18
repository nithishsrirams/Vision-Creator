"""
metrics.py — Compression quality metrics

Three numbers that tell you how well a compression run went:

  Compression ratio   — how much smaller is the output vs input?
  Shannon entropy     — what's the theoretical best compression possible?
  Encoding efficiency — how close did we get to that theoretical best?
"""

import math
from collections import Counter


def compression_ratio(original_bits: int, compressed_bits: int) -> float:
    """
    Ratio of original size to compressed size.

    A ratio of 2.0 means the compressed output is half the size.
    A ratio of 1.0 means no compression happened (output = input size).
    A ratio < 1.0 means the "compressed" output is actually larger
    (this can happen on short or highly random inputs).

    We return 0.0 for degenerate cases rather than dividing by zero.
    """
    if compressed_bits == 0:
        return 0.0
    return original_bits / compressed_bits


def shannon_entropy(text: str) -> float:
    """
    Compute the Shannon entropy of the text in bits per character.

    Entropy is the theoretical lower bound on average bits-per-symbol
    for any lossless compression scheme given this symbol distribution.
    No algorithm can do better on average (Shannon's source coding theorem).

    H = -sum(p(x) * log2(p(x)))  for all symbols x

    Examples:
      "aaaa"     → entropy = 0.0  (one symbol, totally predictable)
      "aabb"     → entropy = 1.0  (two equally likely symbols)
      "abcd"     → entropy = 2.0  (four equally likely symbols)
      ASCII text → typically 4.0–5.0 bits per character
    """
    if not text:
        return 0.0

    counts = Counter(text)
    total  = len(text)

    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


def encoding_efficiency(text: str, compressed_bits: int) -> float:
    """
    How close did the compressor get to the theoretical Shannon entropy?

    efficiency = (entropy * num_chars) / compressed_bits

    A value of 1.0 means perfect — we hit the entropy lower bound exactly.
    A value of 0.9 means we used 10% more bits than the theoretical minimum.
    A value > 1.0 is not physically possible for a correct compressor.

    Adaptive Huffman typically achieves 0.85–0.97 on natural text.
    Short strings score lower because the tree hasn't had time to adapt.
    """
    if compressed_bits == 0:
        return 0.0

    n       = len(text)
    entropy = shannon_entropy(text)

    # Entropy * n = minimum bits theoretically required for this string
    theoretical_min = entropy * n

    if theoretical_min == 0:
        # Text is perfectly predictable (all one character).
        # The compressor still emits some bits (the NYT + first raw symbol),
        # so efficiency isn't 1.0 here. We return 1.0 as a ceiling since
        # the "information content" is 0.
        return 1.0

    return theoretical_min / compressed_bits


def compute_all(text: str, original_bits: int, compressed_bits: int) -> dict:
    """
    Compute and return all three metrics in one call.
    This is what app.py calls — one dict, all the numbers.
    """
    return {
        "compression_ratio": round(compression_ratio(original_bits, compressed_bits), 4),
        "entropy_bpc":       round(shannon_entropy(text), 4),
        "efficiency":        round(encoding_efficiency(text, compressed_bits), 4),
        "original_bits":     original_bits,
        "compressed_bits":   compressed_bits,
        "original_bytes":    len(text.encode("utf-8")),
        "compressed_bytes":  math.ceil(compressed_bits / 8),
    }


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fgk import encode

    test_cases = [
        ("aaaa",                    "Repetitive (low entropy)"),
        ("abcd",                    "Uniform 4-char (medium entropy)"),
        ("The quick brown fox",     "Natural English"),
        ("hello world",             "Short phrase"),
        ("mississippi",             "Classic Huffman example"),
    ]

    print("Compression metrics")
    print("-" * 70)
    header = f"  {'Text':<25}  {'Ratio':>6}  {'Entropy':>9}  {'Efficiency':>12}"
    print(header)
    print(f"  {'-' * 60}")

    for text, label in test_cases:
        hex_str, orig_bits, comp_bits = encode(text)
        m = compute_all(text, orig_bits, comp_bits)
        print(
            f"  {label:<25}"
            f"  {m['compression_ratio']:>6.3f}x"
            f"  {m['entropy_bpc']:>7.3f} bpc"
            f"  {m['efficiency']:>10.3f}"
        )
