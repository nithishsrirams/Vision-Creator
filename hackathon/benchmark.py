"""
benchmark.py — Send 100 requests to /ocr and /ocr/batch, record latency stats for README

Usage in Colab:
    python benchmark.py
    python benchmark.py --url https://xxxx.ngrok-free.app  # if using ngrok
    python benchmark.py --requests 200 --batch-size 8      # custom settings
"""

import argparse
import io
import os
import statistics
import sys
import time

import requests
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_URL      = "http://localhost:8000"
DEFAULT_REQUESTS = 100
DEFAULT_BATCH    = 8


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_digit_image(digit: int = 7) -> bytes:
    """Creates a synthetic 28x28 digit image and returns PNG bytes."""
    img  = Image.new("L", (28, 28), color=255)
    draw = ImageDraw.Draw(img)

    shapes = {
        0: [[(6,4),(20,4)],[(20,4),(20,24)],[(6,24),(20,24)],[(6,4),(6,24)]],
        1: [[(14,4),(14,24)]],
        2: [[(6,4),(20,4)],[(20,4),(20,14)],[(6,14),(20,14)],[(6,14),(6,24)],[(6,24),(20,24)]],
        7: [[(5,5),(22,5)],[(22,5),(14,22)]],
    }
    for line in shapes.get(digit, shapes[7]):
        draw.line(line, fill=0, width=2)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def percentile(data: list, p: float) -> float:
    data = sorted(data)
    k    = (len(data) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(data) - 1)
    return round(data[f] + (data[c] - data[f]) * (k - f), 2)


def print_stats(label: str, latencies: list, unit: str = "ms"):
    avg    = round(statistics.mean(latencies), 2)
    median = round(statistics.median(latencies), 2)
    p95    = percentile(latencies, 95)
    p99    = percentile(latencies, 99)
    mn     = round(min(latencies), 2)
    mx     = round(max(latencies), 2)
    stdev  = round(statistics.stdev(latencies) if len(latencies) > 1 else 0, 2)

    print(f"\n  {label}")
    print(f"  {'─' * 45}")
    print(f"  Requests   : {len(latencies)}")
    print(f"  Mean       : {avg} {unit}")
    print(f"  Median     : {median} {unit}")
    print(f"  Std dev    : {stdev} {unit}")
    print(f"  P95        : {p95} {unit}")
    print(f"  P99        : {p99} {unit}")
    print(f"  Min        : {mn} {unit}")
    print(f"  Max        : {mx} {unit}")

    return {
        "mean": avg, "median": median, "p95": p95,
        "p99": p99, "min": mn, "max": mx, "stdev": stdev,
    }


# ── Benchmarks ────────────────────────────────────────────────────────────────

def benchmark_single(base_url: str, n: int, image_bytes: bytes) -> dict:
    """Send n sequential POST /ocr requests and collect wall-clock latency."""
    print(f"\n  Running {n} sequential requests to POST /ocr ...")

    server_denoise = []   # denoise_ms reported by server
    server_total   = []   # total_ms reported by server
    client_rtt     = []   # round-trip time measured by client
    errors         = 0

    for i in range(n):
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{base_url}/ocr",
                files={"file": ("sample.png", image_bytes, "image/png")},
                timeout=30,
            )
            rtt = (time.perf_counter() - t0) * 1000

            if r.status_code == 200:
                data = r.json()
                client_rtt.append(round(rtt, 2))
                server_denoise.append(data.get("denoise_ms", 0))
                server_total.append(data.get("total_ms", 0))
            else:
                errors += 1

        except Exception as e:
            errors += 1

        # Progress bar
        done = int((i + 1) / n * 30)
        print(f"    [{'█' * done}{'░' * (30 - done)}] {i+1}/{n}", end="\r")

    print()  # newline after progress bar

    if errors:
        print(f"  ⚠  {errors} request(s) failed")

    stats_rtt     = print_stats("Client round-trip latency (POST /ocr)", client_rtt)
    stats_denoise = print_stats("Server denoise_ms (Median + DnCNN)",    server_denoise)
    stats_total   = print_stats("Server total_ms (end-to-end)",          server_total)

    return {
        "endpoint"      : "POST /ocr",
        "n_requests"    : n,
        "errors"        : errors,
        "client_rtt"    : stats_rtt,
        "server_denoise": stats_denoise,
        "server_total"  : stats_total,
    }


def benchmark_batch(base_url: str, n_batches: int, batch_size: int, image_bytes: bytes) -> dict:
    """Send n_batches requests to POST /ocr/batch (batch_size images each)."""
    total_images = n_batches * batch_size
    print(f"\n  Running {n_batches} batch requests × {batch_size} images = {total_images} total images ...")

    batch_ms_list    = []
    per_image_ms     = []
    errors           = 0

    for i in range(n_batches):
        t0 = time.perf_counter()
        try:
            files = [
                ("files", (f"img_{j}.png", image_bytes, "image/png"))
                for j in range(batch_size)
            ]
            r = requests.post(f"{base_url}/ocr/batch", files=files, timeout=60)
            rtt = (time.perf_counter() - t0) * 1000

            if r.status_code == 200:
                data = r.json()
                batch_ms_list.append(round(data.get("batch_ms", rtt), 2))
                per_image_ms.append(round(data.get("batch_ms", rtt) / batch_size, 2))
            else:
                errors += 1

        except Exception as e:
            errors += 1

        done = int((i + 1) / n_batches * 30)
        print(f"    [{'█' * done}{'░' * (30 - done)}] {i+1}/{n_batches}", end="\r")

    print()

    if errors:
        print(f"  ⚠  {errors} batch request(s) failed")

    stats_batch     = print_stats(f"Batch latency (POST /ocr/batch, {batch_size} imgs each)", batch_ms_list)
    stats_per_image = print_stats("Per-image latency inside batch", per_image_ms)

    return {
        "endpoint"      : "POST /ocr/batch",
        "batch_size"    : batch_size,
        "n_batches"     : n_batches,
        "total_images"  : total_images,
        "errors"        : errors,
        "batch_ms"      : stats_batch,
        "per_image_ms"  : stats_per_image,
    }


# ── README table generator ────────────────────────────────────────────────────

def print_readme_table(single: dict, batch: dict):
    s = single["server_total"]
    d = single["server_denoise"]
    b = batch["per_image_ms"]
    r = single["client_rtt"]

    print("\n" + "=" * 60)
    print("  README LATENCY TABLE — copy-paste this into README.md")
    print("=" * 60)

    table = f"""
## Latency Benchmarks

> Measured on {single['n_requests']} sequential requests  
> Hardware: GPU (CUDA)  
> Image size: 28×28 grayscale PNG

### POST `/ocr` — single image

| Metric        | Value      |
|---------------|------------|
| Mean          | {s['mean']} ms |
| Median        | {s['median']} ms |
| P95           | {s['p95']} ms |
| P99           | {s['p99']} ms |
| Min           | {s['min']} ms |
| Max           | {s['max']} ms |
| Client RTT    | {r['mean']} ms |

### Denoiser breakdown (Median + DnCNN)

| Metric        | Value      |
|---------------|------------|
| Mean          | {d['mean']} ms |
| P95           | {d['p95']} ms |

### POST `/ocr/batch` — {batch['batch_size']} images per request

| Metric              | Value      |
|---------------------|------------|
| Mean per image      | {b['mean']} ms |
| Median per image    | {b['median']} ms |
| P95 per image       | {b['p95']} ms |
"""
    print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "=" * 60)
    print("  OCR Service Benchmark")
    print("=" * 60)
    print(f"  URL        : {args.url}")
    print(f"  Requests   : {args.requests}")
    print(f"  Batch size : {args.batch_size}")

    # Check server is alive first
    try:
        r = requests.get(f"{args.url}/health", timeout=5)
        health = r.json()
        if not health.get("models_ready"):
            print("\n  ✗ Server is not ready — models not loaded.")
            print(f"  Errors: {health.get('errors')}")
            sys.exit(1)
        print(f"  Device     : {health.get('device', 'unknown')}")
        print("  Status     : ✓ Server ready")
    except Exception as e:
        print(f"\n  ✗ Cannot reach server at {args.url}: {e}")
        print("  Make sure the server is running first.")
        sys.exit(1)

    # Generate one test image (reused for all requests)
    image_bytes = make_digit_image(digit=7)
    print(f"  Image      : 28×28 synthetic digit PNG ({len(image_bytes)} bytes)")

    # Run benchmarks
    single_stats = benchmark_single(args.url, args.requests, image_bytes)

    n_batches    = args.requests // args.batch_size
    batch_stats  = benchmark_batch(args.url, n_batches, args.batch_size, image_bytes)

    # Print README table
    print_readme_table(single_stats, batch_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the OCR microservice",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url",        default=DEFAULT_URL,
                        help="Base URL of the server")
    parser.add_argument("--requests",   type=int, default=DEFAULT_REQUESTS,
                        help="Number of single-image requests to send")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH,
                        help="Images per batch request")
    main(parser.parse_args())
