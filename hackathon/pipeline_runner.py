"""
pipeline_runner.py
End-to-end orchestrator for the 2-stage neural compression pipeline.

What this script does:
    1. Takes a noisy scanned digit image as input
    2. Sends it to Stage 1 (OCR microservice, port 8000) → gets back predicted text
    3. Sends that text to Stage 2 (Huffman microservice, port 8001) → compressed bitstream
    4. Sends the bitstream back to Stage 2 for decompression → recovered text
    5. Verifies the recovered text matches the original OCR output (lossless check)
    6. Prints a full report with timings, compression metrics, and pass/fail status

Modes:
    Single image run   — process one image, print full report
    Batch run          — process a folder of images, print summary table
    Benchmark mode     — hammer one image N times, print latency percentiles
    Demo mode          — formatted output designed for screen recording

Both services must be running before calling this script:
    uvicorn stage1_ocr.app:app         --host 0.0.0.0 --port 8000
    uvicorn stage2_huffman.app:app     --host 0.0.0.0 --port 8001

Usage:
    python pipeline_runner.py --image path/to/digit.png
    python pipeline_runner.py --image path/to/digit.png --demo
    python pipeline_runner.py --batch  path/to/image_folder/
    python pipeline_runner.py --image path/to/digit.png --benchmark --n 100
    python pipeline_runner.py --image path/to/digit.png --save results.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Service URLs — change these if running on different hosts/ports
# ---------------------------------------------------------------------------

STAGE1_URL = os.environ.get("STAGE1_URL", "http://localhost:8000")
STAGE2_URL = os.environ.get("STAGE2_URL", "http://localhost:8001")

# Supported image extensions for batch mode
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Terminal colors — makes the output readable at a glance
# Used lightly so the script still works cleanly when piped to a file
# ---------------------------------------------------------------------------

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def green(s):  return f"{GREEN}{s}{RESET}"
def red(s):    return f"{RED}{s}{RESET}"
def yellow(s): return f"{YELLOW}{s}{RESET}"
def cyan(s):   return f"{CYAN}{s}{RESET}"
def bold(s):   return f"{BOLD}{s}{RESET}"
def dim(s):    return f"{DIM}{s}{RESET}"


# ---------------------------------------------------------------------------
# Service health checks
#
# We check both services before doing anything. There's nothing worse than
# sitting through a benchmark run and discovering at step 3/5 that the
# compression service wasn't started.
# ---------------------------------------------------------------------------

def check_services(quiet: bool = False) -> bool:
    """
    Ping both services and confirm they respond with status: ok.

    Returns True if both are healthy, False otherwise.
    Prints a clear status line for each service.
    """
    if not quiet:
        print(f"\n{bold('Service health check')}")
        print(f"  {dim('─' * 50)}")

    all_healthy = True

    for name, url in [("Stage 1 — OCR (port 8000)", STAGE1_URL),
                      ("Stage 2 — Huffman (port 8001)", STAGE2_URL)]:
        try:
            resp = requests.get(f"{url}/health", timeout=4)
            resp.raise_for_status()
            data = resp.json()

            if not quiet:
                print(f"  {green('OK')}  {name}")
                extras = {k: v for k, v in data.items() if k != "status"}
                if extras:
                    details = "  ".join(f"{k}={v}" for k, v in extras.items())
                    print(f"      {dim(details)}")

        except requests.exceptions.ConnectionError:
            if not quiet:
                print(f"  {red('FAIL')}  {name}")
                print(f"       {red('Connection refused')} — is the service running?")
                print(f"       Start it with: {cyan(f'uvicorn <module>:app --port <port>')}")
            all_healthy = False
        except Exception as e:
            if not quiet:
                print(f"  {red('FAIL')}  {name} — {e}")
            all_healthy = False

    if not all_healthy and not quiet:
        print(f"\n  {red('Cannot proceed — fix the above before running.')}")

    return all_healthy


# ---------------------------------------------------------------------------
# Core pipeline — one image through all stages
#
# This is the heart of the script. Everything else (batch, benchmark, demo)
# calls this function. It returns a structured result dict so callers can
# do whatever they want with the data.
# ---------------------------------------------------------------------------

def run_pipeline(image_path: str) -> dict:
    """
    Run one image through the full pipeline and return a result dict.

    The result dict contains:
        ocr_text        — text returned by Stage 1
        recovered_text  — text recovered after compress + decompress
        lossless        — bool: recovered_text == ocr_text
        stage1          — full Stage 1 response (confidence, denoise_ms, etc.)
        compress        — full Stage 2 /compress response
        decompress      — full Stage 2 /decompress response
        timings         — dict of timing measurements in milliseconds
        error           — None on success, error message string on failure
    """
    result = {
        "image_path":     image_path,
        "ocr_text":       None,
        "recovered_text": None,
        "lossless":       False,
        "stage1":         None,
        "compress":       None,
        "decompress":     None,
        "timings":        {},
        "error":          None,
    }

    wall_start = time.perf_counter()

    # ── Stage 1: OCR ─────────────────────────────────────────────────────────
    try:
        t0 = time.perf_counter()

        image_name = os.path.basename(image_path)
        ext = Path(image_path).suffix.lower()
        content_type = {
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".bmp":  "image/bmp",
            ".tiff": "image/tiff",
            ".tif":  "image/tiff",
        }.get(ext, "image/png")

        with open(image_path, "rb") as f:
            resp1 = requests.post(
                f"{STAGE1_URL}/ocr",
                files={"file": (image_name, f, content_type)},
                timeout=30,
            )

        resp1.raise_for_status()
        stage1_data = resp1.json()
        result["timings"]["stage1_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        result["stage1"]   = stage1_data
        result["ocr_text"] = stage1_data["text"]

    except requests.exceptions.Timeout:
        result["error"] = "Stage 1 timed out (>30s) — model may still be loading"
        return result
    except requests.exceptions.HTTPError:
        result["error"] = f"Stage 1 HTTP error {resp1.status_code}: {resp1.text[:200]}"
        return result
    except FileNotFoundError:
        result["error"] = f"Image not found: {image_path}"
        return result
    except Exception as e:
        result["error"] = f"Stage 1 unexpected error: {e}"
        return result

    # ── Stage 2a: Compress ───────────────────────────────────────────────────
    try:
        t1 = time.perf_counter()

        resp2 = requests.post(
            f"{STAGE2_URL}/compress",
            json={"text": result["ocr_text"]},
            timeout=10,
        )
        resp2.raise_for_status()
        compress_data = resp2.json()
        result["timings"]["compress_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        result["compress"] = compress_data

    except requests.exceptions.HTTPError:
        result["error"] = f"Stage 2 /compress HTTP error {resp2.status_code}: {resp2.text[:200]}"
        return result
    except Exception as e:
        result["error"] = f"Stage 2 /compress unexpected error: {e}"
        return result

    # ── Stage 2b: Decompress ─────────────────────────────────────────────────
    # compressed_bits is required — it tells the decoder where padding ends.
    # This is the field the old skeleton was missing, which would have caused
    # the decompressor to read phantom symbols from the padding bits.
    try:
        t2 = time.perf_counter()

        resp3 = requests.post(
            f"{STAGE2_URL}/decompress",
            json={
                "compressed_hex":  compress_data["compressed_hex"],
                "compressed_bits": compress_data["compressed_bits"],
            },
            timeout=10,
        )
        resp3.raise_for_status()
        decompress_data = resp3.json()
        result["timings"]["decompress_ms"]  = round((time.perf_counter() - t2) * 1000, 2)
        result["decompress"]                = decompress_data
        result["recovered_text"]            = decompress_data["text"]

    except requests.exceptions.HTTPError:
        result["error"] = f"Stage 2 /decompress HTTP error {resp3.status_code}: {resp3.text[:200]}"
        return result
    except Exception as e:
        result["error"] = f"Stage 2 /decompress unexpected error: {e}"
        return result

    # ── Lossless verification ─────────────────────────────────────────────────
    result["lossless"] = (result["recovered_text"] == result["ocr_text"])

    result["timings"]["end_to_end_ms"] = round(
        (time.perf_counter() - wall_start) * 1000, 2
    )

    return result


# ---------------------------------------------------------------------------
# Print functions — one for each output mode
# ---------------------------------------------------------------------------

def print_single_result(result: dict):
    """
    Full verbose output for a single image run.
    Every field clearly labeled, pass/fail highlighted in color.
    """
    print()

    if result["error"]:
        print(f"  {red('ERROR')}: {result['error']}")
        return

    img_name = os.path.basename(result["image_path"])
    s1 = result["stage1"]
    c  = result["compress"]
    t  = result["timings"]

    print(f"  {bold('Image')}              {img_name}")
    print()
    print(f"  {bold('Stage 1 — OCR')}")
    print(f"    predicted text   : {cyan(repr(result['ocr_text']))}")
    print(f"    confidence       : {s1['confidence']:.4f}")
    print(f"    denoise_ms       : {s1.get('denoise_ms', 'n/a')}")
    print(f"    stage1_total_ms  : {t['stage1_ms']} ms")
    print()
    print(f"  {bold('Stage 2 — Compression')}")
    print(f"    original_bytes   : {c['original_bytes']}")
    print(f"    compressed_bytes : {c['compressed_bytes']}")
    print(f"    compression_ratio: {c['compression_ratio']:.4f}x")
    print(f"    entropy_bpc      : {c['entropy_bpc']:.4f} bits/char")
    print(f"    efficiency       : {c['efficiency']:.4f}")
    print(f"    compress_ms      : {t['compress_ms']} ms")
    print()
    print(f"  {bold('Stage 2 — Decompression')}")
    print(f"    recovered text   : {cyan(repr(result['recovered_text']))}")
    print(f"    decompress_ms    : {t['decompress_ms']} ms")
    print()

    # Lossless check — the most important line in the whole output
    if result["lossless"]:
        verdict = green("PASS")
        detail  = (f"{cyan(repr(result['ocr_text']))} "
                   f"== {cyan(repr(result['recovered_text']))}")
    else:
        verdict = red("FAIL")
        detail  = (f"expected {cyan(repr(result['ocr_text']))} "
                   f"got {red(repr(result['recovered_text']))}")

    print(f"  {bold('Lossless check')}     [{verdict}]  {detail}")
    print()
    print(f"  {bold('End-to-end latency')} {t['end_to_end_ms']} ms")


def print_demo_result(result: dict):
    """
    Clean, large-format output designed for a demo screen recording.
    Uses box-drawing characters for a polished look.
    """
    w = 56

    def box(lines):
        print(f"  ┌{'─' * w}┐")
        for line in lines:
            # Strip ANSI codes for length calculation
            import re
            clean = re.sub(r'\033\[[0-9;]*m', '', line)
            padding = max(0, w - len(clean) - 1)
            print(f"  │ {line}{' ' * padding}│")
        print(f"  └{'─' * w}┘")

    print()
    print(f"  {bold('2-Stage Neural Compression Pipeline')}")
    print(f"  {'─' * w}")

    if result["error"]:
        box([f"ERROR: {result['error']}"])
        return

    s1 = result["stage1"]
    c  = result["compress"]
    t  = result["timings"]

    print()
    print(f"  {bold('STEP 1')}  Noisy image → OCR microservice  ({t['stage1_ms']} ms)")
    box([
        f"  Input     : {os.path.basename(result['image_path'])}",
        f"  Denoiser  : Median filter + DnCNN (17-layer residual)",
        f"  Result    : {cyan(repr(result['ocr_text']))}",
        f"  Confidence: {s1['confidence']:.1%}",
    ])

    print()
    print(f"  {bold('STEP 2')}  OCR text → Huffman compression  ({t['compress_ms']} ms)")
    box([
        f"  Input     : {repr(result['ocr_text'])}",
        f"  Original  : {c['original_bytes']} bytes  ({c['original_bits']} bits)",
        f"  Compressed: {c['compressed_bytes']} bytes  ({c['compressed_bits']} bits)",
        f"  Ratio     : {c['compression_ratio']:.3f}x",
        f"  Entropy   : {c['entropy_bpc']:.3f} bits/char",
        f"  Efficiency: {c['efficiency']:.1%} of Shannon limit",
    ])

    print()
    print(f"  {bold('STEP 3')}  Bitstream → decompress → verify  ({t['decompress_ms']} ms)")
    lossless_str = green("LOSSLESS  ✓") if result["lossless"] else red("FAILED  ✗")
    box([
        f"  Recovered : {cyan(repr(result['recovered_text']))}",
        f"  Verified  : {lossless_str}",
    ])

    print()
    print(f"  {'─' * w}")
    status = green("ALL STAGES PASSED") if result["lossless"] else red("LOSSLESS CHECK FAILED")
    print(f"  {bold('Result')}  {status}")
    print(f"  {bold('Total')}   {t['end_to_end_ms']} ms end-to-end")
    print()


# ---------------------------------------------------------------------------
# Batch mode — process an entire folder
# ---------------------------------------------------------------------------

def run_batch(folder: str, save_path: str = None):
    """
    Run every image in a folder through the pipeline and print a summary table.
    Good for validating the pipeline on a whole test set at once.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        print(f"\n  {red('ERROR')}: {folder} is not a directory")
        return

    image_files = sorted([
        p for p in folder_path.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if not image_files:
        print(f"\n  {yellow('WARNING')}: No images found in {folder}")
        return

    print(f"\n{bold('Batch run')}  {len(image_files)} images in {folder}")
    print(f"  {'─' * 68}")
    print(f"  {'Image':<26}  {'OCR':>5}  {'Ratio':>7}  {'ms':>7}  {'Lossless':>10}")
    print(f"  {'─' * 68}")

    all_results = []
    passed = failed = errors = 0

    for img_path in image_files:
        result = run_pipeline(str(img_path))
        all_results.append(result)

        name = img_path.name[:24]

        if result["error"]:
            errors += 1
            print(f"  {name:<26}  {red('ERROR — ' + result['error'][:35])}")
            continue

        ocr_str  = repr(result["ocr_text"])
        ratio    = result["compress"]["compression_ratio"]
        e2e      = result["timings"]["end_to_end_ms"]
        verdict  = green("PASS") if result["lossless"] else red("FAIL")

        if result["lossless"]:
            passed += 1
        else:
            failed += 1

        print(
            f"  {name:<26}"
            f"  {ocr_str:>5}"
            f"  {ratio:>7.3f}x"
            f"  {e2e:>7.1f}"
            f"  {verdict:>10}"
        )

    print(f"  {'─' * 68}")
    print(f"\n  {green(str(passed))} passed  {red(str(failed))} failed  "
          f"{yellow(str(errors))} errors  / {len(image_files)} total")

    if save_path:
        with open(save_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved to {save_path}")


# ---------------------------------------------------------------------------
# Benchmark mode — latency percentiles over N runs
# ---------------------------------------------------------------------------

def run_benchmark(image_path: str, n: int = 50, save_path: str = None):
    """
    Run the same image N times (default 50) and compute p50/p95/p99 latency
    for each stage.  These numbers go directly into the README benchmark table.
    """
    print(f"\n{bold('Benchmark mode')}  {os.path.basename(image_path)}  x{n} runs")
    print(f"  {'─' * 50}")

    stage1_times     = []
    compress_times   = []
    decompress_times = []
    e2e_times        = []
    lossless_fails   = 0
    errors           = 0

    for i in range(n):
        result = run_pipeline(image_path)

        if result["error"]:
            errors += 1
            if errors <= 3:
                print(f"  Run {i+1:>3}: {red('ERROR')} — {result['error']}")
            continue

        t = result["timings"]
        stage1_times.append(t["stage1_ms"])
        compress_times.append(t["compress_ms"])
        decompress_times.append(t["decompress_ms"])
        e2e_times.append(t["end_to_end_ms"])

        if not result["lossless"]:
            lossless_fails += 1

        verdict_sym = green("✓") if result["lossless"] else red("✗")
        print(f"  [{i+1:>3}/{n}] {verdict_sym}  e2e={t['end_to_end_ms']:>7.1f} ms"
              f"  stage1={t['stage1_ms']:>6.1f}  compress={t['compress_ms']:>6.1f}"
              f"  decompress={t['decompress_ms']:>6.1f}")

    def pct(data, p):
        """Nearest-rank percentile (matches numpy percentile with method='lower')."""
        if not data:
            return 0.0
        s = sorted(data)
        # linear interpolation for a smoother estimate
        k = (p / 100) * (len(s) - 1)
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (k - lo) * (s[hi] - s[lo])

    def row(label, times):
        if not times:
            return
        print(
            f"  {label:<32}"
            f"  {pct(times, 50):>8.2f}"
            f"  {pct(times, 95):>8.2f}"
            f"  {pct(times, 99):>8.2f}"
            f"  {statistics.mean(times):>9.2f}"
        )

    success = len(e2e_times)
    print(f"\n  Runs: {success}/{n} ok  |  errors: {errors}  |  lossless_fails: {lossless_fails}")
    print()
    print(f"  {'Stage':<32}  {'p50 ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}  {'mean ms':>9}")
    print(f"  {'─' * 70}")
    row("Stage 1  (OCR + denoiser)",  stage1_times)
    row("Stage 2  /compress",         compress_times)
    row("Stage 2  /decompress",       decompress_times)
    print(f"  {'─' * 70}")
    row("End-to-end total",           e2e_times)
    print()

    verdict = green("PASSED on all runs") if lossless_fails == 0 else red(f"{lossless_fails} failures")
    print(f"  Lossless check: {verdict}")

    summary = {
        "image_path": image_path,
        "total_runs": n,
        "successful": success,
        "errors":     errors,
        "lossless_fails": lossless_fails,
        "latency_ms": {
            "stage1": {
                "p50":  pct(stage1_times, 50),
                "p95":  pct(stage1_times, 95),
                "p99":  pct(stage1_times, 99),
                "mean": statistics.mean(stage1_times) if stage1_times else 0,
            },
            "compress": {
                "p50":  pct(compress_times, 50),
                "p95":  pct(compress_times, 95),
                "p99":  pct(compress_times, 99),
                "mean": statistics.mean(compress_times) if compress_times else 0,
            },
            "decompress": {
                "p50":  pct(decompress_times, 50),
                "p95":  pct(decompress_times, 95),
                "p99":  pct(decompress_times, 99),
                "mean": statistics.mean(decompress_times) if decompress_times else 0,
            },
            "end_to_end": {
                "p50":  pct(e2e_times, 50),
                "p95":  pct(e2e_times, 95),
                "p99":  pct(e2e_times, 99),
                "mean": statistics.mean(e2e_times) if e2e_times else 0,
            },
        },
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved to {save_path}")

    return summary


# ---------------------------------------------------------------------------
# Benchmark-50 mode — time 50 *distinct* images through the full pipeline
#
# run_benchmark() hammers the same image N times (good for isolating model
# variance).  run_benchmark_50() instead walks through a folder of images,
# picks exactly 50 (cycling if the folder is smaller), and runs each once.
# This gives end-to-end latency across realistic input variety.
# ---------------------------------------------------------------------------

def run_benchmark_50(image_source: str, save_path: str = None):
    """
    Time exactly 50 images through the full pipeline and report p50/p95/p99.

    image_source can be:
      • a folder  — all images in it are used (cycled if < 50)
      • a single image file — that file is used 50 times (same as --benchmark --n 50)
    """
    source = Path(image_source)

    if source.is_dir():
        candidates = sorted([
            str(p) for p in source.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])
        if not candidates:
            print(f"\n  {red('ERROR')}: No images found in {image_source}")
            return None
        # cycle through candidates until we have exactly 50
        image_list = [candidates[i % len(candidates)] for i in range(50)]
        label = f"50 images from {source.name}/"
    elif source.is_file():
        image_list = [str(source)] * 50
        label = f"{source.name}  ×50"
    else:
        print(f"\n  {red('ERROR')}: {image_source} is not a file or directory")
        return None

    n = len(image_list)   # always 50
    print(f"\n{bold('Benchmark-50')}  {label}")
    print(f"  {dim('Timing each image once through the full pipeline — OCR → compress → decompress → verify')}")
    print(f"  {'─' * 72}")
    print(f"  {'#':>3}  {'Image':<28}  {'e2e ms':>8}  {'s1 ms':>7}  "
          f"{'cmp ms':>7}  {'dcmp ms':>8}  {'OK':>4}")
    print(f"  {'─' * 72}")

    stage1_times     = []
    compress_times   = []
    decompress_times = []
    e2e_times        = []
    lossless_fails   = 0
    errors           = 0
    per_image        = []   # for JSON export

    for i, img_path in enumerate(image_list):
        result = run_pipeline(img_path)
        name   = Path(img_path).name[:26]

        if result["error"]:
            errors += 1
            per_image.append({"index": i, "image": img_path, "error": result["error"]})
            print(f"  {i+1:>3}  {name:<28}  {red('ERROR: ' + result['error'][:30])}")
            continue

        t  = result["timings"]
        ok = green("✓") if result["lossless"] else red("✗")

        stage1_times.append(t["stage1_ms"])
        compress_times.append(t["compress_ms"])
        decompress_times.append(t["decompress_ms"])
        e2e_times.append(t["end_to_end_ms"])

        if not result["lossless"]:
            lossless_fails += 1

        per_image.append({
            "index":        i,
            "image":        img_path,
            "stage1_ms":    t["stage1_ms"],
            "compress_ms":  t["compress_ms"],
            "decompress_ms":t["decompress_ms"],
            "e2e_ms":       t["end_to_end_ms"],
            "lossless":     result["lossless"],
        })

        print(f"  {i+1:>3}  {name:<28}  {t['end_to_end_ms']:>8.1f}"
              f"  {t['stage1_ms']:>7.1f}  {t['compress_ms']:>7.1f}"
              f"  {t['decompress_ms']:>8.1f}  {ok:>4}")

    # ── Percentile computation ─────────────────────────────────────────────
    def pct(data, p):
        if not data:
            return 0.0
        s = sorted(data)
        k = (p / 100) * (len(s) - 1)
        lo, hi = int(k), min(int(k) + 1, len(s) - 1)
        return s[lo] + (k - lo) * (s[hi] - s[lo])

    def row(label, times):
        if not times:
            return
        print(
            f"  {label:<32}"
            f"  {pct(times, 50):>8.2f}"
            f"  {pct(times, 95):>8.2f}"
            f"  {pct(times, 99):>8.2f}"
            f"  {statistics.mean(times):>9.2f}"
        )

    success = len(e2e_times)
    print(f"\n  {'─' * 72}")
    print(f"  Images: {success}/{n} ok  |  errors: {errors}  |  lossless_fails: {lossless_fails}")
    print()
    print(f"  {'Stage':<32}  {'p50 ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}  {'mean ms':>9}")
    print(f"  {'─' * 70}")
    row("Stage 1  (OCR + denoiser)",  stage1_times)
    row("Stage 2  /compress",         compress_times)
    row("Stage 2  /decompress",       decompress_times)
    print(f"  {'─' * 70}")
    row("End-to-end total",           e2e_times)
    print()

    lossless_verdict = (green("ALL LOSSLESS ✓") if lossless_fails == 0
                        else red(f"{lossless_fails} lossless failures ✗"))
    print(f"  Lossless check: {lossless_verdict}")

    summary = {
        "mode":          "benchmark_50",
        "image_source":  image_source,
        "total_images":  n,
        "successful":    success,
        "errors":        errors,
        "lossless_fails":lossless_fails,
        "latency_ms": {
            "stage1": {
                "p50":  pct(stage1_times, 50),
                "p95":  pct(stage1_times, 95),
                "p99":  pct(stage1_times, 99),
                "mean": statistics.mean(stage1_times) if stage1_times else 0,
            },
            "compress": {
                "p50":  pct(compress_times, 50),
                "p95":  pct(compress_times, 95),
                "p99":  pct(compress_times, 99),
                "mean": statistics.mean(compress_times) if compress_times else 0,
            },
            "decompress": {
                "p50":  pct(decompress_times, 50),
                "p95":  pct(decompress_times, 95),
                "p99":  pct(decompress_times, 99),
                "mean": statistics.mean(decompress_times) if decompress_times else 0,
            },
            "end_to_end": {
                "p50":  pct(e2e_times, 50),
                "p95":  pct(e2e_times, 95),
                "p99":  pct(e2e_times, 99),
                "mean": statistics.mean(e2e_times) if e2e_times else 0,
            },
        },
        "per_image": per_image,
    }

    if save_path:
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved to {save_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Neural compression pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline_runner.py --image samples/digit_7.png
  python pipeline_runner.py --image samples/digit_7.png --demo
  python pipeline_runner.py --batch  samples/
  python pipeline_runner.py --image samples/digit_7.png --benchmark --n 50
  python pipeline_runner.py --image samples/digit_7.png --benchmark50
  python pipeline_runner.py --batch  samples/ --benchmark50
  python pipeline_runner.py --image samples/digit_7.png --save result.json
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, metavar="PATH",
                       help="Path to a single input image")
    group.add_argument("--batch", type=str, metavar="FOLDER",
                       help="Path to a folder — runs all images in it")

    parser.add_argument("--benchmark",   action="store_true",
                        help="Latency benchmark mode (requires --image)")
    parser.add_argument("--benchmark50", action="store_true",
                        help="Time exactly 50 images through the full pipeline and "
                             "report p50/p95/p99.  Pass --image for a single file "
                             "(repeated 50×) or --batch for a folder of images.")
    parser.add_argument("--demo",        action="store_true",
                        help="Demo-formatted output for screen recording")
    parser.add_argument("--n",           type=int, default=50, metavar="N",
                        help="Benchmark iterations (default: 50)")
    parser.add_argument("--save",        type=str, default=None, metavar="PATH",
                        help="Save JSON results to this path")
    parser.add_argument("--stage1_url",  type=str, default=None,
                        help="Override Stage 1 URL (default: http://localhost:8000)")
    parser.add_argument("--stage2_url",  type=str, default=None,
                        help="Override Stage 2 URL (default: http://localhost:8001)")

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()

    global STAGE1_URL, STAGE2_URL
    if args.stage1_url:
        STAGE1_URL = args.stage1_url
    if args.stage2_url:
        STAGE2_URL = args.stage2_url

    if not check_services():
        sys.exit(1)

    if args.batch:
        if args.benchmark50:
            run_benchmark_50(args.batch, save_path=args.save)
            return
        run_batch(args.batch, save_path=args.save)
        return

    if not os.path.exists(args.image):
        print(f"\n  {red('ERROR')}: Image not found: {args.image}")
        sys.exit(1)

    if args.benchmark50:
        run_benchmark_50(args.image, save_path=args.save)
        return

    if args.benchmark:
        run_benchmark(args.image, n=args.n, save_path=args.save)
        return

    # Single image run
    print(f"\n{bold('Running pipeline...')}")
    result = run_pipeline(args.image)

    if args.demo:
        print_demo_result(result)
    else:
        print_single_result(result)

    if args.save:
        with open(args.save, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n  Result saved to {args.save}")

    # Exit code reflects pipeline success — useful in CI / shell scripts
    sys.exit(0 if (not result["error"] and result["lossless"]) else 1)


if __name__ == "__main__":
    main()
