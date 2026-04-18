"""
app.py — FastAPI OCR microservice (Stage 1)
Weights: /content/drive/MyDrive/hackathon/stage1_ocr/weights/
"""

import io
import os
import sys
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms

# ── Exact weight paths ────────────────────────────────────────────────────────
DNCNN_WEIGHTS = "/content/drive/MyDrive/hackathon/stage1_ocr/weights/dncnn.pth"
OCR_WEIGHTS   = "/content/drive/MyDrive/hackathon/stage1_ocr/weights/ocr_snp.pth"

# ── sys.path so imports work ──────────────────────────────────────────────────
_HACKATHON = "/content/drive/MyDrive/hackathon"
_STAGE1    = "/content/drive/MyDrive/hackathon/stage1_ocr"
_DENOISER  = "/content/drive/MyDrive/hackathon/stage1_ocr/denoiser"

for _p in [_HACKATHON, _STAGE1, _DENOISER]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from stage1_ocr.denoiser.model    import OCRNet
from stage1_ocr.denoiser.pipeline import MedianDnCNNPipeline

# ── Normalization constants ───────────────────────────────────────────────────
MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "OCR Microservice — Stage 1",
    description = "POST an image → get the predicted digit back",
    version     = "1.0.0",
)

# ── Global model state ────────────────────────────────────────────────────────
pipeline     = None
ocr_model    = None
device       = None
normalize    = None
startup_time = None
load_errors  = []


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def load_models():
    global pipeline, ocr_model, device, normalize, startup_time, load_errors

    t0          = time.time()
    load_errors = []

    # Best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"\n[startup] device      : {device}")
    print(f"[startup] dncnn path  : {DNCNN_WEIGHTS}  exists={os.path.exists(DNCNN_WEIGHTS)}")
    print(f"[startup] ocr path    : {OCR_WEIGHTS}  exists={os.path.exists(OCR_WEIGHTS)}")

    # Load DnCNN
    if os.path.exists(DNCNN_WEIGHTS):
        try:
            pipeline = MedianDnCNNPipeline(weights_path=DNCNN_WEIGHTS, device=str(device))
            print("[startup] ✓ DnCNN loaded")
        except Exception as e:
            load_errors.append(f"DnCNN load failed: {e}")
            print(f"[startup] ✗ DnCNN load failed: {e}")
    else:
        load_errors.append(f"dncnn.pth not found at: {DNCNN_WEIGHTS}")
        print(f"[startup] ✗ dncnn.pth not found at: {DNCNN_WEIGHTS}")

    # Load OCRNet
    if os.path.exists(OCR_WEIGHTS):
        try:
            ckpt      = torch.load(OCR_WEIGHTS, map_location=device)
            ocr_model = OCRNet(dropout=0.0, num_classes=10).to(device)
            ocr_model.load_state_dict(ckpt["model_state"])
            ocr_model.eval()
            val_acc = float(ckpt.get("val_acc", 0)) * 100
            print(f"[startup] ✓ OCRNet loaded (val_acc={val_acc:.2f}%)")
        except Exception as e:
            load_errors.append(f"OCRNet load failed: {e}")
            print(f"[startup] ✗ OCRNet load failed: {e}")
    else:
        load_errors.append(f"ocr_snp.pth not found at: {OCR_WEIGHTS}")
        print(f"[startup] ✗ ocr_snp.pth not found at: {OCR_WEIGHTS}")

    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(MNIST_MEAN,), std=(MNIST_STD,)),
    ])

    startup_time = round(time.time() - t0, 2)

    if not load_errors:
        print(f"[startup] ✓ Fully ready in {startup_time}s\n")
    else:
        print(f"[startup] ⚠  {len(load_errors)} model(s) missing — see /health\n")


# =============================================================================
# GET /health
# =============================================================================

@app.get("/health")
async def health():
    dncnn_ok = pipeline  is not None
    ocr_ok   = ocr_model is not None
    all_ok   = dncnn_ok and ocr_ok

    return JSONResponse(
        status_code=200 if all_ok else 206,
        content={
            "status"         : "ok" if all_ok else "partial",
            "device"         : str(device),
            "dncnn_loaded"   : dncnn_ok,
            "ocr_loaded"     : ocr_ok,
            "models_ready"   : all_ok,
            "startup_seconds": startup_time,
            "weights"        : {"dncnn": DNCNN_WEIGHTS, "ocr": OCR_WEIGHTS},
            "errors"         : load_errors,
        }
    )


# =============================================================================
# POST /ocr
# =============================================================================

@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    if pipeline is None or ocr_model is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "Models not loaded", "missing": load_errors}
        )

    if file.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/bmp"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected PNG/JPEG, got '{file.content_type}'"
        )

    t0 = time.time()
    try:
        # Read + preprocess
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("L")
        img = img.resize((28, 28), Image.LANCZOS)

        # Denoise — timed separately
        t_denoise    = time.time()
        denoised     = pipeline(img)
        denoised_pil = transforms.ToPILImage()(denoised)
        denoise_ms   = round((time.time() - t_denoise) * 1000, 2)

        # Normalize + predict
        tensor = normalize(denoised_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = ocr_model.predict_proba(tensor)

        probs_list = probs.squeeze(0).cpu().tolist()
        digit      = int(probs.argmax(dim=1).item())
        confidence = float(probs_list[digit])
        total_ms   = round((time.time() - t0) * 1000, 2)

        return {
            "text"      : str(digit),
            "confidence": round(confidence, 4),
            "denoise_ms": denoise_ms,
            "total_ms"  : total_ms,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")



# =============================================================================
# POST /ocr/batch — predict multiple images in one request (bonus)
# =============================================================================

@app.post("/ocr/batch")
async def ocr_batch(files: list[UploadFile] = File(...)):
    """
    Upload multiple PNG/JPEG images in one request.
    Returns a prediction for each image plus aggregate stats.

    Curl example:
        curl -X POST http://localhost:8000/ocr/batch \
             -F "files=@digit1.png" \
             -F "files=@digit2.png" \
             -F "files=@digit3.png"

    Max 32 images per request to avoid OOM on GPU.
    """
    MAX_BATCH = 32

    # Guards
    if pipeline is None or ocr_model is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "Models not loaded", "missing": load_errors}
        )
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > MAX_BATCH:
        raise HTTPException(
            status_code=400,
            detail=f"Max {MAX_BATCH} images per batch, got {len(files)}."
        )

    allowed = ("image/png", "image/jpeg", "image/jpg", "image/bmp")
    for f in files:
        if f.content_type not in allowed:
            raise HTTPException(
                status_code=400,
                detail=f"File '{f.filename}' has unsupported type '{f.content_type}'. Send PNG or JPEG."
            )

    t_batch_start = time.time()
    results       = []

    for idx, upload in enumerate(files):
        t0 = time.time()
        try:
            # Read + preprocess
            raw = await upload.read()
            img = Image.open(io.BytesIO(raw)).convert("L")
            img = img.resize((28, 28), Image.LANCZOS)

            # Denoise
            t_denoise    = time.time()
            denoised     = pipeline(img)
            denoised_pil = transforms.ToPILImage()(denoised)
            denoise_ms   = round((time.time() - t_denoise) * 1000, 2)

            # Predict
            tensor = normalize(denoised_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = ocr_model.predict_proba(tensor)

            probs_list = probs.squeeze(0).cpu().tolist()
            digit      = int(probs.argmax(dim=1).item())
            confidence = float(probs_list[digit])
            total_ms   = round((time.time() - t0) * 1000, 2)

            results.append({
                "index"     : idx,
                "filename"  : upload.filename,
                "text"      : str(digit),
                "confidence": round(confidence, 4),
                "denoise_ms": denoise_ms,
                "total_ms"  : total_ms,
                "error"     : None,
            })

        except Exception as e:
            # One bad image doesn't kill the whole batch
            results.append({
                "index"     : idx,
                "filename"  : upload.filename,
                "text"      : None,
                "confidence": None,
                "denoise_ms": None,
                "total_ms"  : None,
                "error"     : str(e),
            })

    # Aggregate stats across the batch
    successful   = [r for r in results if r["error"] is None]
    failed       = [r for r in results if r["error"] is not None]
    avg_conf     = round(sum(r["confidence"] for r in successful) / len(successful), 4) if successful else None
    avg_denoise  = round(sum(r["denoise_ms"] for r in successful) / len(successful), 2) if successful else None
    batch_ms     = round((time.time() - t_batch_start) * 1000, 2)

    return {
        "count"          : len(files),
        "succeeded"      : len(successful),
        "failed"         : len(failed),
        "batch_ms"       : batch_ms,
        "avg_confidence" : avg_conf,
        "avg_denoise_ms" : avg_denoise,
        "predictions"    : results,
    }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
