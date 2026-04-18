"""
Stage 2 — Adaptive Huffman Compression Microservice
Port: 8001

Endpoints:
    GET  /health         — liveness check
    POST /compress       — encode text to compressed hex bitstream
    POST /decompress     — decode hex bitstream back to text
    POST /roundtrip      — compress + decompress in one call (useful for testing)
    GET  /stats          — running totals since service started

The compressor uses our hand-rolled FGK adaptive Huffman implementation.
No stdlib compression libraries (zlib, gzip, bz2, lzma) are used anywhere.
"""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from .fgk import encode, decode
from stage2_huffman.metrics import compute_all


# ---------------------------------------------------------------------------
# Request / response models
#
# Pydantic does automatic validation — if a caller sends the wrong type
# or omits a required field, FastAPI returns a 422 with a clear error
# message before our code even runs.
# ---------------------------------------------------------------------------

class CompressRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("text must not be empty")
        return v


class CompressResponse(BaseModel):
    compressed_hex:   str
    original_bytes:   int
    compressed_bytes: int
    original_bits:    int
    compressed_bits:  int
    compression_ratio: float
    entropy_bpc:      float       # Shannon entropy in bits-per-character
    efficiency:       float       # how close we got to theoretical optimum
    encode_ms:        float       # how long encoding took


class DecompressRequest(BaseModel):
    compressed_hex:  str
    compressed_bits: int          # exact bit count — needed to strip padding

    @field_validator("compressed_hex")
    @classmethod
    def hex_must_not_be_empty(cls, v):
        if len(v) == 0:
            raise ValueError("compressed_hex must not be empty")
        return v

    @field_validator("compressed_bits")
    @classmethod
    def bits_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("compressed_bits must be > 0")
        return v


class DecompressResponse(BaseModel):
    text:        str
    decode_ms:   float


class RoundtripResponse(BaseModel):
    original_text:   str
    recovered_text:  str
    lossless:        bool
    compressed_hex:  str
    compressed_bits: int
    compression_ratio: float
    entropy_bpc:     float
    efficiency:      float
    total_ms:        float


# ---------------------------------------------------------------------------
# Service-level stats — accumulated since startup
# ---------------------------------------------------------------------------

@dataclass
class ServiceStats:
    total_compress_calls:   int   = 0
    total_decompress_calls: int   = 0
    total_chars_compressed: int   = 0
    total_chars_recovered:  int   = 0
    total_encode_ms:        float = 0.0
    total_decode_ms:        float = 0.0
    started_at:             float = field(default_factory=time.time)


stats = ServiceStats()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: confirm imports are working and algorithm is functional
    print("\n[Stage 2] Huffman compression service starting...")

    test_text = "startup_check"
    hex_str, orig_bits, comp_bits = encode(test_text)
    recovered = decode(hex_str, comp_bits)

    if recovered != test_text:
        raise RuntimeError(
            f"FGK self-test failed on startup: "
            f"expected {repr(test_text)}, got {repr(recovered)}"
        )

    print("[Stage 2] FGK self-test passed — service ready on port 8001")
    yield
    # Shutdown: nothing to clean up
    print("[Stage 2] Compression service shutting down")


app = FastAPI(
    title="Stage 2 — Adaptive Huffman Compression",
    description=(
        "Lossless text compression using the FGK adaptive Huffman algorithm. "
        "No stdlib compression libraries used."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """
    Liveness check. Returns 200 if the service is up and the algorithm
    is functional. Useful for docker-compose healthchecks and pipeline_runner.
    """
    uptime_s = round(time.time() - stats.started_at, 1)
    return {
        "status":       "ok",
        "algorithm":    "FGK Adaptive Huffman",
        "stdlib_used":  False,
        "uptime_s":     uptime_s,
    }


# ---------------------------------------------------------------------------
# POST /compress
# ---------------------------------------------------------------------------

@app.post("/compress", response_model=CompressResponse)
def compress(req: CompressRequest):
    """
    Compress text using adaptive Huffman encoding.

    The response includes the compressed bitstream as a hex string,
    plus all three compression quality metrics. Keep the compressed_bits
    field — you'll need it to decompress correctly (it tells the decoder
    how many bits are real vs padding in the final byte).
    """
    t0 = time.perf_counter()

    try:
        compressed_hex, original_bits, compressed_bits = encode(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding error: {str(e)}")

    encode_ms = (time.perf_counter() - t0) * 1000

    # Compute all three metrics in one call
    m = compute_all(req.text, original_bits, compressed_bits)

    # Update running stats
    stats.total_compress_calls   += 1
    stats.total_chars_compressed += len(req.text)
    stats.total_encode_ms        += encode_ms

    return CompressResponse(
        compressed_hex=compressed_hex,
        original_bytes=m["original_bytes"],
        compressed_bytes=m["compressed_bytes"],
        original_bits=original_bits,
        compressed_bits=compressed_bits,
        compression_ratio=m["compression_ratio"],
        entropy_bpc=m["entropy_bpc"],
        efficiency=m["efficiency"],
        encode_ms=round(encode_ms, 3),
    )


# ---------------------------------------------------------------------------
# POST /decompress
# ---------------------------------------------------------------------------

@app.post("/decompress", response_model=DecompressResponse)
def decompress(req: DecompressRequest):
    """
    Decompress a hex bitstream back to the original text.

    You must pass compressed_bits (from the /compress response) alongside
    the hex string. This is needed to strip the zero-padding from the
    final byte — without it, the decoder would try to read phantom symbols.
    """
    t0 = time.perf_counter()

    # Validate hex string is actually valid hex before passing to decoder
    try:
        bytes.fromhex(req.compressed_hex)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail="compressed_hex is not valid hexadecimal"
        )

    try:
        recovered = decode(req.compressed_hex, req.compressed_bits)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Decoding error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal decode error: {str(e)}")

    decode_ms = (time.perf_counter() - t0) * 1000

    # Update running stats
    stats.total_decompress_calls += 1
    stats.total_chars_recovered  += len(recovered)
    stats.total_decode_ms        += decode_ms

    return DecompressResponse(
        text=recovered,
        decode_ms=round(decode_ms, 3),
    )


# ---------------------------------------------------------------------------
# POST /roundtrip
# ---------------------------------------------------------------------------

@app.post("/roundtrip", response_model=RoundtripResponse)
def roundtrip(req: CompressRequest):
    """
    Compress then immediately decompress in a single request.

    Primarily useful for:
      - Testing / demo: proves lossless round-trip in one call
      - pipeline_runner.py: can use this for the recorded demo
      - Hackathon presentation: one endpoint that shows the full story

    Returns both the compressed artifact AND the recovered text, plus
    an explicit lossless: true/false field for instant verification.
    """
    t0 = time.perf_counter()

    # Compress
    compressed_hex, original_bits, compressed_bits = encode(req.text)

    # Decompress
    recovered = decode(compressed_hex, compressed_bits)

    total_ms = (time.perf_counter() - t0) * 1000

    # Metrics
    m = compute_all(req.text, original_bits, compressed_bits)

    # Update stats for both operations
    stats.total_compress_calls   += 1
    stats.total_decompress_calls += 1
    stats.total_chars_compressed += len(req.text)
    stats.total_chars_recovered  += len(recovered)

    return RoundtripResponse(
        original_text=req.text,
        recovered_text=recovered,
        lossless=(recovered == req.text),
        compressed_hex=compressed_hex,
        compressed_bits=compressed_bits,
        compression_ratio=m["compression_ratio"],
        entropy_bpc=m["entropy_bpc"],
        efficiency=m["efficiency"],
        total_ms=round(total_ms, 3),
    )


# ---------------------------------------------------------------------------
# GET /stats
# ---------------------------------------------------------------------------

@app.get("/stats")
def get_stats():
    """
    Running totals since service startup.
    Useful for the README latency benchmark table.
    """
    n_enc = stats.total_compress_calls
    n_dec = stats.total_decompress_calls

    return {
        "compress_calls":        n_enc,
        "decompress_calls":      n_dec,
        "total_chars_compressed": stats.total_chars_compressed,
        "total_chars_recovered":  stats.total_chars_recovered,
        "avg_encode_ms":         round(stats.total_encode_ms / n_enc, 3) if n_enc else 0,
        "avg_decode_ms":         round(stats.total_decode_ms / n_dec, 3) if n_dec else 0,
        "uptime_s":              round(time.time() - stats.started_at, 1),
    }


# ---------------------------------------------------------------------------
# Run directly: uvicorn stage2_huffman.app:app --port 8001
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "stage2_huffman.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
