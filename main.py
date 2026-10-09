#!/usr/bin/env python3
"""
Photo Sync Server, R2-backed edition.
Same HTTP API as the original filesystem version; storage is Cloudflare R2.

Client (macOS) POSTs to /upload/raw, /upload/masters, /upload/compressed with a
file. Bytes are streamed into R2 at `data/photos/<category>/<filename>`.

Duplicate detection: sha256 of body is stored in x-amz-meta-sha256. A per-server
in-memory index is seeded from the local tracker JSON (which can be bootstrapped
via `python3 -c "import main; main.rebuild_tracker_from_r2()"`).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import rawpy
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from PIL import Image

import r2_storage

BASE_DIR = Path(__file__).parent
THUMB_CACHE = BASE_DIR / "thumb_cache"
LOGS_DIR = BASE_DIR / "logs"
TRACKER_FILE = BASE_DIR / "file_tracker.json"
THUMB_CACHE.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = ("raw", "masters", "compressed")

THUMBNAIL_SIZE = (400, 400)
THUMBNAIL_QUALITY = 85
SIZE_PRESETS = {"tiny": 100, "small": 150, "medium": 300, "large": 400, "huge": 600}
QUALITY_PRESETS = {"low": 60, "medium": 75, "high": 85, "max": 95}
RAW_EXTS = {".arw", ".cr2", ".cr3", ".nef", ".orf", ".raf", ".rw2", ".dng", ".raw"}
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mpg", ".mpeg", ".m4v", ".mkv"}

# Display-quality derivative made from the uploaded original. Matches what
# the Mac client used to produce in its "phase 1" so the website manifest
# and the *_compressed.jpg naming stay unchanged.
COMPRESSED_MAX_DIM = int(os.environ.get("COMPRESSED_MAX_DIM", "5120"))
COMPRESSED_QUALITY = int(os.environ.get("COMPRESSED_QUALITY", "92"))
MEDIA_TYPES = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".gif": "image/gif", ".bmp": "image/bmp", ".tiff": "image/tiff",
    ".tif": "image/tiff",
}

_tracker_lock = threading.Lock()


# --- Optional website manifest rebuild hook (off unless BLOG_REBUILD_URL is set) ---
BLOG_REBUILD_URL = os.environ.get("BLOG_REBUILD_URL", "")
BLOG_REBUILD_TOKEN = os.environ.get("BLOG_REBUILD_TOKEN", os.environ.get("DEV_ADMIN_TOKEN", ""))
BLOG_REBUILD_DEBOUNCE_SECONDS = int(os.environ.get("BLOG_REBUILD_DEBOUNCE_SECONDS", "30"))
_rebuild_timer: Optional[threading.Timer] = None
_rebuild_lock = threading.Lock()


def _signal_blog_manifest_rebuild() -> None:
    if not BLOG_REBUILD_URL:
        return
    try:
        req = urllib.request.Request(
            BLOG_REBUILD_URL,
            data=b"{}",
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Admin-Password": BLOG_REBUILD_TOKEN,
            },
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            body = response.read(512).decode("utf-8", errors="replace")
        log_action("REBUILD", f"Manifest rebuild signalled ({response.status}): {body[:200]}", "manifest.json")
    except urllib.error.HTTPError as e:
        body = e.read(512).decode("utf-8", errors="replace")
        log_action("ERROR", f"Manifest rebuild signal failed HTTP {e.code}: {body[:200]}", "manifest.json")
    except Exception as e:
        log_action("ERROR", f"Manifest rebuild signal error: {e}", "manifest.json")


def trigger_blog_rebuild(delay: int = BLOG_REBUILD_DEBOUNCE_SECONDS) -> None:
    global _rebuild_timer
    with _rebuild_lock:
        if _rebuild_timer is not None:
            _rebuild_timer.cancel()
        _rebuild_timer = threading.Timer(delay, _signal_blog_manifest_rebuild)
        _rebuild_timer.daemon = True
        _rebuild_timer.start()


# --- Tracker JSON ----------------------------------------------------------

def load_tracker() -> dict:
    with _tracker_lock:
        if TRACKER_FILE.exists():
            with open(TRACKER_FILE) as f:
                data = json.load(f)
            for cat in CATEGORIES:
                data.setdefault(cat, {})
            return data
        return {cat: {} for cat in CATEGORIES}


def save_tracker(tracker: dict) -> None:
    with _tracker_lock:
        tmp = TRACKER_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(tracker, f, indent=2)
        os.replace(tmp, TRACKER_FILE)


def rebuild_tracker_from_r2() -> dict:
    """Reconstruct the tracker by listing R2 and reading x-amz-meta-sha256."""
    tracker = {cat: {} for cat in CATEGORIES}
    for cat in CATEGORIES:
        for info in r2_storage.list_category(cat):
            # We don't list sha256 in list-objects; HEAD each to get metadata
            h = r2_storage.head(cat, info.filename)
            sha = ""
            if h:
                sha = h.get("Metadata", {}).get("sha256", "")
            tracker[cat][info.filename] = {
                "hash": sha,
                "size": info.size,
                "uploaded": info.last_modified,
                "original_name": info.filename,
            }
    save_tracker(tracker)
    return tracker


# --- Helpers ---------------------------------------------------------------

def log_action(action_type: str, message: str, file_name: Optional[str] = None) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{ts}] [{action_type}] {message}"
    if file_name:
        entry += f" - {file_name}"
    log_file = LOGS_DIR / f"sync_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, "a") as f:
        f.write(entry + "\n")
    print(entry)


def get_unique_filename(category: str, filename: str) -> str:
    stem = Path(filename).stem
    ext = Path(filename).suffix
    counter = 1
    candidate = filename
    while r2_storage.exists(category, candidate):
        candidate = f"{stem} ({counter}){ext}"
        counter += 1
    return candidate


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- App ------------------------------------------------------------------

app = FastAPI(
    title="Photo Sync Server (R2)",
    description="R2-backed photo sync API",
    version="2.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"name": "Photo Sync Server", "version": "2.0.0", "backend": "r2", "docs": "/docs"}


@app.get("/health")
async def health_check():
    r2_ok, r2_detail = r2_storage.check()
    return {
        "ok": r2_ok,
        "backend": "r2",
        "bucket": r2_storage.R2_BUCKET,
        "prefix": r2_storage.R2_PREFIX,
        "timestamp": datetime.now().isoformat(),
        "checks": {
            "r2": {"ok": r2_ok, "detail": r2_detail},
        },
    }


@app.get("/storage")
async def storage_info():
    try:
        counts = {cat: 0 for cat in CATEGORIES}
        totals = {cat: 0 for cat in CATEGORIES}
        for cat in CATEGORIES:
            for info in r2_storage.list_category(cat):
                counts[cat] += 1
                totals[cat] += info.size
        return {
            "backend": "r2",
            "bucket": r2_storage.R2_BUCKET,
            "prefix": r2_storage.R2_PREFIX,
            "counts": counts,
            "bytes": totals,
            "total_count": sum(counts.values()),
            "total_bytes": sum(totals.values()),
        }
    except Exception as e:
        log_action("ERROR", f"storage_info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _file_entry(info: r2_storage.ObjectInfo) -> dict:
    return {
        "name": info.filename,
        "path": info.filename,
        "size_bytes": info.size,
        "size_mb": round(info.size / (1024 ** 2), 2),
        "modified": info.last_modified,
        "etag": info.etag,
    }


@app.get("/files/raw")
async def list_raw_files():
    try:
        files = [_file_entry(i) for i in r2_storage.list_category("raw")]
        files.sort(key=lambda x: x["modified"], reverse=True)
        return {"count": len(files), "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/masters")
async def list_masters_files():
    try:
        files = [_file_entry(i) for i in r2_storage.list_category("masters")]
        files.sort(key=lambda x: x["modified"], reverse=True)
        return {"count": len(files), "files": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files/all")
async def list_all_photos():
    try:
        out = {}
        for cat in CATEGORIES:
            files = [_file_entry(i) for i in r2_storage.list_category(cat)]
            files.sort(key=lambda x: x["modified"], reverse=True)
            out[cat] = {"count": len(files), "files": files}
        out["total_count"] = sum(out[c]["count"] for c in CATEGORIES)
        return out
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Thumbnail -------------------------------------------------------------

def _thumb_cache_path(photo_type: str, filename: str, size: int, quality: int) -> Path:
    return THUMB_CACHE / f"{photo_type}_{filename}_s{size}_q{quality}.jpg"


def _generate_thumbnail(photo_type: str, filename: str, size: int, quality: int) -> Path:
    thumb_path = _thumb_cache_path(photo_type, filename, size, quality)
    if thumb_path.exists():
        return thumb_path

    data = r2_storage.get_bytes(photo_type, filename)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Photo not found: {filename}")

    ext = Path(filename).suffix.lower()
    try:
        if ext in RAW_EXTS:
            with rawpy.imread(io.BytesIO(data)) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        img = Image.open(io.BytesIO(thumb.data))
                    else:
                        rgb = raw.postprocess(use_camera_wb=True, half_size=True)
                        img = Image.fromarray(rgb)
                except Exception:
                    rgb = raw.postprocess(use_camera_wb=True, half_size=True)
                    img = Image.fromarray(rgb)
        else:
            img = Image.open(io.BytesIO(data))

        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        if img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        img.save(thumb_path, "JPEG", quality=quality, optimize=True)
        log_action("THUMBNAIL", f"Generated (size={size}, quality={quality})", filename)
        return thumb_path
    except Exception as e:
        log_action("ERROR", f"Thumb gen failed: {e}", filename)
        raise


@app.get("/thumbnail/{photo_type}/{filename}")
async def get_thumbnail(
    photo_type: str,
    filename: str,
    size: Optional[str] = Query(None),
    quality: Optional[str] = Query(None),
):
    if photo_type not in ("raw", "masters", "compressed"):
        raise HTTPException(status_code=400, detail="photo_type must be 'raw', 'masters', or 'compressed'")
    max_size = THUMBNAIL_SIZE[0]
    if size:
        if size.lower() in SIZE_PRESETS:
            max_size = SIZE_PRESETS[size.lower()]
        else:
            try:
                max_size = int(size)
                if not 50 <= max_size <= 2000:
                    raise HTTPException(status_code=400, detail="size 50-2000")
            except ValueError:
                raise HTTPException(status_code=400, detail=f"bad size: {size}")
    jpeg_quality = THUMBNAIL_QUALITY
    if quality:
        if quality.lower() in QUALITY_PRESETS:
            jpeg_quality = QUALITY_PRESETS[quality.lower()]
        else:
            try:
                jpeg_quality = int(quality)
                if not 1 <= jpeg_quality <= 100:
                    raise HTTPException(status_code=400, detail="quality 1-100")
            except ValueError:
                raise HTTPException(status_code=400, detail=f"bad quality: {quality}")

    thumb_path = _generate_thumbnail(photo_type, filename, max_size, jpeg_quality)
    return FileResponse(
        thumb_path,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "public, max-age=86400",
            "Content-Disposition": f'inline; filename="thumb_{filename}.jpg"',
        },
    )


@app.get("/photo/{photo_type}/{filename}")
async def get_full_photo(photo_type: str, filename: str):
    if photo_type not in ("raw", "masters", "compressed"):
        raise HTTPException(status_code=400, detail="photo_type must be 'raw', 'masters', or 'compressed'")
    # If public base is set, redirect to direct URL (fastest, offloads from server)
    pub = r2_storage.public_url(photo_type, filename)
    if pub:
        return RedirectResponse(url=pub, status_code=302)
    # Else stream from R2
    res = r2_storage.get_stream(photo_type, filename)
    if res is None:
        raise HTTPException(status_code=404, detail=f"Photo not found: {filename}")
    body, length, content_type = res
    ext = Path(filename).suffix.lower()
    media_type = MEDIA_TYPES.get(ext, content_type or "application/octet-stream")
    log_action("DOWNLOAD", f"Serving {photo_type} photo", filename)
    headers = {"Content-Length": str(length), "Content-Disposition": f'attachment; filename="{filename}"'}

    def iter_chunks():
        while True:
            chunk = body.read(64 * 1024)
            if not chunk:
                break
            yield chunk

    return StreamingResponse(iter_chunks(), media_type=media_type, headers=headers)


# --- Upload helpers --------------------------------------------------------

# --- Compressed derivative (server-side) ------------------------------------
#
# This used to run on the Mac: the client decoded every NEF at full resolution
# with rawpy and uploaded the JPEG itself, which cost ~20s/photo and put a
# multi-hour preview pass in front of the originals actually reaching R2. The
# server already decodes RAW for thumbnails and now owns this too, so the
# client's job is reduced to shipping bytes.
#
# Serialised behind a lock on purpose: a full-resolution postprocess of a 24MP
# NEF allocates a ~72MB RGB array plus rawpy's own buffers, and this box has
# 4GB shared with every other service on it. One at a time is slower per photo
# and survives a card dump of 2000 files, which parallel decoding would not.

_derive_lock = threading.Lock()


def _compressed_name(filename: str) -> str:
    return f"{Path(filename).stem}_compressed.jpg"


def _render_display_jpeg(filename: str, data: bytes) -> Optional[bytes]:
    """Full-quality display JPEG from an original. None if not an image."""
    ext = Path(filename).suffix.lower()
    if ext in VIDEO_EXTS:
        # Video transcoding stays with the client (it has ffmpeg); the server
        # has no encoder and silently producing nothing is better than a
        # half-broken MP4.
        return None

    if ext in RAW_EXTS:
        img = None
        # A Nikon NEF carries the camera's own full-size JPEG render inside it.
        # Lifting that out costs a JPEG decode; demosaicing the sensor data with
        # postprocess() costs ~20s and ~70MB per frame. For a *display*
        # derivative the embedded render is what the photographer saw on the
        # back of the camera, so it is the better output as well as the cheaper
        # one. Only fall back to a real decode when the embedded image is
        # missing or too small to serve.
        with rawpy.imread(io.BytesIO(data)) as raw:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    candidate = Image.open(io.BytesIO(thumb.data))
                else:
                    candidate = Image.fromarray(thumb.data)
                if max(candidate.size) >= COMPRESSED_MAX_DIM:
                    img = candidate
            except Exception:
                img = None
            if img is None:
                rgb = raw.postprocess(
                    use_camera_wb=True, half_size=False, no_auto_bright=False
                )
                img = Image.fromarray(rgb)
    else:
        img = Image.open(io.BytesIO(data))

    if img.mode != "RGB":
        img = img.convert("RGB")
    if max(img.size) > COMPRESSED_MAX_DIM:
        img.thumbnail((COMPRESSED_MAX_DIM, COMPRESSED_MAX_DIM), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=COMPRESSED_QUALITY, optimize=True)
    return buf.getvalue()


def derive_compressed(filename: str, data: Optional[bytes] = None,
                      force: bool = False) -> str:
    """Make and store the compressed derivative for one original.

    Returns a short status string; never raises, because this runs on a
    background thread where an exception would be invisible.
    """
    name = _compressed_name(filename)
    try:
        if not force and r2_storage.exists("compressed", name):
            return "exists"
        payload = data
        if payload is None:
            payload = r2_storage.get_bytes("raw", filename)
            if payload is None:
                return "source-missing"
        jpeg = _render_display_jpeg(filename, payload)
        if jpeg is None:
            return "unsupported"
        r2_storage.put_bytes("compressed", name, jpeg, content_type="image/jpeg")
        # Only the tracker needs serialising: it is a whole-file JSON
        # read-modify-write, so two concurrent derivations would lose one of
        # their entries. The R2 round-trips either side are safe in parallel.
        with _derive_lock:
            tracker = load_tracker()
            tracker["compressed"][name] = {
                "hash": sha256_bytes(jpeg),
                "size": len(jpeg),
                "uploaded": datetime.now().isoformat(),
                "original_name": name,
                "original": filename,
                "status": "synced",
                "derived_by": "server",
            }
            save_tracker(tracker)
        log_action("DERIVE", f"compressed {len(jpeg)/(1024**2):.2f} MB", name)
        return "derived"
    except Exception as e:
        log_action("ERROR", f"derive failed: {type(e).__name__}: {e}", filename)
        return f"error:{type(e).__name__}"


def _derive_async(filename: str, data: bytes) -> None:
    threading.Thread(
        target=derive_compressed, args=(filename, data), daemon=True,
        name=f"derive-{filename}",
    ).start()


def missing_compressed() -> list[str]:
    """Originals in raw/ with no compressed derivative."""
    have = {o.filename for o in r2_storage.list_category("compressed")}
    out = []
    for obj in r2_storage.list_category("raw"):
        name = obj.filename
        # AppleDouble turds ("._NAME") were uploaded by an early client and are
        # not images; deriving from them fails on every pass and would otherwise
        # sit at the head of the backlog forever.
        if name.startswith("._") or name in (".DS_Store",):
            continue
        if Path(name).suffix.lower() in VIDEO_EXTS:
            continue
        if _compressed_name(name) not in have:
            out.append(name)
    return sorted(out)


_backfill_state = {"running": False, "done": 0, "total": 0, "current": None}


# Each file costs two R2 round-trips and one JPEG decode, so throughput is
# bound by the network, not the CPU. A small pool multiplies it without
# multiplying the decode peak beyond what this box can spare.
DERIVE_WORKERS = 3


def _backfill_worker(names: list[str]) -> None:
    _backfill_state.update(running=True, done=0, total=len(names), current=None)
    try:
        with ThreadPoolExecutor(max_workers=DERIVE_WORKERS) as pool:
            for name, _ in zip(names, pool.map(derive_compressed, names)):
                _backfill_state["current"] = name
                _backfill_state["done"] += 1
    finally:
        _backfill_state.update(running=False, current=None)


async def _upload_common(category: str, file: UploadFile, content_type_fallback: str) -> dict:
    log_action("UPLOAD_START", f"Receiving {category} file", file.filename)
    content = await file.read()
    file_hash = sha256_bytes(content)
    file_size = len(content)

    tracker = load_tracker()

    # Duplicate check
    for tracked_name, tracked_data in tracker[category].items():
        if tracked_data.get("hash") == file_hash and tracked_data.get("size") == file_size:
            log_action("SKIP", f"Duplicate (matches {tracked_name})", file.filename)
            return {
                "status": "duplicate",
                "message": f"File already exists as {tracked_name}",
                "matched_file": tracked_name,
            }

    # Filename conflict handling
    final_name = file.filename
    if r2_storage.exists(category, file.filename):
        h = r2_storage.head(category, file.filename)
        existing_sha = (h.get("Metadata", {}).get("sha256", "") if h else "")
        if existing_sha and existing_sha != file_hash:
            final_name = get_unique_filename(category, file.filename)
            log_action("RENAME", f"Same name, different content -> {final_name}", file.filename)

    content_type = file.content_type or content_type_fallback
    r2_storage.put_bytes(category, final_name, content, content_type=content_type)

    tracker[category][final_name] = {
        "hash": file_hash,
        "size": file_size,
        "uploaded": datetime.now().isoformat(),
        "original_name": file.filename,
    }
    save_tracker(tracker)
    log_action("SUCCESS", f"{category} uploaded ({file_size/(1024**2):.2f} MB)", final_name)

    return {
        "status": "success",
        "filename": final_name,
        "size_bytes": file_size,
        "size_mb": round(file_size / (1024 ** 2), 2),
        "hash": file_hash,
        # Callers that derive from the original need the bytes; _upload_common
        # has already consumed the UploadFile stream so they cannot re-read it.
        "_content": content,
    }


def _strip_private(res: dict) -> dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


@app.post("/upload/raw")
async def upload_raw(file: UploadFile = File(...)):
    try:
        res = await _upload_common("raw", file, "application/octet-stream")
        if res.get("status") == "success":
            # Derive the display JPEG here rather than making the client do it.
            _derive_async(res["filename"], res["_content"])
            trigger_blog_rebuild()
        return _strip_private(res)
    except HTTPException:
        raise
    except Exception as e:
        log_action("ERROR", f"Upload failed: {e}", file.filename)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/masters")
async def upload_master(file: UploadFile = File(...)):
    try:
        res = await _upload_common("masters", file, "image/jpeg")
        if res.get("status") == "success":
            trigger_blog_rebuild()
        return _strip_private(res)
    except HTTPException:
        raise
    except Exception as e:
        log_action("ERROR", f"Upload failed: {e}", file.filename)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/compressed")
async def upload_compressed(file: UploadFile = File(...), original: str = Form(...)):
    try:
        res = await _upload_common("compressed", file, "image/jpeg")
        res["original"] = original
        return _strip_private(res)
    except HTTPException:
        raise
    except Exception as e:
        log_action("ERROR", f"Upload failed: {e}", file.filename)
        raise HTTPException(status_code=500, detail=str(e))


# --- Delete ----------------------------------------------------------------

def _check_delete_token(request: Request):
    """Deletes are destructive and this app is fronted by an open proxy,
    so require the shared token (gcore/.env) unless none is configured."""
    expected = os.environ.get("PHOTO_DELETE_TOKEN", "")
    if expected and request.headers.get("X-Delete-Token") != expected:
        raise HTTPException(status_code=403, detail="delete token required")


@app.delete("/raw/{filename}")
async def delete_raw_file(filename: str, request: Request):
    _check_delete_token(request)
    if not r2_storage.exists("raw", filename):
        raise HTTPException(status_code=404, detail=f"Not found: {filename}")
    r2_storage.delete("raw", filename)
    tracker = load_tracker()
    tracker["raw"].pop(filename, None)
    save_tracker(tracker)
    log_action("DELETE", "RAW deleted", filename)
    return {"status": "success", "filename": filename}


@app.delete("/masters/{filename}")
async def delete_master_file(filename: str, request: Request):
    _check_delete_token(request)
    if not r2_storage.exists("masters", filename):
        raise HTTPException(status_code=404, detail=f"Not found: {filename}")
    r2_storage.delete("masters", filename)
    tracker = load_tracker()
    tracker["masters"].pop(filename, None)
    save_tracker(tracker)
    log_action("DELETE", "master deleted", filename)
    return {"status": "success", "filename": filename}


@app.get("/derive/missing")
async def derive_missing():
    """Originals with no compressed derivative."""
    names = missing_compressed()
    return {"count": len(names), "files": names[:500], "truncated": len(names) > 500}


@app.get("/derive/status")
async def derive_status():
    return dict(_backfill_state)


@app.post("/derive/backfill")
async def derive_backfill(request: Request, limit: int = Query(0, ge=0)):
    """Generate every missing derivative, one at a time, in the background.

    Token-gated: not destructive, but it is the most expensive thing this
    process can be told to do and the proxy in front of it is open.
    """
    _check_delete_token(request)
    if _backfill_state["running"]:
        return {"status": "already-running", **dict(_backfill_state)}
    names = missing_compressed()
    if limit:
        names = names[:limit]
    if not names:
        return {"status": "nothing-to-do", "count": 0}
    threading.Thread(target=_backfill_worker, args=(names,), daemon=True,
                     name="derive-backfill").start()
    return {"status": "started", "count": len(names)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "5656")))
