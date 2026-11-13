# gp-photosync

Repo `Volcanex/gp-photosync` (was `photo-sync-server`), checked out at `/home/gabriel/gp-photosync`;
`/home/gabriel/photo-sync-server` is a compat symlink (g-site reads the R2 mount through it).
Unit names (`photo-sync-server`, `photo-reconcile`, `r2-photos-mount`) are unchanged.

**Owner:** Gabriel Penman  
R2-backed FastAPI server that receives photo uploads from macOS (raw, masters) and stores them in Cloudflare R2, deriving `compressed` and `auto` copies of each RAW itself, and receives the uploader's heartbeat. Generates thumbnails on demand with local disk cache. Signals the site to rebuild its photo manifest after each upload (debounced 30s).

## Running

systemd unit: `photo-sync-server.service` (runs as user `gabriel`)  
Port: 5656 on 127.0.0.1 (`HOST` overrides the bind)  
Credentials: `/home/gabriel/gp-photosync/.env.r2` (R2 keys + PORT)

```bash
systemctl status photo-sync-server
systemctl restart photo-sync-server
```

R2 photos are FUSE-mounted at `storage/photos/` by `r2-photos-mount.service`.

## Deriving

The client no longer compresses. It uploads originals and this server makes the
5120px display JPEG, via `/derive/missing`, `/derive/status` and
`POST /derive/backfill` (gated by the delete token, because it starts a long job).

- A Nikon NEF carries the camera's own full-size JPEG inside it. `extract_thumb()`
  lifts it for the cost of a JPEG decode; `postprocess()` demosaics the sensor
  data for ~20s and ~70MB a frame. For a *display* copy the embedded render is
  also the better output (it is what the photographer saw on the back of the
  camera), so the full decode is only a fallback for when the embedded one is
  missing or too small.
- Three derive at once. Only the tracker read-modify-write is serialised, so the
  R2 round trips overlap; that is what took a frame from ~8s to ~3.4s. The
  ceiling is memory, not CPU: this box shares 7GB with other services.
- `._` AppleDouble files uploaded by an early client are skipped, not retried.
  ~200 of them are in R2, and they sat at the head of every backfill failing
  with UnidentifiedImageError, while being counted "done", which made a timing
  run read six times faster than it really was.

## Auto masters

Category `auto`: one automatic grade per RAW in Gabriel's own style,
`auto/<stem>_auto.jpg` (5120px, q92), made by `automaster.py`. The site shows the
best version of a frame: his master (matched by filename stem, `-2` style
variants included) > auto master > compressed camera JPEG. Uploading a master
needs nothing from this server: the manifest rebuild picks it over the auto.

- The grade is the "LUT blend" model from `research/grading-model/`, trained on
  ~550 of his RAW -> Lightroom master pairs and picked by him in a blind test
  (2026-11-13) over the camera JPEG, the old darktable grade and two parametric
  models. 12 features from the RAW render, camera JPEG and EXIF -> a tiny MLP ->
  4 blend weights over shared 17^3 LUTs -> one LUT per frame, applied to the
  rawpy linear render (camera WB, normalised to median 0.18, x/(x+0.18)
  encoded). Weights: `automaster_lut.npz` (~270KB, no per-photo data).
  NumPy + Pillow + rawpy only; no torch on this box.
- No denoise, no grain, no sharpening. His masters carry Lightroom noise
  reduction and grain, so brightened dark frames look noisier than his.
- Orientation: rawpy applies the NEF's orientation tag, but a survey of 300
  NEFs found every one tagged 1 (the D610 records no rotation), so portrait
  frames come out landscape, like the camera JPEG. Not detectable from metadata.
- ~8s and ~0.72GB peak per frame (measured as `gabriel` in this venv). Each
  render runs in a child process (`python automaster.py SRC OUTDIR`, JSON on
  the last stdout line) so the service never keeps that memory.
- Uploads and `POST /derive/auto/backfill` (token-gated, `limit=`,
  `files=a.NEF,b.NEF`, `force=true`) feed one queue drained by `AUTO_WORKERS`
  threads (default 1). `GET /derive/auto/missing`, `GET /derive/auto/status`.
  Tracker entries record `model` (`lut-blend-v1`) and the blend `weights`.
- `reconcile.py` diffs `auto` like the other categories and prints the auto
  backlog without paging, since a backlog is normal until the backfill runs.

## Uploader heartbeat

The Mac client (Volcanex/photo-sync-client, "Heartbeat" in its CLAUDE.md)
POSTs its run report to `/heartbeat` after every agent run, through the same
key URL as uploads. Only the latest beat is kept, in `logs/heartbeat.json`;
`GET /heartbeat` shows it with its age; the latest failed beat is kept beside
it as `last_failure`, so a failure followed by a good run still reaches the
nightly reconcile. `reconcile.py` pages the spine on a failure in the last 24h
(`photo-uploader-failing`, keyed on the error text) and on silence
(`photo-uploader-silent`: no beat for 24h while the last one had files pending,
or 72h at all; before any beat, 72h from `heartbeat_watch_since` in
`logs/reconcile-paged.json`), paged once until the finding changes, like the
drift alerts. Malformed beats (non-boolean `ok`, non-object `pending`, bodies
over 64KB) get a 4xx. Like uploads, the endpoint has no auth beyond the key URL.

## Grading model research (research/grading-model/)

Scratch-grade code that learned Gabriel's Lightroom style from ~700 RAW/master
pairs: `extract.py` registers each master onto the camera JPEG and a neutral
rawpy render, `fit_all.py` fits an 11-number edit per photo (WB gains,
saturation, 8-knot tone curve, see `core.py`), `models_ab.py` / `lutnet.py`
predict it from RAW-only features, `evaluate.py` scores on a fixed held-out
split. Only code, aggregate scores and the small parametric models are
committed: this repo is public, so per-photo tables, his presets and images stay
off it. The shipped LUT model is `automaster_lut.npz` at the top level (the
torch checkpoint listed training filenames and was removed). Re-running the
research needs its data pipeline again (`extract.py`); the 2026-11 working data
on h was deleted to free disk.

## Gotchas

- `file_tracker.json` is the in-process duplicate-detection index. If R2 and the tracker drift, rebuild with: `python3 -c "import main; main.rebuild_tracker_from_r2()"`
- `venv/` and `storage/` are gitignored. The venv must be present for the unit to start.
- `thumb_cache/` holds generated JPEG thumbnails: safe to wipe, regenerated on demand.
- Handlers that touch R2 or decode images are plain `def` (threadpool), never `async def`: on the event loop one slow listing froze `/health` past brain-status's 8s probe and paged a false down.
- Every tracker read-modify-write takes `_derive_lock`; uploads once overwrote derive writes and lost 484 compressed entries.
- `logs/reconcile-paged.json` is the last finding each reconcile alert paged. An unchanged finding stays quiet; deleting the file costs one repeat page.
- `main.py.bak-*` files are manual backups; ignore.

## Control surfaces (2026-09-01)

- **Browse/manage UI**: the Photos panel in gcommand (`pages/photos/`), a gated
  proxy to this API and the only delete path.
- **Deletes need a token**: `X-Delete-Token` = `PHOTO_DELETE_TOKEN` from the
  instance passport (the unit loads both env files), and deletes are refused
  when it is unset.
- **No auth of its own**: the app binds 127.0.0.1 and nginx proxies to it
  there. The public path is a key URL held in the instance passport, and outside
  clients (the Mac uploader included) reach the app only through it. Keep it
  that way: never publish the port or front it with an open proxy.
- **`reconcile.py`** (`photo-reconcile.timer`, nightly 04:50) diffs the
  tracker against R2 and pages the spine on drift; read-only by design.
- Thumbnails cover all four categories; first generation is ~2s/photo
  (full object fetched from R2), then cached in `thumb_cache/`.
