# gp-photosync

Repo `Volcanex/gp-photosync` (was `photo-sync-server`), checked out at `/home/gabriel/gp-photosync`;
`/home/gabriel/photo-sync-server` is a compat symlink (g-site reads the R2 mount through it).
Unit names (`photo-sync-server`, `photo-reconcile`, `r2-photos-mount`) are unchanged.

**Owner:** Gabriel Penman  
R2-backed FastAPI server that receives photo uploads from macOS (raw, masters, compressed) and stores them in Cloudflare R2. Generates thumbnails on demand with local disk cache. Signals the site to rebuild its photo manifest after each upload (debounced 30s).

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

## Gotchas

- `file_tracker.json` is the in-process duplicate-detection index. If R2 and the tracker drift, rebuild with: `python3 -c "import main; main.rebuild_tracker_from_r2()"`
- `venv/` and `storage/` are gitignored. The venv must be present for the unit to start.
- `thumb_cache/` holds generated JPEG thumbnails: safe to wipe, regenerated on demand.
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
- Thumbnails cover all three categories; first generation is ~2s/photo
  (full object fetched from R2), then cached in `thumb_cache/`.
