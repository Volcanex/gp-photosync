# gp-photosync

A small FastAPI server that takes photo uploads from a Mac client and stores them in Cloudflare R2. It keeps raws, edited masters and compressed copies apart, makes thumbnails on request and caches them on disk, and can ping a website to rebuild its photo list after an upload. A nightly script checks the server's own records against the bucket and reports anything that has drifted.

It has no web interface of its own, so there's no screenshot.

## Running it

You need Python 3.11 or 3.12 and an R2 bucket, and the bucket's credentials go in `.env.r2` with the rest of the settings.

```sh
python3 -m venv venv
venv/bin/pip install -r requirements.txt
cp .env.example .env.r2    # then fill in the R2 values
set -a && . ./.env.r2 && set +a
venv/bin/python main.py
```

It listens on 127.0.0.1:5656 by default, and `HOST` and `PORT` change that. `GET /health` checks that the bucket is reachable, and `python3 reconcile.py` runs the drift check by hand.

The API has no auth of its own, so it belongs behind a proxy that does the authenticating. Don't expose the port directly. Deletes need an `X-Delete-Token` header that matches `PHOTO_DELETE_TOKEN`, and if that isn't set, every delete is refused.

## Licence

Source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE). You can read it, run it and change it for any noncommercial purpose.

Required Notice: Copyright Gabriel Penman (https://gabrielpenman.com)
