"""Build the v2 sample folder pages from rendered files.

    build_pages.py SRC_DIR OUT_DIR
SRC_DIR holds <stem>_<candidate>.jpg files (candidate in CANDIDATES order,
plus 'master' where he has one). Writes into OUT_DIR:
  index.html     labelled comparison for the 16 sample frames
  blind.html     held-out frames, candidates shuffled and lettered, next to his master
  blind-key.html the letter -> candidate key (open after choosing)
Images are copied as 2560px JPEGs for the pages; full-size files stay in full/.
"""
import html, json, random, shutil, sys
from pathlib import Path

from PIL import Image

CANDIDATES = ["camera", "auto_v1", "ridge", "gbm", "lut_blend"]
LABELS = {"camera": "Camera JPEG", "auto_v1": "Auto grade v1 (darktable)", "ridge": "Model: regression",
          "gbm": "Model: boosted trees", "lut_blend": "Model: LUT blend", "master": "Your master"}
CSS = """<style>
:root{--bg:#f6f5f2;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#dedcd6}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--fg:#ecebe7;--mut:#9a9893;--card:#1d1d1b;--line:#33322f}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif}
main{max-width:1500px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:8px 0}h2{font-size:17px;margin:28px 0 8px}
p{color:var(--mut);max-width:70ch}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
figure img{width:100%;display:block}
figcaption{padding:6px 10px;font-size:13px}
figure.master{outline:2px solid var(--fg)}
</style>"""


def small(src, dst, n=2560):
    im = Image.open(src); im.thumbnail((n, n)); im.convert("RGB").save(dst, "JPEG", quality=88)


def main(src, out, sample_stems, blind_stems, seed=20261112):
    src, out = Path(src), Path(out)
    (out / "img").mkdir(parents=True, exist_ok=True); (out / "full").mkdir(exist_ok=True)
    have = {}
    for f in sorted(src.glob("*.jpg")):
        stem, cand = f.stem.split("_", 1)[0] + "_" + f.stem.split("_", 2)[1], f.stem.split("_", 2)[2]
        have.setdefault(stem, {})[cand] = f
        if cand in ("ridge", "gbm", "lut_blend"):  # full size only for the new renders, to judge noise
            shutil.copy(f, out / "full" / f.name)
        small(f, out / "img" / f.name)

    def fig(stem, cand, caption, cls=""):
        name = have[stem][cand].name
        href = f"full/{name}" if cand in ("ridge", "gbm", "lut_blend") else f"img/{name}"
        return (f'<figure class="{cls}"><a href="{href}"><img loading="lazy" src="img/{name}" alt="{html.escape(caption)}"></a>'
                f'<figcaption>{html.escape(caption)}</figcaption></figure>')

    parts = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Auto master samples v2</title>{CSS}<main>",
             "<h1>Auto master samples, v2</h1><p>Each row is one frame. Candidates are rendered from the RAW with the grading model "
             "learned from your masters (no denoise, no sharpening). Frames with a master were held out of training. Click any picture for the full-size file.</p>"]
    for stem in sample_stems:
        if stem not in have:
            continue
        figs = [fig(stem, c, LABELS[c]) for c in CANDIDATES if c in have[stem]]
        if "master" in have[stem]:
            figs.append(fig(stem, "master", LABELS["master"], "master"))
        parts.append(f"<h2>{stem}</h2><div class=row>{''.join(figs)}</div>")
    (out / "index.html").write_text("".join(parts) + "</main>")

    rng = random.Random(seed); key = {}
    parts = [f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Blind test</title>{CSS}<main>",
             "<h1>Blind test</h1><p>Your master is outlined on the left of each row. The lettered pictures are the camera JPEG, the old auto grade and the "
             "learned models, in a different random order on every row. Note your favourite letter per row, then open blind-key.html.</p>"]
    for stem in blind_stems:
        if stem not in have or "master" not in have[stem]:
            continue
        cands = [c for c in CANDIDATES if c in have[stem]]
        rng.shuffle(cands)
        key[stem] = {chr(65 + i): c for i, c in enumerate(cands)}
        def blind_fig(c, letter):
            # neutral filenames, so the picture's address cannot give the answer away
            name = f"{stem}_{letter}.jpg"
            small(have[stem][c], out / "blind" / name)
            return (f'<figure><a href="blind/{name}"><img loading="lazy" src="blind/{name}" alt="{letter}"></a>'
                    f'<figcaption>{letter}</figcaption></figure>')
        (out / "blind").mkdir(exist_ok=True)
        figs = [fig(stem, "master", "Your master", "master")] + [blind_fig(c, chr(65 + i)) for i, c in enumerate(cands)]
        parts.append(f"<h2>{stem}</h2><div class=row>{''.join(figs)}</div>")
    (out / "blind.html").write_text("".join(parts) + "</main>")
    rows = "".join(f"<tr><td>{s}</td>" + "".join(f"<td>{l}: {html.escape(LABELS[c])}</td>" for l, c in k.items()) + "</tr>" for s, k in key.items())
    (out / "blind-key.html").write_text(f"<!doctype html><meta charset=utf-8><title>Blind test key</title>{CSS}<main><h1>Blind test key</h1><table>{rows}</table></main>")
    (out / "blind-key.json").write_text(json.dumps(key, indent=1))
    return key


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[3]))
    main(sys.argv[1], sys.argv[2], cfg["samples"], cfg["blind"])
