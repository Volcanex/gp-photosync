"""Camera JPEG, master and v1 auto grade at full size for the blind-test
frames that are not already in the v1 sample folder. Read-only on R2."""
import io, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, "/home/gabriel/gp-photosync"); sys.path.insert(0, "/home/gabriel/gp-photosync-auto")
import rawpy, automaster, r2_storage

out = Path("/root/gm/frames"); out.mkdir(exist_ok=True)
metas = {json.loads(l)["stem"]: json.loads(l) for l in open("/root/gm/meta.jsonl")}
for stem in sys.argv[1:]:
    nef = r2_storage.get_bytes("raw", f"{stem}.NEF")
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "s.nef"; src.write_bytes(nef)
        with rawpy.imread(str(src)) as r:
            (out / f"{stem}_camera.jpg").write_bytes(r.extract_thumb().data)
        if not (out / f"{stem}_auto_v1.jpg").exists():
            automaster.render(src, out / f"{stem}_auto_v1.jpg")
    (out / f"{stem}_master.jpg").write_bytes(r2_storage.get_bytes("masters", metas[stem]["master"]))
    print(stem, flush=True)
