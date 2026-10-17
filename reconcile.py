"""Nightly tracker-vs-R2 reconcile. The JSON tracker can drift from the
bucket (uploads outside this server, failed writes); drift makes the panel
and dedup lie. Reports to the alert spine. Read-only, never repairs, because
the fix differs by direction (missing-in-R2 = data loss, alert loudly;
missing-in-tracker = just rebuild).

Run:  python3 reconcile.py   (systemd photo-reconcile.timer, nightly)
"""
import hashlib
import json
import os
import re
import subprocess

import r2_storage
from main import LOGS_DIR, load_tracker

# The last finding each alert paged. The timer runs nightly and the spine's
# cooldown is six hours, so without this one unchanged finding paged every
# night (DSC_3970.jpg paged four mornings running, 2026-10-14..17). A finding
# now pages when it first appears or changes, and is forgotten once it clears.
PAGED = LOGS_DIR / "reconcile-paged.json"

# brain_alerts pulls in fastmcp, which this venv doesn't have, so call the
# spine through g-brain's own interpreter instead of vendoring it.
_BT_DIR = os.environ.get("BRAIN_TOOLS_DIR", "")


def _spine(fn, *args):
    if not _BT_DIR:
        print(f"alert spine not configured (BRAIN_TOOLS_DIR unset): {fn} {args!r}")
        return
    out = subprocess.run([f"{_BT_DIR}/.venv/bin/python", "-c",
                          f"import json, sys; sys.path.insert(0, {_BT_DIR!r}); "
                          "import brain_alerts; "
                          f"print(json.dumps(brain_alerts.{fn}(*{args!r})))"],
                         capture_output=True, text=True, timeout=30).stdout
    print(out, end="")
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return None


def _load_paged():
    try:
        return json.loads(PAGED.read_text())
    except (OSError, ValueError):
        return {}


def _save_paged(paged):
    PAGED.parent.mkdir(parents=True, exist_ok=True)
    tmp = PAGED.with_suffix(".tmp")
    tmp.write_text(json.dumps(paged, indent=2))
    os.replace(tmp, PAGED)


def _page_on_change(paged, key, level, msg):
    """Page `msg` under `key` unless the same finding was already paged."""
    sig = hashlib.sha256(msg.encode()).hexdigest()[:16]
    if paged.get(key) == sig:
        print(f"unchanged since last page, not re-paging: {msg}")
        return
    if key in paged:
        _spine("clear", key)  # a changed finding must not sit out the old cooldown
    result = _spine("send", level, "photo-reconcile", msg, key)
    if result and result.get("ok"):
        paged[key] = sig  # a failed push stays unrecorded, so the next run retries


_PART = re.compile(r"\.part-\d{4}$")


def _files(names):
    """Object names as the files they hold. A chunked upload is stored as
    `<name>.manifest.json` plus `<name>.part-NNNN` and counts as `<name>`;
    macOS `._` sidecars are not photos."""
    out = set()
    for n in names:
        if n.startswith("._") or _PART.search(n):
            continue
        out.add(n.removesuffix(".manifest.json"))
    return out


def run():
    tracker = load_tracker()
    drift = {}
    lost = {}
    for cat in ("raw", "masters", "compressed"):
        tracked = _files(tracker.get(cat, {}))
        actual = _files(o.filename for o in r2_storage.list_category(cat))
        missing_r2 = tracked - actual
        lost[cat] = sorted(missing_r2)
        untracked = actual - tracked
        if missing_r2 or untracked:
            drift[cat] = (len(missing_r2), len(untracked))
        print(f"{cat}: tracked={len(tracked)} r2={len(actual)} "
              f"missing_in_r2={len(missing_r2)} untracked={len(untracked)}")

    paged = _load_paged()
    if any(m for m, _ in drift.values()):
        _page_on_change(paged, "photo-reconcile-loss", "warn",
                        "Tracker entries missing from R2 (possible data loss): "
                        + "; ".join(f"{c}: {', '.join(lost[c])}" for c, (m, u) in drift.items() if m))
    else:
        paged.pop("photo-reconcile-loss", None)
        _spine("clear", "photo-reconcile-loss")
    untracked = {c: u for c, (m, u) in drift.items() if u}
    if untracked:
        _page_on_change(paged, "photo-reconcile-drift", "info",
                        "Untracked objects in R2 (tracker behind): "
                        + ", ".join(f"{c}:{u}" for c, u in untracked.items()))
    else:
        paged.pop("photo-reconcile-drift", None)
        _spine("clear", "photo-reconcile-drift")
    if not drift:
        print("clean: tracker matches R2")
    _save_paged(paged)


if __name__ == "__main__":
    run()
