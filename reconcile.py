"""Nightly tracker-vs-R2 reconcile. The JSON tracker can drift from the
bucket (uploads outside this server, failed writes); drift makes the panel
and dedup lie. Reports to the alert spine. Read-only, never repairs, because
the fix differs by direction (missing-in-R2 = data loss, alert loudly;
missing-in-tracker = just rebuild).

Run:  python3 reconcile.py   (systemd photo-reconcile.timer, nightly)
"""
import os
import re
import subprocess

import r2_storage
from main import load_tracker

# brain_alerts pulls in fastmcp, which this venv doesn't have, so call the
# spine through g-brain's own interpreter instead of vendoring it.
_BT_DIR = os.environ.get("BRAIN_TOOLS_DIR", "")


def _spine(fn, *args):
    if not _BT_DIR:
        print(f"alert spine not configured (BRAIN_TOOLS_DIR unset): {fn} {args!r}")
        return
    subprocess.run([f"{_BT_DIR}/.venv/bin/python", "-c",
                    f"import sys; sys.path.insert(0, {_BT_DIR!r}); "
                    "import brain_alerts; "
                    f"brain_alerts.{fn}(*{args!r})"],
                   timeout=30)


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

    if any(m for m, _ in drift.values()):
        _spine("send", "warn", "photo-reconcile",
               "Tracker entries missing from R2 (possible data loss): "
               + "; ".join(f"{c}: {', '.join(lost[c])}" for c, (m, u) in drift.items() if m),
               "photo-reconcile-loss")
    elif drift:
        _spine("send", "info", "photo-reconcile",
               "Untracked objects in R2 (tracker behind): "
               + ", ".join(f"{c}:{u}" for c, (m, u) in drift.items() if u),
               "photo-reconcile-drift")
    else:
        _spine("clear", "photo-reconcile-loss")
        _spine("clear", "photo-reconcile-drift")
        print("clean: tracker matches R2")


if __name__ == "__main__":
    run()
