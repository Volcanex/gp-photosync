"""Nightly tracker-vs-R2 reconcile. The JSON tracker can drift from the
bucket (uploads outside this server, failed writes); drift makes the panel
and dedup lie. Reports to the alert spine. Read-only, never repairs, because
the fix differs by direction (missing-in-R2 = data loss, alert loudly;
missing-in-tracker = just rebuild).

Run:  python3 reconcile.py   (systemd photo-reconcile.timer, nightly)
"""
import os
import subprocess

import r2_storage
from main import load_tracker

# brain_alerts pulls in fastmcp, which this venv doesn't have, so call the
# spine through brain-tools' own interpreter instead of vendoring it.
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


def run():
    tracker = load_tracker()
    drift = {}
    for cat in ("raw", "masters", "compressed"):
        tracked = set(tracker.get(cat, {}))
        actual = {o.filename for o in r2_storage.list_category(cat)}
        missing_r2 = tracked - actual
        untracked = actual - tracked
        if missing_r2 or untracked:
            drift[cat] = (len(missing_r2), len(untracked))
        print(f"{cat}: tracked={len(tracked)} r2={len(actual)} "
              f"missing_in_r2={len(missing_r2)} untracked={len(untracked)}")

    if any(m for m, _ in drift.values()):
        _spine("send", "warn", "photo-reconcile",
               "Tracker entries missing from R2 (possible data loss): "
               + ", ".join(f"{c}:{m}" for c, (m, u) in drift.items() if m),
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
