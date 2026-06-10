"""
Rebuild data/real_capture/words/manifest.csv from the .npy files on disk.

The manifest is the index that training/evaluate read. If it ever falls out of
sync with the actual seq/<gloss>_<n>.npy files — e.g. it was deleted, edited, or
a capture run started a fresh one instead of appending — the trainer only sees
part of your data (or the wrong classes). This scans every sequence file and
regenerates the manifest from scratch, assigning the SAME train/val split that
capture_words.py uses (every 5th take of a class -> val).

session_id cannot be recovered from the .npy files, so it is preserved from the
existing manifest when present and backfilled as "legacy" for any take missing
from it. Keep this in mind: a rebuild collapses unknown takes into one session.

It only reads the .npy files and rewrites the manifest; it never deletes data.

Usage:
    python capture/rebuild_manifest.py
"""

import csv
import glob
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CAPTURE_OUTPUT_DIR as _OUTPUT_DIR

VAL_EVERY = 5   # must match capture_words.py
MANIFEST_COLUMNS = ["sample_id", "gloss", "subset", "session_id"]   # must match capture_words.py

HERE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(HERE, _OUTPUT_DIR)
SEQ_DIR  = os.path.join(OUT_DIR, "seq")
MANIFEST = os.path.join(OUT_DIR, "manifest.csv")


def _existing_sessions():
    """Map sample_id -> session_id from the current manifest, to survive a
    rebuild. Tolerates the old 3-column manifest (everything becomes legacy)."""
    sessions = {}
    if os.path.isfile(MANIFEST):
        for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
            sessions[r["sample_id"]] = r.get("session_id") or "legacy"
    return sessions

# sample_id is "<gloss>_<index>" with a zero-padded index. The gloss itself may
# contain underscores, so split on the LAST underscore + digits.
_NAME_RE = re.compile(r"^(?P<gloss>.+)_(?P<idx>\d+)$")


def main():
    if not os.path.isdir(SEQ_DIR):
        print(f"[ERROR] No sequence folder at {SEQ_DIR}. Capture first.")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(SEQ_DIR, "*.npy")))
    if not files:
        print(f"[ERROR] No .npy files in {SEQ_DIR}.")
        sys.exit(1)

    sessions = _existing_sessions()   # preserve session_id across the rebuild

    # Group sample ids by gloss so we can assign the split per class.
    by_gloss = defaultdict(list)
    skipped = []
    for path in files:
        sample_id = os.path.splitext(os.path.basename(path))[0]
        m = _NAME_RE.match(sample_id)
        if not m:
            skipped.append(sample_id)
            continue
        by_gloss[m.group("gloss")].append((int(m.group("idx")), sample_id))

    rows = []
    for gloss in sorted(by_gloss):
        for idx, sample_id in sorted(by_gloss[gloss]):
            subset = "val" if (idx > 0 and idx % VAL_EVERY == 0) else "train"
            session_id = sessions.get(sample_id, "legacy")
            rows.append([sample_id, gloss, subset, session_id])

    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(MANIFEST_COLUMNS)
        wr.writerows(rows)

    print(f"Rebuilt manifest from {len(rows)} sequence files -> {MANIFEST}\n")
    print(f"  {'class':<14}{'train':>6}{'val':>5}{'total':>7}")
    for gloss in sorted(by_gloss):
        ids = by_gloss[gloss]
        v = sum(1 for idx, _ in ids if idx > 0 and idx % VAL_EVERY == 0)
        print(f"  {gloss:<14}{len(ids) - v:>6}{v:>5}{len(ids):>7}")
    if skipped:
        print(f"\n  [WARN] {len(skipped)} file(s) had an unexpected name and were "
              f"skipped: {', '.join(skipped[:5])}{'...' if len(skipped) > 5 else ''}")
    if "nothing" not in by_gloss:
        print("\n  [WARN] No 'nothing' class on disk — the model will never stay "
              "silent. Capture it with capture/capture_words.py.")


if __name__ == "__main__":
    main()
