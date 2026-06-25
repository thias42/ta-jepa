"""Tests for the FMA-small manifest builder.

Loads scripts/prepare_fma.py as a module (it's a script, not a package member) and
checks the brittle bits: the mp3 path layout and the 3-header-row / fixed-column
parsing of tracks.csv, including subset filtering and split mapping.
"""

import csv
import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))  # so the script's `import _bootstrap` resolves

_SCRIPT = _SCRIPTS / "prepare_fma.py"
spec = importlib.util.spec_from_file_location("prepare_fma", _SCRIPT)
fma = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fma)


def test_mp3_path_layout(tmp_path):
    assert fma.fma_mp3_path(tmp_path, 5) == tmp_path / "000" / "000005.mp3"
    assert fma.fma_mp3_path(tmp_path, 139000) == tmp_path / "139" / "139000.mp3"


def _row(track_id, split, subset, genre, width=41):
    row = [""] * width
    row[fma.COL_TRACK_ID] = str(track_id)
    row[fma.COL_SPLIT] = split
    row[fma.COL_SUBSET] = subset
    row[fma.COL_GENRE] = genre
    return row


def test_build_filters_subset_and_maps_split(tmp_path):
    audio_root = tmp_path / "fma_small"
    # Two small-subset tracks (one present, one missing) + one medium (excluded).
    for tid in (5, 7):
        p = fma.fma_mp3_path(audio_root, tid)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00")
    csv_path = tmp_path / "tracks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["h0"]); w.writerow(["h1"]); w.writerow(["track_id"])  # 3 header rows
        w.writerow(_row(5, "training", "small", "Rock"))
        w.writerow(_row(7, "test", "small", "Hip-Hop"))
        w.writerow(_row(999, "training", "small", "Jazz"))   # small but mp3 missing
        w.writerow(_row(11, "training", "medium", "Folk"))   # excluded: not small

    out = tmp_path / "fma.jsonl"
    stats = fma.build(csv_path, audio_root, out)
    assert stats["n"] == 2                      # only present small tracks
    assert stats["missing"] == 1                # track 999
    assert stats["splits"] == {"train": 1, "test": 1}
    assert set(stats["genres"]) == {"Rock", "Hip-Hop"}

    from tajepa.data.manifest import read_manifest

    entries = {e.clip_id: e for e in read_manifest(out)}
    assert entries["000005"].domain == "music"
    assert entries["000005"].label == "Rock"
    assert entries["000007"].split == "test"
