"""Download (if needed), extract, and build a manifest for FMA-small.

FMA-small (Defferrard et al.): 8000 30-second mp3 tracks, 8 balanced top-genres,
with an official train/val/test split. Wired in as a **music pretraining** source
(the plan's FMA/MTG-Jamendo slot). We keep ``genre_top`` as a label so the same
held-out-probe machinery can later report a music-genre number too.

Notes:
- FMA zips are bzip2-compressed and trip the system ``unzip`` ("need PK compat
  v4.6"); we extract with Python ``zipfile``, which handles them.
- ``tracks.csv`` has a 3-row header; track_id is column 0, set/split col 31,
  set/subset col 32, track/genre_top col 40.

    python scripts/prepare_fma.py        # uses data/downloads/{fma_small,fma_metadata}.zip
Then:
    python scripts/extract_embeddings.py \
        --manifest data/manifests/fma_small.jsonl \
        --cache data/cache/encodec_24khz/fma_small
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.data.manifest import ManifestEntry, write_manifest

AUDIO_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
META_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
SPLIT_MAP = {"training": "train", "validation": "val", "test": "test"}
COL_TRACK_ID, COL_SPLIT, COL_SUBSET, COL_GENRE = 0, 31, 32, 40


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"Downloading {url} -> {dest} ...")
    urllib.request.urlretrieve(url, dest)
    return dest


def unzip_if_needed(zip_path: Path, out_dir: Path, marker: str) -> None:
    """Extract via Python zipfile (handles FMA's bzip2) unless already present."""
    if next(out_dir.rglob(marker), None) is not None:
        return
    print(f"Extracting {zip_path} -> {out_dir} (large for audio) ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def find_audio_root(audio_dir: Path) -> Path:
    """Dir containing the 3-digit numbered subdirs (mp3 lives at root/NNN/NNNNNN.mp3)."""
    any_mp3 = next(audio_dir.rglob("*.mp3"), None)
    if any_mp3 is None:
        raise FileNotFoundError(f"No mp3 files found under {audio_dir}")
    return any_mp3.parent.parent


def fma_mp3_path(audio_root: Path, track_id: int) -> Path:
    tid = f"{track_id:06d}"
    return audio_root / tid[:3] / f"{tid}.mp3"


def build(tracks_csv: Path, audio_root: Path, out_manifest: Path) -> dict:
    entries: list[ManifestEntry] = []
    missing = 0
    with open(tracks_csv) as f:
        reader = csv.reader(f)
        for _ in range(3):  # skip the 3-row header
            next(reader)
        for row in reader:
            if row[COL_SUBSET] != "small":
                continue
            tid = int(row[COL_TRACK_ID])
            mp3 = fma_mp3_path(audio_root, tid)
            if not mp3.exists():
                missing += 1
                continue
            entries.append(
                ManifestEntry(
                    path=str(mp3.resolve()),
                    domain="music",
                    split=SPLIT_MAP.get(row[COL_SPLIT], "train"),
                    duration=30.0,
                    clip_id=f"{tid:06d}",
                    label=row[COL_GENRE] or None,
                )
            )
    write_manifest(entries, out_manifest)
    splits: dict[str, int] = {}
    genres: set[str] = set()
    for e in entries:
        splits[e.split] = splits.get(e.split, 0) + 1
        if e.label:
            genres.add(e.label)
    return {"n": len(entries), "missing": missing, "splits": splits, "genres": sorted(genres)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-zip", type=Path, default=Path("data/downloads/fma_small.zip"))
    ap.add_argument("--meta-zip", type=Path, default=Path("data/downloads/fma_metadata.zip"))
    ap.add_argument("--audio-dir", type=Path, default=Path("data/fma_small"))
    ap.add_argument("--meta-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/manifests/fma_small.jsonl"))
    ap.add_argument("--download", action="store_true", help="Download zips if missing.")
    args = ap.parse_args()

    if args.download:
        download(META_URL, args.meta_zip)
        download(AUDIO_URL, args.audio_zip)

    unzip_if_needed(args.meta_zip, args.meta_dir, "tracks.csv")
    tracks_csv = next(args.meta_dir.rglob("tracks.csv"))
    unzip_if_needed(args.audio_zip, args.audio_dir, "*.mp3")
    audio_root = find_audio_root(args.audio_dir)

    stats = build(tracks_csv, audio_root, args.out)
    print(f"Wrote {stats['n']} entries to {args.out}")
    print(f"  splits: {stats['splits']}")
    print(f"  genres ({len(stats['genres'])}): {stats['genres']}")
    if stats["missing"]:
        print(f"  NOTE: {stats['missing']} small-subset tracks had no mp3 on disk (skipped)")


if __name__ == "__main__":
    main()
