"""Download (if needed), extract, and build a manifest for FSD50K.

FSD50K (Fonseca et al.): ~51k Freesound clips, 200 general sound-event classes, mono
44.1 kHz, 0.3–30 s. Wired in as a **general-audio pretraining** source — the diverse,
non-music data the plan calls for (a clean stand-in for AudioSet, which is painful to
download). We use the *dev* set (train/val) for pretraining; the eval set is optional.

Two wrinkles vs the other prepares:
- FSD50K's audio is a **multi-part split zip** on Zenodo (``.z01..``+``.zip``); Python
  ``zipfile`` can't read those, so we recombine/extract with ``7z`` (or ``zip -s``).
- It is **multi-label**, so labels are kept only as an informational string — FSD50K is
  for pretraining here, not single-label probing (ESC-50 remains the labelled eval).

    python scripts/prepare_fsd50k.py --download           # ~24 GB dev (+8 GB eval if --include-eval)
    python scripts/prepare_fsd50k.py --root data/fsd50k   # if parts already downloaded

Then (locally or via Modal): extract_embeddings.py over data/manifests/fsd50k.jsonl.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.data.manifest import ManifestEntry, write_manifest

ZENODO = "https://zenodo.org/records/4060432/files"
DEV_PARTS = [f"FSD50K.dev_audio.z0{i}" for i in range(1, 6)] + ["FSD50K.dev_audio.zip"]
EVAL_PARTS = ["FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip"]
GROUND_TRUTH = "FSD50K.ground_truth.zip"


def download(name: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {name} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"  downloading {name} ...")
    urllib.request.urlretrieve(f"{ZENODO}/{name}?download=1", dest)
    return dest


def _which_7z() -> str | None:
    for c in ("7z", "7za", "7zz"):
        if shutil.which(c):
            return c
    return None


def recombine_extract(main_zip: Path, out_dir: Path) -> None:
    """Extract a split zip (``main_zip`` + sibling ``.z01..`` parts) into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seven = _which_7z()
    if seven:
        subprocess.run([seven, "x", "-y", f"-o{out_dir}", str(main_zip)], check=True)
        return
    if shutil.which("zip") and shutil.which("unzip"):
        combined = main_zip.with_name(main_zip.stem + "_single.zip")
        if not combined.exists():
            subprocess.run(["zip", "-q", "-s", "0", str(main_zip), "--out", str(combined)], check=True)
        subprocess.run(["unzip", "-q", "-o", str(combined), "-d", str(out_dir)], check=True)
        return
    raise RuntimeError(
        "Need '7z' (p7zip) or 'zip'+'unzip' to recombine FSD50K split archives; install one."
    )


def find_audio_root(extract_dir: Path, marker_dir: str) -> Path:
    """Return the dir that directly contains the .wav files (handles a nested top dir)."""
    hit = next((p for p in extract_dir.rglob(marker_dir) if p.is_dir()), None)
    if hit is None:
        wav = next(extract_dir.rglob("*.wav"), None)
        if wav is None:
            raise FileNotFoundError(f"No FSD50K audio under {extract_dir}")
        return wav.parent
    return hit


def build(
    gt_csv: Path, audio_root: Path, split: str, entries: list[ManifestEntry]
) -> None:
    """Append manifest entries from one ground-truth csv (dev.csv or eval.csv)."""
    with open(gt_csv) as f:
        for row in csv.DictReader(f):
            fname = row["fname"]
            wav = audio_root / f"{fname}.wav"
            if not wav.exists():
                continue
            # dev.csv has a per-row split (train/val); eval.csv has none -> caller's split.
            row_split = {"train": "train", "val": "val"}.get(row.get("split", ""), split)
            entries.append(
                ManifestEntry(
                    path=str(wav.resolve()),
                    domain="general",
                    split=row_split,
                    clip_id=str(fname),
                    label=(row.get("labels") or None),  # multi-label string, informational
                )
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/fsd50k"))
    ap.add_argument("--out", type=Path, default=Path("data/manifests/fsd50k.jsonl"))
    ap.add_argument("--download", action="store_true", help="Download archives if missing.")
    ap.add_argument("--include-eval", action="store_true",
                    help="Also include the eval set (~8 GB) as the test split.")
    ap.add_argument("--cleanup-archives", action="store_true",
                    help="Delete the downloaded archives after extraction (frees ~24 GB; "
                         "use on ephemeral cloud disks — locally, omit to keep them cached).")
    args = ap.parse_args()

    dl = args.root / "downloads"
    parts = list(DEV_PARTS) + ([*EVAL_PARTS] if args.include_eval else [])
    if args.download:
        print("Downloading FSD50K archives (large) ...")
        download(GROUND_TRUTH, dl)
        for name in parts:
            download(name, dl)

    # Ground truth (normal zip)
    gt_dir = args.root / "ground_truth"
    if next(gt_dir.rglob("dev.csv"), None) is None:
        with zipfile.ZipFile(dl / GROUND_TRUTH) as zf:
            zf.extractall(gt_dir)
    dev_csv = next(gt_dir.rglob("dev.csv"))

    # Dev audio (split zip)
    dev_dir = args.root / "dev_audio"
    if next(dev_dir.rglob("*.wav"), None) is None:
        recombine_extract(dl / "FSD50K.dev_audio.zip", dev_dir)
    dev_audio = find_audio_root(dev_dir, "FSD50K.dev_audio")

    entries: list[ManifestEntry] = []
    build(dev_csv, dev_audio, "train", entries)

    if args.include_eval:
        eval_dir = args.root / "eval_audio"
        if next(eval_dir.rglob("*.wav"), None) is None:
            recombine_extract(dl / "FSD50K.eval_audio.zip", eval_dir)
        eval_audio = find_audio_root(eval_dir, "FSD50K.eval_audio")
        build(next(gt_dir.rglob("eval.csv")), eval_audio, "test", entries)

    # Free the archives (~24 GB) once the WAVs are extracted — peak-disk relief on
    # ephemeral cloud containers. Skipped by default so local re-runs stay cached.
    if args.cleanup_archives and dl.exists():
        print(f"Cleaning up archives in {dl} ...")
        shutil.rmtree(dl, ignore_errors=True)

    write_manifest(entries, args.out)
    splits: dict[str, int] = {}
    for e in entries:
        splits[e.split] = splits.get(e.split, 0) + 1
    print(f"Wrote {len(entries)} entries to {args.out}")
    print(f"  splits: {splits}")


if __name__ == "__main__":
    main()
