"""Download (if needed), extract, and build a manifest for ESC-50.

ESC-50 (Piczak): 2000 clips, 5 s each, 50 environmental classes, 5 official CV
folds. The plan holds it out as an environmental *eval* set, so we record the
class ``label`` and official ``fold`` on every entry and map folds to splits
(1-3 train, 4 val, 5 test) while keeping ``fold`` for proper 5-fold CV later.

    python scripts/prepare_esc50.py            # downloads to data/downloads/ if absent
    python scripts/prepare_esc50.py --zip /path/to/esc50.zip

Then cache embeddings:
    python scripts/extract_embeddings.py \
        --manifest data/manifests/esc50.jsonl \
        --cache data/cache/encodec_24khz/esc50
"""

from __future__ import annotations

import argparse
import csv
import urllib.request
import zipfile
from pathlib import Path

import _bootstrap  # noqa: F401

from tajepa.data.manifest import ManifestEntry, write_manifest

URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
FOLD_SPLIT = {1: "train", 2: "train", 3: "train", 4: "val", 5: "test"}


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using existing archive {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    print(f"Downloading ESC-50 -> {dest} ...")
    urllib.request.urlretrieve(URL, dest)
    return dest


def extract(zip_path: Path, out_dir: Path) -> Path:
    """Extract and return the dataset root (the dir containing audio/ and meta/)."""
    audio = next(out_dir.rglob("meta/esc50.csv"), None)
    if audio is not None:
        print(f"Already extracted at {audio.parent.parent}")
        return audio.parent.parent
    print(f"Extracting {zip_path} -> {out_dir} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    meta = next(out_dir.rglob("meta/esc50.csv"))
    return meta.parent.parent


def build(root: Path, out_manifest: Path) -> int:
    csv_path = root / "meta" / "esc50.csv"
    audio_dir = root / "audio"
    entries: list[ManifestEntry] = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            fold = int(row["fold"])
            wav = (audio_dir / row["filename"]).resolve()
            entries.append(
                ManifestEntry(
                    path=str(wav),
                    domain="environmental",
                    split=FOLD_SPLIT[fold],
                    duration=5.0,
                    sample_rate=44100,
                    clip_id=Path(row["filename"]).stem,
                    label=row["category"],
                    fold=fold,
                )
            )
    write_manifest(entries, out_manifest)
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=Path("data/downloads/esc50.zip"))
    ap.add_argument("--extract-dir", type=Path, default=Path("data/esc50"))
    ap.add_argument("--out", type=Path, default=Path("data/manifests/esc50.jsonl"))
    args = ap.parse_args()

    zip_path = download(args.zip)
    root = extract(zip_path, args.extract_dir)
    n = build(root, args.out)
    splits: dict[str, int] = {}
    labels: set[str] = set()
    import json

    for line in open(args.out):
        e = json.loads(line)
        splits[e["split"]] = splits.get(e["split"], 0) + 1
        labels.add(e["label"])
    print(f"Wrote {n} entries to {args.out}")
    print(f"  splits: {splits}")
    print(f"  classes: {len(labels)}")


if __name__ == "__main__":
    main()
