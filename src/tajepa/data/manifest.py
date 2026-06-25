"""Audio manifests.

A manifest is a JSONL file, one entry per audio clip, recording its path, domain
tag (music / environmental / speech / ...), split, and — when cheaply available —
duration and sample rate. Everything downstream (embedding extraction, datasets)
consumes manifests rather than walking directories, so the multi-domain mix
(AudioSet + FMA/MTG-Jamendo + ESC-50/UrbanSound) is described in one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass
class ManifestEntry:
    path: str
    domain: str = "unknown"          # music | environmental | speech | unknown
    split: str = "train"             # train | val | test
    duration: float | None = None    # seconds, if known
    sample_rate: int | None = None
    clip_id: str | None = None
    label: str | None = None         # class label, for probe/eval datasets
    fold: int | None = None          # CV fold, for datasets with an official protocol

    def __post_init__(self) -> None:
        if self.clip_id is None:
            self.clip_id = Path(self.path).stem


def _probe(path: Path) -> tuple[float | None, int | None]:
    """Best-effort (duration_seconds, sample_rate) without decoding the whole file."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / info.samplerate, int(info.samplerate)
    except Exception:
        return None, None


def build_manifest(
    roots: Iterable[str | Path],
    domain: str = "unknown",
    split: str = "train",
    probe: bool = True,
    recursive: bool = True,
) -> list[ManifestEntry]:
    """Scan one or more directories for audio files and build manifest entries."""
    entries: list[ManifestEntry] = []
    for root in roots:
        root = Path(root)
        if root.is_file():
            paths = [root]
        else:
            globber = root.rglob("*") if recursive else root.glob("*")
            paths = sorted(p for p in globber if p.suffix.lower() in AUDIO_EXTENSIONS)
        for p in paths:
            dur, sr = _probe(p) if probe else (None, None)
            entries.append(
                ManifestEntry(
                    path=str(p.resolve()),
                    domain=domain,
                    split=split,
                    duration=dur,
                    sample_rate=sr,
                )
            )
    return entries


def write_manifest(entries: Iterable[ManifestEntry], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(asdict(e)) + "\n")


def read_manifest(path: str | Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(ManifestEntry(**json.loads(line)))
    return entries


def iter_manifest(path: str | Path) -> Iterator[ManifestEntry]:
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield ManifestEntry(**json.loads(line))
