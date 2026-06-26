"""Tests for the FSD50K manifest builder (CSV parsing + dev/eval split handling).

Loads scripts/prepare_fsd50k.py as a module; the download/extract paths need the real
51k-clip archives, so only the pure parsing logic is exercised here.
"""

import csv
import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))

spec = importlib.util.spec_from_file_location("prepare_fsd50k", _SCRIPTS / "prepare_fsd50k.py")
fsd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fsd)


def test_build_dev_split_and_multilabel(tmp_path):
    audio = tmp_path / "dev_audio"
    audio.mkdir()
    for fid in ("12345", "67890", "11111"):
        (audio / f"{fid}.wav").write_bytes(b"\x00")
    csv_path = tmp_path / "dev.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname", "labels", "mids", "split"])
        w.writerow(["12345", "Dog,Bark", "/m/0bt9lr,/m/05tny_", "train"])
        w.writerow(["67890", "Piano", "/m/05r5c", "val"])
        w.writerow(["99999", "Speech", "/m/09x0r", "train"])  # no wav on disk -> skipped

    entries = []
    fsd.build(csv_path, audio, "train", entries)
    by_id = {e.clip_id: e for e in entries}

    assert set(by_id) == {"12345", "67890"}          # 99999 skipped (missing wav)
    assert by_id["12345"].domain == "general"
    assert by_id["12345"].split == "train"
    assert by_id["12345"].label == "Dog,Bark"        # multi-label kept as string
    assert by_id["67890"].split == "val"             # per-row split honored


def test_build_eval_uses_caller_split(tmp_path):
    audio = tmp_path / "eval_audio"
    audio.mkdir()
    (audio / "42.wav").write_bytes(b"\x00")
    csv_path = tmp_path / "eval.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fname", "labels", "mids"])       # eval.csv has no split column
        w.writerow(["42", "Rain", "/m/06mb1"])

    entries = []
    fsd.build(csv_path, audio, "test", entries)
    assert len(entries) == 1 and entries[0].split == "test"
