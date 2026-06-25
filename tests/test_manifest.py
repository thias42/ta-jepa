import numpy as np
import soundfile as sf

from tajepa.data.manifest import build_manifest, read_manifest, write_manifest


def test_build_read_roundtrip(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for i in range(3):
        sf.write(audio_dir / f"clip_{i}.wav", np.zeros(2400, dtype=np.float32), 24000)
    (audio_dir / "notes.txt").write_text("ignore me")

    entries = build_manifest([audio_dir], domain="music")
    assert len(entries) == 3
    assert all(e.domain == "music" for e in entries)
    assert all(e.duration is not None for e in entries)  # probed

    out = tmp_path / "m.jsonl"
    write_manifest(entries, out)
    loaded = read_manifest(out)
    assert [e.clip_id for e in loaded] == [e.clip_id for e in entries]
