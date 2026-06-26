import numpy as np

from tajepa.data.manifest import ManifestEntry, write_manifest
from tajepa.data.embedding_dataset import ManifestEmbeddingDataset, EmbeddingSequenceDataset


def test_embedding_sequence_single_and_multi_dir(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d, n in ((a, 3), (b, 2)):
        d.mkdir()
        for i in range(n):
            np.save(d / f"{d.name}_{i}.npy", np.random.randn(300, 16).astype(np.float32))

    single = EmbeddingSequenceDataset(a, window_frames=128)
    assert len(single) == 3
    assert single[0]["features"].shape == (128, 16)         # cropped to window

    multi = EmbeddingSequenceDataset([a, b], window_frames=128)
    assert len(multi) == 5                                   # union of both caches
    ids = {multi[i]["clip_id"] for i in range(len(multi))}
    assert ids == {"a_0", "a_1", "a_2", "b_0", "b_1"}


def test_manifest_embedding_join(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    entries = []
    for i in range(6):
        clip = f"clip_{i}"
        np.save(cache / f"{clip}.npy", np.random.randn(50, 8).astype(np.float32))
        entries.append(
            ManifestEntry(
                path=f"/nonexistent/{clip}.wav",
                domain="environmental",
                split="test" if i % 2 else "train",
                clip_id=clip,
                label="dog" if i < 3 else "rain",
                fold=(i % 5) + 1,
            )
        )
    manifest = tmp_path / "m.jsonl"
    write_manifest(entries, manifest)

    ds = ManifestEmbeddingDataset(manifest, cache)
    assert len(ds) == 6
    assert ds.num_classes == 2
    item = ds[0]
    assert item["features"].shape == (50, 8)
    assert item["label"] in {"dog", "rain"}
    assert item["label_idx"] in {0, 1}

    train = ManifestEmbeddingDataset(manifest, cache, split="train")
    assert all(e.split == "train" for e in train.entries)
    assert len(train) == 3
