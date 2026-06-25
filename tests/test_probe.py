import numpy as np
import torch

from tajepa.data.manifest import ManifestEntry, write_manifest
from tajepa.data.embedding_dataset import ManifestEmbeddingDataset
from tajepa.eval import IdentityRepresentation, run_linear_probe
from tajepa.eval.probe import pool_time


def test_pool_time_shapes():
    seq = torch.randn(3, 10, 8)
    assert pool_time(seq, "mean").shape == (3, 8)
    assert pool_time(seq, "meanstd").shape == (3, 16)


def test_probe_learns_separable_classes(tmp_path):
    # Two classes with clearly different feature means -> probe should reach high
    # accuracy on held-out clips, exercising the full extract->fit->score path.
    rng = np.random.default_rng(0)
    cache = tmp_path / "cache"
    cache.mkdir()
    entries = []
    for cls, mean in enumerate([-2.0, 2.0]):
        for i in range(20):
            clip = f"c{cls}_{i}"
            arr = (rng.standard_normal((30, 6)).astype(np.float32) * 0.3) + mean
            np.save(cache / f"{clip}.npy", arr)
            split = "test" if i >= 14 else "train"
            entries.append(
                ManifestEntry(path=f"/x/{clip}.wav", clip_id=clip,
                              label=f"class{cls}", split=split)
            )
    manifest = tmp_path / "m.jsonl"
    write_manifest(entries, manifest)

    train = ManifestEmbeddingDataset(manifest, cache, split="train")
    test = ManifestEmbeddingDataset(manifest, cache, split="test")
    rep = IdentityRepresentation(train[0]["features"].shape[-1])
    res = run_linear_probe(train, test, rep, pool="meanstd", epochs=200, device="cpu")

    assert res.num_classes == 2
    assert res.test_acc > 0.9
