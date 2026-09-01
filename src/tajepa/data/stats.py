"""Dataset-level feature statistics — the one source of truth for codec standardization.

Why this exists: the grounding head (``latent -> codec frame``) is trained against a
*standardized* codec target, so its output lives in whatever space those statistics
define. If training and evaluation disagree about that space, the mismatch is silently
charged to the model. That bug once inverted a headline result (a music-trained model
scored against the *eval* set's statistics looked like it failed to transfer; scored in
its own output space it beat the baseline). So: compute the statistics **once** from the
training cache, store them on the model (see ``JEPA.set_codec_stats``), and let every
consumer read them back off the checkpoint.

Statistics are per-dimension mean/std over frames, pooled across all cache dirs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def codec_stats(
    cache_dir: str | Path | list[str | Path],
    max_clips: int = 300,
    max_frames: int = 512,
    pattern: str = "*.npy",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-dimension ``(mean, std)`` of cached frame features. Shapes ``[D]``.

    ``cache_dir`` may be one directory or several (multi-domain training uses several).
    Sampling mirrors ``EmbeddingSequenceDataset`` exactly: files from every cache are
    pooled and drawn from uniformly, so each domain contributes **in proportion to its
    clip count** — the same mixture the grounding target is standardized against during
    training. Sampling the caches evenly instead would misstate the statistics whenever
    the caches differ in size (FSD50K is ~5x FMA), and taking each cache's *first* N files
    would bias toward whatever sorts first (one genre, one label).
    """
    dirs = [cache_dir] if isinstance(cache_dir, (str, Path)) else list(cache_dir)
    pooled: list[Path] = []
    for d in dirs:
        pooled.extend(sorted(Path(d).rglob(pattern)))
    pooled.sort()
    if not pooled:
        roots = ", ".join(str(d) for d in dirs)
        raise FileNotFoundError(f"No feature files matching {pattern} under: {roots}")
    # even stride across the pooled list: proportional to each cache, unbiased within it
    step = max(1, len(pooled) // max(1, max_clips))
    files = pooled[::step][:max_clips]
    frames = np.concatenate([np.load(f)[:max_frames] for f in files], axis=0)
    x = torch.from_numpy(frames).float()
    return x.mean(0), x.std(0).clamp_min(1e-4)


def dataset_stats(dataset, max_clips: int = 300, max_frames: int = 512):
    """``codec_stats`` for an already-constructed dataset (uses ``[i]['features']``)."""
    feats = [dataset[i]["features"][:max_frames] for i in range(min(max_clips, len(dataset)))]
    x = torch.cat(feats, 0)
    return x.mean(0), x.std(0).clamp_min(1e-4)


def ensure_codec_stats(model, train_cache=None, *, what: str = "model") -> bool:
    """Make sure ``model`` knows the space its grounding head emits into.

    Checkpoints written before the statistics were recorded carry none, and their
    codec-space output is then uninterpretable — silently, and in a direction that
    penalizes the model on transfer sets. Pass the cache the model was *trained* on to
    recover them. Returns True if the model ends up with statistics.
    """
    import warnings

    if getattr(model, "has_codec_stats", False):
        return True   # the checkpoint's own statistics always win
    if train_cache:
        model.set_codec_stats(*codec_stats(train_cache))
        return True
    warnings.warn(
        f"{what} carries no codec statistics (pre-stats checkpoint). Codec-space results "
        "will be biased. Re-train, or pass --train-stats pointing at the cache it was "
        "trained on.", RuntimeWarning, stacklevel=2)
    return False
