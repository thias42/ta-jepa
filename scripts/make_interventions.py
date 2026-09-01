"""Phase 2 — build the paired intervention cache (the action-conditioned training data).

For each clip: sample an intervention, render the intervened signal, and encode **both** the
clean and the intervened audio with the same frozen codec. Cache the pair plus the per-frame
action.

Why pairs. Every Phase 2 result so far measured control by re-extracting a descriptor from
rendered audio, which is why *codec-recoverability of the descriptor* became the selection
criterion (harmonic_ratio works, onset is dead). With a known applied intervention the
comparison is against the **true intervened audio** instead, so that criterion retires — which
matters immediately for reverb, since RT60 is a windowed rather than per-frame quantity and
would have looked undiagnosable under the old eval.

Both sides are encoded in the same pass rather than reusing an existing clean cache, so the
two are frame-aligned by construction and cannot drift if the frontend config differs.

    python scripts/make_interventions.py --manifest data/manifests/fma_small.jsonl \
        --out data/cache/interventions/fma_small --max-clips 500 --seconds 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import torch

from tajepa.codec.frontend import build_frontend
from tajepa.config import CodecConfig, resolve_device
from tajepa.data.io import load_resampled
from tajepa.data.manifest import read_manifest
from tajepa.interventions import (
    AXES, action_frames, apply_intervention, sample_spec, transient_score,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--seconds", type=float, default=10.0, help="Trim each clip (0 = full).")
    ap.add_argument("--p-axis", type=float, default=0.6,
                    help="Per-axis firing probability. Axes are drawn independently so the "
                         "set contains combinations — a model cannot pass by learning three "
                         "disjoint detectors.")
    ap.add_argument("--action-lead", type=float, default=0.12,
                    help="Seconds between commanding an action and its first acoustic "
                         "effect. With 0 the action arrives simultaneously with its "
                         "consequence and carries almost no information — the first trained "
                         "model ignored it entirely. Training and eval must use the matching "
                         "--action-lead-frames.")
    ap.add_argument("--reverb-min-transient", type=float, default=0.05,
                    help="Skip the reverb axis on clips flatter than this. A room is heard "
                         "through its response to transients; on stationary texture a reverb "
                         "change is near-invisible (applied RT60 recoverable at R^2 +0.22 on "
                         "transient-rich clips vs -2.0 on stationary ones), and commanding an "
                         "action with no observable consequence teaches the model to ignore "
                         "that axis. 0 disables the gate.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or resolve_device("auto")
    codec = build_frontend(CodecConfig(device=device))
    sr, fps = int(codec.sample_rate), float(codec.frame_rate)
    entries = [e for e in read_manifest(args.manifest)
               if args.split is None or e.split == args.split]
    if args.max_clips:
        entries = entries[: args.max_clips]
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta_path, n_ok, n_fail = args.out / "meta.jsonl", 0, 0
    with open(meta_path, "w") as meta:
        for i, e in enumerate(entries):
            try:
                wav = load_resampled(e.path, sr, mono=True)[0].numpy()
                if args.seconds:
                    wav = wav[: int(args.seconds * sr)]
                if wav.size < sr:                       # too short to host an event
                    raise ValueError(f"clip shorter than 1s ({wav.size} samples)")
                spec = sample_spec(rng, wav.size / sr, p_axis=args.p_axis,
                                   transient=transient_score(wav, sr),
                                   reverb_min_transient=args.reverb_min_transient,
                                   lead_s=args.action_lead)
                inter = apply_intervention(wav, sr, spec, rng)

                with torch.no_grad():
                    stack = torch.from_numpy(np.stack([wav, inter])).float().to(device)
                    emb = codec.encode(stack).cpu().numpy()   # [2, T, D]
                act = action_frames(spec, emb.shape[1], fps)

                # uncompressed on purpose: float32 barely compresses (19% saved) but
                # decompression cost 6x the load time, which serialised against the GPU
                np.savez(args.out / f"{e.clip_id}.npz",
                                    clean=emb[0].astype(np.float32),
                                    intervened=emb[1].astype(np.float32),
                                    action=act)
                meta.write(json.dumps({"clip_id": e.clip_id, "domain": e.domain,
                                       "split": e.split, "frames": int(emb.shape[1]),
                                       "transient": transient_score(wav, sr),
                                       "axes": list(AXES), **spec.__dict__,
                                       "deltas": spec.deltas}) + "\n")
                n_ok += 1
            except Exception as exc:                    # a few FMA mp3s are corrupt
                n_fail += 1
                print(f"  [skip] {e.clip_id}: {type(exc).__name__}: {exc}")
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(entries)}  ok={n_ok} failed={n_fail}", flush=True)

    print(f"\nWrote {n_ok} pairs to {args.out} ({n_fail} skipped)")
    print(f"Each .npz: clean [T,D], intervened [T,D], action [T,{len(AXES)}] over {AXES}")
    print(f"Action lead: {args.action_lead:.3f}s "
          f"(~{round(args.action_lead * fps)} frames) — train and eval with "
          f"--action-lead-frames {round(args.action_lead * fps)}")


if __name__ == "__main__":
    main()
