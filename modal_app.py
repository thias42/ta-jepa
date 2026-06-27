"""Modal app — run ta-jepa extraction / training / eval on serverless GPUs, with
Cloudflare R2 for storage.

Why this shape (see the cloud discussion in the project history):
- We train on **cached codec embeddings** (~10–15 GB), not raw audio, so the training
  loop is GPU-bound on a tiny dataset — cheap. Audio only matters during a one-time
  extraction, which we run *on Modal from the public source* so you upload nothing.
- **R2** stores the reusable artifacts (embedding caches + checkpoints) as **tarballs**
  (one big transfer beats thousands of small-file GETs; R2 has no egress fees).
- Each Modal function just syncs from R2 to local SSD and shells out to the existing,
  tested scripts — so the cloud path runs exactly the same code as local.

Setup (one time) — see docs/cloud-modal.md:
  1. Create an R2 bucket and an API token (Access Key ID + Secret).
  2. modal secret create r2-credentials \
        R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com \
        R2_BUCKET=ta-jepa R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...
  3. pip install modal && modal setup

Usage:
  modal run modal_app.py::extract  --dataset fma_small --frontend encodec_24khz
  modal run modal_app.py::extract  --dataset esc50     --frontend encodec_24khz
  modal run modal_app.py::train    --model jepa --dataset fma_small \
        --save-name jepa_fma --extra-args "--dim 256 --enc-depth 6 --offsets 1 2 4 8 --max-steps 25000"
  modal run modal_app.py::evaluate --eval-kind forecast --dataset esc50 \
        --jepa-ckpt jepa_fma --apc-ckpt apc_fma
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path

import modal

# --- GPUs per phase (edit to taste; T4/L4 are cheapest, A10G a good train default) ---
EXTRACT_GPU = "T4"
TRAIN_GPU = "A10G"
EVAL_GPU = "T4"

# Ephemeral scratch disk (MiB). We only need ~50 GB (FSD50K unpack) / ~50 GB (multi-domain
# train), but Modal's ephemeral_disk floor is 512 GiB (range 524288–3145728 MiB), so use
# the minimum — plenty of headroom.
DISK_MIB = 524288  # 512 GiB (Modal minimum)

REPO = "/root/ta-jepa"
SCRATCH = "/scratch"

app = modal.App("ta-jepa")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "p7zip-full")  # p7zip: FSD50K split-zip archives
    .pip_install(
        "torch>=2.2", "torchaudio>=2.2", "transformers>=4.40", "pytorch-lightning>=2.2",
        "numpy>=1.24", "pyyaml>=6.0", "soundfile>=0.12", "librosa>=0.10", "einops>=0.7",
        "tqdm>=4.66", "tensorboard>=2.15", "boto3>=1.34",
    )
    .env({"PYTHONPATH": f"{REPO}/src", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_dir("src", f"{REPO}/src")
    .add_local_dir("scripts", f"{REPO}/scripts")
    .add_local_dir("configs", f"{REPO}/configs")
)

# R2 credentials are injected as env vars by this Modal Secret.
r2_secret = modal.Secret.from_name("r2-credentials")

# Datasets we can (re)build from their public source on Modal. Commands run with
# cwd=SCRATCH, so the scripts' default data/ paths land under /scratch/data.
PREPARE = {
    "fma_small": ["python", f"{REPO}/scripts/prepare_fma.py", "--download"],
    "esc50": ["python", f"{REPO}/scripts/prepare_esc50.py"],
    "fsd50k": ["python", f"{REPO}/scripts/prepare_fsd50k.py", "--download", "--cleanup-archives"],
}
# (frontend -> (extract script, extra args)); mel uses the config for hop/n_mels.
EXTRACTORS = {
    "encodec_24khz": ([f"{REPO}/scripts/extract_embeddings.py"], ["--device", "cuda"]),
    "logmel": ([f"{REPO}/scripts/extract_mel.py"], ["--config", f"{REPO}/configs/mel_baseline.yaml"]),
    "descriptors": ([f"{REPO}/scripts/extract_descriptors.py"], []),  # Phase 2a control signals
}


# --------------------------------------------------------------------------- #
# R2 helpers (run inside the container; boto3 + creds from the secret)
# --------------------------------------------------------------------------- #
def _r2():
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )


def _bucket() -> str:
    return os.environ.get("R2_BUCKET", "ta-jepa")


def _upload(local: str, key: str) -> None:
    print(f"R2 upload  {local} -> s3://{_bucket()}/{key}")
    _r2().upload_file(local, _bucket(), key)


def _download(key: str, local: str) -> None:
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    print(f"R2 download s3://{_bucket()}/{key} -> {local}")
    _r2().download_file(_bucket(), key, local)


def _upload_dir_tar(local_dir: str, key: str) -> None:
    """Tar ``local_dir`` (arcname = its basename) and upload to ``key``."""
    with tempfile.NamedTemporaryFile(suffix=".tar") as tmp:
        with tarfile.open(tmp.name, "w") as tar:
            tar.add(local_dir, arcname=Path(local_dir).name)
        _upload(tmp.name, key)


def _download_tar(key: str, dest_parent: str) -> str:
    """Download tar ``key`` and extract under ``dest_parent``; return the extracted dir."""
    Path(dest_parent).mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar") as tmp:
        _download(key, tmp.name)
        with tarfile.open(tmp.name, "r") as tar:
            names = tar.getnames()
            tar.extractall(dest_parent)
    top = names[0].split("/")[0] if names else ""
    return str(Path(dest_parent) / top)


def _run(cmd: list[str], cwd: str = SCRATCH) -> None:
    Path(cwd).mkdir(parents=True, exist_ok=True)  # Modal containers have no /scratch by default
    print("RUN:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


# --------------------------------------------------------------------------- #
# Functions
# --------------------------------------------------------------------------- #
@app.function(image=image, secrets=[r2_secret], gpu=EXTRACT_GPU,
              ephemeral_disk=DISK_MIB, timeout=4 * 3600)
def extract(dataset: str, frontend: str = "encodec_24khz") -> None:
    """Build a dataset from its public source and cache its features to R2.

    Uploads ``cache/<frontend>/<dataset>.tar`` and ``manifests/<dataset>.jsonl``.
    """
    if dataset not in PREPARE:
        raise ValueError(f"Unknown dataset '{dataset}'. Known: {sorted(PREPARE)}")
    if frontend not in EXTRACTORS:
        raise ValueError(f"Unknown frontend '{frontend}'. Known: {sorted(EXTRACTORS)}")

    _run(PREPARE[dataset])
    manifest = f"{SCRATCH}/data/manifests/{dataset}.jsonl"
    cache_dir = f"{SCRATCH}/data/cache/{frontend}/{dataset}"
    script, extra = EXTRACTORS[frontend]
    _run(["python", *script, "--manifest", manifest, "--cache", cache_dir, *extra])

    _upload_dir_tar(cache_dir, f"cache/{frontend}/{dataset}.tar")
    _upload(manifest, f"manifests/{dataset}.jsonl")
    print(f"DONE extract: {dataset} [{frontend}] -> R2")


@app.function(image=image, secrets=[r2_secret], gpu=TRAIN_GPU,
              ephemeral_disk=DISK_MIB, timeout=8 * 3600)
def train(
    model: str, dataset: str, frontend: str = "encodec_24khz",
    save_name: str = "", extra_args: str = "",
) -> None:
    """Train ``{jepa,apc,ajepa}`` on cached dataset(s) from R2; upload the checkpoint.

    ``dataset`` may be comma-separated (e.g. ``fma_small,fsd50k``) for multi-domain
    pretraining — each cache is pulled and passed as a separate ``--cache`` dir.
    """
    if model not in ("jepa", "apc", "ajepa", "actions"):
        raise ValueError(f"model must be jepa|apc|ajepa|actions, got {model}")
    datasets = [d.strip() for d in dataset.split(",") if d.strip()]
    save_name = save_name or f"{model}_{'_'.join(datasets)}"
    cache_dirs = [
        _download_tar(f"cache/{frontend}/{d}.tar", f"{SCRATCH}/cache/{frontend}") for d in datasets
    ]
    ckpt = f"{SCRATCH}/runs/{save_name}.ckpt"
    cmd = [
        "python", f"{REPO}/scripts/train_{model}.py",
        "--cache", *cache_dirs, "--accelerator", "gpu", "--save", ckpt,
        *shlex.split(extra_args),
    ]
    _run(cmd)
    _upload(ckpt, f"runs/{save_name}.ckpt")
    print(f"DONE train: {save_name} -> R2 runs/{save_name}.ckpt")


@app.function(image=image, secrets=[r2_secret], gpu=TRAIN_GPU,
              ephemeral_disk=DISK_MIB, timeout=8 * 3600)
def train_control(dataset: str, save_name: str = "", extra_args: str = "") -> None:
    """Phase 2a: train the controllable JEPA on codec + descriptor caches from R2.

    Needs both ``cache/encodec_24khz/<d>.tar`` and ``cache/descriptors/<d>.tar`` for each
    dataset (run ``extract`` with ``--frontend encodec_24khz`` and ``--frontend
    descriptors`` first). ``dataset`` may be comma-separated for multi-domain.
    """
    datasets = [d.strip() for d in dataset.split(",") if d.strip()]
    save_name = save_name or f"control_{'_'.join(datasets)}"
    feat_dirs = [_download_tar(f"cache/encodec_24khz/{d}.tar", f"{SCRATCH}/cache/encodec_24khz")
                 for d in datasets]
    ctrl_dirs = [_download_tar(f"cache/descriptors/{d}.tar", f"{SCRATCH}/cache/descriptors")
                 for d in datasets]
    ckpt = f"{SCRATCH}/runs/{save_name}.ckpt"
    cmd = ["python", f"{REPO}/scripts/train_control.py",
           "--features", *feat_dirs, "--control", *ctrl_dirs,
           "--accelerator", "gpu", "--save", ckpt, *shlex.split(extra_args)]
    _run(cmd)
    _upload(ckpt, f"runs/{save_name}.ckpt")
    print(f"DONE train_control: {save_name} -> R2 runs/{save_name}.ckpt")


@app.function(image=image, secrets=[r2_secret], gpu=EVAL_GPU, timeout=2 * 3600)
def control_eval(dataset: str, ckpt: str, extra_args: str = "") -> None:
    """Closed-loop controllability eval: render + re-measure (prints the matrix)."""
    feat = _download_tar(f"cache/encodec_24khz/{dataset}.tar", f"{SCRATCH}/cache/encodec_24khz")
    ctrl = _download_tar(f"cache/descriptors/{dataset}.tar", f"{SCRATCH}/cache/descriptors")
    local_ckpt = f"{SCRATCH}/runs/{ckpt}.ckpt"
    _download(f"runs/{ckpt}.ckpt", local_ckpt)
    cmd = ["python", f"{REPO}/scripts/run_controllability.py",
           "--ckpt", local_ckpt, "--features", feat, "--control", ctrl, *shlex.split(extra_args)]
    _run(cmd)


@app.function(image=image, secrets=[r2_secret], gpu=EVAL_GPU, timeout=2 * 3600)
def evaluate(
    eval_kind: str, dataset: str, frontend: str = "encodec_24khz",
    jepa_ckpt: str = "", apc_ckpt: str = "", extra_args: str = "",
) -> None:
    """Run ``forecast`` or ``probe`` on a cached dataset + checkpoint(s) from R2."""
    cache_dir = _download_tar(f"cache/{frontend}/{dataset}.tar", f"{SCRATCH}/cache/{frontend}")
    manifest = f"{SCRATCH}/manifests/{dataset}.jsonl"
    _download(f"manifests/{dataset}.jsonl", manifest)

    ckpt_args = []
    for name, kind in ((jepa_ckpt, "jepa"), (apc_ckpt, "apc")):
        if name:
            local = f"{SCRATCH}/runs/{name}.ckpt"
            _download(f"runs/{name}.ckpt", local)
            ckpt_args += [f"--{kind}-ckpt", local]

    if eval_kind == "forecast":
        cmd = ["python", f"{REPO}/scripts/run_forecast.py",
               "--manifest", manifest, "--cache", cache_dir, *ckpt_args, *shlex.split(extra_args)]
    elif eval_kind == "probe":
        cmd = ["python", f"{REPO}/scripts/run_probe.py",
               "--manifest", manifest, "--cache", cache_dir, *ckpt_args, *shlex.split(extra_args)]
    else:
        raise ValueError("eval_kind must be 'forecast' or 'probe'")
    _run(cmd)
