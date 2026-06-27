# Cloud runs: Modal + Cloudflare R2

Run extraction / training / eval on serverless GPUs (Modal), with embedding caches and
checkpoints stored in Cloudflare R2. Defined in [`modal_app.py`](../modal_app.py).

## Why this setup

We train on **cached codec embeddings** (~10–15 GB), not raw audio, so the training loop is
GPU-bound on a tiny dataset — cheap (cents to a couple dollars per run). Raw audio only
matters during a one-time extraction, which Modal does **from the public source**, so you
upload nothing. R2 holds the reusable artifacts (caches + checkpoints) as tarballs (one big
transfer beats thousands of small-file GETs; R2 has **no egress fees**). Modal bills
per-second with no idle cost.

Modal can't store cheaply long-term and Cloudflare can't train — so: **Modal = compute,
R2 = storage.**

## One-time setup

1. **Cloudflare R2**: create a bucket (e.g. `ta-jepa`) and an R2 **API token** → note the
   Access Key ID, Secret Access Key, and your account's S3 endpoint
   `https://<accountid>.r2.cloudflarestorage.com`.

2. **Modal**:
   ```bash
   pip install modal
   modal setup                      # auth this machine
   modal secret create r2-credentials \
       R2_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com \
       R2_BUCKET=ta-jepa \
       R2_ACCESS_KEY_ID=<key-id> \
       R2_SECRET_ACCESS_KEY=<secret>
   ```

## Workflow

```bash
# 1) Build caches from public sources (uploads cache tar + manifest to R2). One-time per set.
modal run modal_app.py::extract --dataset fma_small --frontend encodec_24khz
modal run modal_app.py::extract --dataset esc50     --frontend encodec_24khz
# (logmel frontend too, if training the A-JEPA baseline:)
modal run modal_app.py::extract --dataset fma_small --frontend logmel

# 2) Train (downloads cache from R2, trains on GPU, uploads checkpoint to R2).
modal run modal_app.py::train --model jepa --dataset fma_small --save-name jepa_fma \
    --extra-args "--dim 256 --enc-depth 6 --pred-depth 3 --offsets 1 2 4 8 \
                  --grounding-coef 1.0 --max-steps 25000 --batch-size 32"
modal run modal_app.py::train --model apc --dataset fma_small --save-name apc_fma \
    --extra-args "--offsets 1 3 5 --hidden 512 --layers 3 --max-steps 2500"

# 3) Evaluate (downloads cache + checkpoints from R2; prints results in the logs).
modal run modal_app.py::evaluate --eval-kind forecast --dataset esc50 \
    --jepa-ckpt jepa_fma --apc-ckpt apc_fma --extra-args "--max-clips 300"
modal run modal_app.py::evaluate --eval-kind probe --dataset esc50 \
    --jepa-ckpt jepa_fma --extra-args "--representation jepa --cv"
```

## R2 layout

```
manifests/<dataset>.jsonl              # clip_id / label / fold / split (audio paths are stale, unused by eval)
cache/<frontend>/<dataset>.tar         # the [T,D] .npy embedding cache + meta.yaml
runs/<name>.ckpt                       # Lightning checkpoints
```

## Cost & knobs

- GPUs per phase are set at the top of `modal_app.py` (`EXTRACT_GPU`, `TRAIN_GPU`,
  `EVAL_GPU`). T4/L4 are cheapest; A10G is a good train default. Bigger = faster = the
  same total $ for compute-bound jobs, so pick by wall-clock preference.
- A 25k-step JEPA run is ~30–60 min on A10G ≈ a dollar or two. Extraction of a new set is
  ~1–2 GPU-hours. R2 storage of ~30 GB ≈ $0.45/month.

## Broadening the training data: FSD50K (general audio)

`fsd50k` is wired in (`scripts/prepare_fsd50k.py`, in the `PREPARE` registry) — ~41k dev
clips of 200 general sound-event classes, the diverse non-music data the plan wants (a clean
AudioSet stand-in). The image includes `p7zip` for its multi-part split archives.

```bash
modal run modal_app.py::extract --dataset fsd50k --frontend encodec_24khz   # ~24 GB download on Modal
```

**Multi-domain training** (FMA music + FSD50K general) is supported directly — pass a
comma-separated `--dataset`; each cache is pulled from R2 and handed to training as a
separate `--cache` dir:

```bash
modal run modal_app.py::extract --dataset fsd50k                       # one-time, ~24 GB on Modal
modal run modal_app.py::train --model jepa --dataset fma_small,fsd50k \
    --save-name jepa_multi --extra-args "--dim 256 --enc-depth 6 --offsets 1 2 4 8 \
    --grounding-coef 1.0 --max-steps 25000"
modal run modal_app.py::evaluate --eval-kind forecast --dataset esc50 \
    --jepa-ckpt jepa_multi --apc-ckpt apc_fma --extra-args "--max-clips 300"
```

Then compare the ESC-50 forecasting curve to the FMA-only run — that's the test of whether
broadening the data closes the transfer gap. (Locally: `train_*.py --cache dirA dirB ...`.)

## Phase 2a: controllable JEPA (supervised descriptor control)

Cache descriptors (the control signals) alongside the codec embeddings, train the
controllable JEPA, then run the closed-loop controllability eval.

```bash
# 1) Descriptor caches (control signals) for train sets + the eval set:
modal run modal_app.py::extract --dataset fma_small --frontend descriptors
modal run modal_app.py::extract --dataset fsd50k    --frontend descriptors
modal run modal_app.py::extract --dataset esc50     --frontend descriptors

# 2) Train the controllable JEPA (multi-domain; needs codec + descriptor caches for each):
modal run --detach modal_app.py::train_control --dataset fma_small,fsd50k \
    --save-name control_multi --extra-args "--dim 256 --enc-depth 6 --offsets 1 2 4 8 \
    --grounding-coef 1.0 --max-steps 25000"

# 3) Closed-loop controllability eval on ESC-50 (renders + re-measures; prints the matrix):
modal run modal_app.py::control_eval --dataset esc50 --ckpt control_multi \
    --extra-args "--offset 1 --bump 2.0 --n-clips 100"
```

The eval prints the controllability matrix `M[perturbed, measured]` and the disentanglement
summary — diagonal positive (each control raises its own descriptor) and dominance ratio
(how cleanly separated the controls are). A well-trained model should be diagonally dominant.

## Phase 2b: learned latent actions (VQ)

No descriptors needed — the action vocabulary is learned. Just codec caches.

```bash
modal run --detach modal_app.py::train --model actions --dataset fma_small,fsd50k \
    --save-name actions_multi --extra-args "--dim 256 --enc-depth 6 --num-codes 16 \
    --code-dim 32 --grounding-coef 1.0 --max-steps 25000"
```

Watch `diag/perplexity` in the logs — it should sit well above 1 (codes in use) but not
imply trivial prediction (leakage). Sweep `--num-codes`. Then the actions-controllability eval
(forces each code, renders, re-extracts MIR; reports per-code effect + consistency /
separability / usage):

```bash
modal run modal_app.py::action_eval --dataset esc50 --ckpt actions_multi --extra-args "--n-clips 40"
```

Read `separability` (>1 = codes do distinct things) and `mean consistency` (→1 = each code
means the same thing across contexts). The smoke model was 0.65 / 0.42 (under-trained).

## Adding another dataset

1. Write `scripts/prepare_<name>.py` → `data/manifests/<name>.jsonl` (mirror the existing
   prepares).
2. Add it to the `PREPARE` registry in `modal_app.py`.
3. `modal run modal_app.py::extract --dataset <name>` and train as above.
