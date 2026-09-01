# Hugging Face: model + anticipation demo Space

Publishes two things:

1. a **model repo** `Maeich/ta-jepa-anticipation` holding `jepa_fma_grounded.ckpt`, and
2. a **Gradio Space** `Maeich/ta-jepa-anticipation-demo` running the anticipation demo,
   which installs `tajepa` from GitHub and downloads the checkpoint from (1) at startup.

Space files live in [`space/anticipation/`](../space/anticipation); the model card to upload
is [`hf-model-card.md`](hf-model-card.md). The HF namespace is `Maeich`; the GitHub repo
(used by the Space's pip install) is `thias42/ta-jepa` — these are intentionally different.

## 0. Authenticate (interactive — run it yourself)

```bash
hf auth login        # or: export HF_TOKEN=hf_xxx   (token needs write access)
```

## 1. Push the model

Robust, version-stable via the `huggingface_hub` Python API:

```bash
P=$(conda run -n ta-jepa which python)
$P - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo = "Maeich/ta-jepa-anticipation"
api.create_repo(repo, repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="runs/jepa_fma_grounded.ckpt",
                path_in_repo="jepa_fma_grounded.ckpt", repo_id=repo, repo_type="model")
api.upload_file(path_or_fileobj="docs/hf-model-card.md",
                path_in_repo="README.md", repo_id=repo, repo_type="model")
print("model pushed:", repo)
PY
```

(The 103 MB checkpoint is stored via LFS/Xet automatically.)

## 2. Push the Space

```bash
P=$(conda run -n ta-jepa which python)
$P - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
space = "Maeich/ta-jepa-anticipation-demo"
api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(folder_path="space/anticipation", repo_id=space, repo_type="space")
print("space pushed:", space)
PY
```

The Space builds (installs `tajepa[demo]` from GitHub — a few minutes), then launches. On the
free CPU tier, EnCodec encode + the small JEPA run comfortably; first request downloads the
EnCodec weights.

CLI equivalents exist (`hf repo create … --repo-type space --space-sdk gradio`,
`hf upload …`); the Python API above avoids CLI-flag drift across `hf` versions.

## Notes

- **Examples:** eight clips are bundled in `space/anticipation/examples/` — ESC-50 sources
  that are individually **CC0**, so the Space carries no attribution or non-commercial
  obligation; credits are in `space/anticipation/ATTRIBUTION.md` anyway. Add your own by
  dropping files in the same folder (mind their licensing).
- **The AR reference:** `space/anticipation/assets/ar_latent_fma.pt` is a linear AR(4)
  fitted offline in the checkpoint's latent space on its training data (FMA). The demo
  scores against it because persistence is a weak bar; shipping it pre-fitted keeps the
  number reproducible. Regenerate with `tajepa.eval.fit_latent_ar` + `LinearAR.save` if you
  change checkpoints — an AR fitted for one checkpoint is meaningless for another.
- **The Space pins an exact commit.** `space/anticipation/requirements.txt` installs
  `tajepa[demo] @ git+https://github.com/thias42/ta-jepa.git@<sha>`, not a branch. So:
  push GitHub first, then **bump that SHA** to whatever you just pushed, then upload the
  Space. Forgetting the bump means the Space keeps running the old package — which is the
  safe failure (stale but working), whereas the old unpinned form could silently reuse a
  cached pip layer *or* pick up an unrelated main. The SHA edit is also what busts HF's
  build cache, since the layer key is this file's content.
- **Check the startup log.** The demo prints which AR reference it loaded:
  `[ta-jepa] AR reference: linear AR(4) loaded from …`. If it prints `AR reference: NONE`,
  `assets/ar_latent_fma.pt` did not ship and the Space is reporting only the weak
  persistence baseline.
- **Config:** the Space reads `MODEL_REPO` / `CKPT_FILE` Space variables (Settings →
  Variables) if you host the checkpoint elsewhere.
- **GitHub must be public** for the Space's `pip install … git+https://github.com/...` to
  resolve.
