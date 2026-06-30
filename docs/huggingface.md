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

- **Examples:** drop a few short audio clips into `space/anticipation/examples/` before
  step 2 to get one-click example clips (none bundled by default — mind clip licensing).
- **Config:** the Space reads `MODEL_REPO` / `CKPT_FILE` Space variables (Settings →
  Variables) if you host the checkpoint elsewhere.
- **GitHub must be public** for the Space's `pip install … git+https://github.com/...` to
  resolve.
