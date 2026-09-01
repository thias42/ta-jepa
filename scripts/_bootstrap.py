"""Make ``tajepa`` importable when running scripts without ``pip install -e .``."""

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def load_jepa_lightning(cls, path):
    """Load a ``JEPALightning``-style checkpoint, explaining the one failure that bites.

    Checkpoints from before the grounding head existed have no ``recon_head`` weights.
    Loading them non-strictly would succeed with a *randomly initialised* decoder and then
    report codec-space forecasts that mean nothing — the exact class of silent wrongness
    this project has spent its time removing. So the load stays strict and the error says
    what happened instead.
    """
    try:
        return cls.load_from_checkpoint(str(path), map_location="cpu")
    except RuntimeError as e:
        if "recon_head" not in str(e):
            raise
        raise SystemExit(
            f"\n{path} predates the grounding head (no recon_head weights), so it has no "
            f"latent->codec decoder.\nCodec-space forecasting is undefined for it — its "
            f"latent-space skill is still measurable via run_forecast.py.\nLoading it "
            f"non-strictly would silently score a random decoder, so this is refused."
        ) from e
