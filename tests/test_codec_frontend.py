"""Codec frontend test.

Downloads EnCodec weights on first run, so it's marked slow and skipped unless
TAJEPA_RUN_CODEC_TESTS=1. The shape/contract assertions are what matter: continuous
[B, T, D] embeddings at the documented 75 Hz / dim 128.
"""

import os

import pytest
import torch

RUN = os.environ.get("TAJEPA_RUN_CODEC_TESTS") == "1"


@pytest.mark.skipif(not RUN, reason="set TAJEPA_RUN_CODEC_TESTS=1 to run (downloads weights)")
def test_encodec_frontend_shapes():
    from tajepa.config import CodecConfig
    from tajepa.codec.frontend import build_frontend

    fe = build_frontend(CodecConfig(name="encodec_24khz", device="cpu"))
    assert fe.frame_rate == pytest.approx(75.0, abs=1.0)
    assert fe.embedding_dim == 128

    wav = torch.randn(2, 1, fe.sample_rate)  # 2 x 1 second
    emb = fe.encode(wav)
    assert emb.dim() == 3 and emb.shape[0] == 2 and emb.shape[-1] == fe.embedding_dim
    # ~75 frames for 1 s
    assert abs(emb.shape[1] - 75) <= 2


def test_registry_lists_encodec():
    from tajepa.codec.frontend import _REGISTRY

    assert "encodec_24khz" in _REGISTRY
