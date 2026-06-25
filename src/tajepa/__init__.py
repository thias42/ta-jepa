"""Temporally-controlled general-purpose audio JEPA.

A causal, action-conditioned latent world model for general audio. See
``temporal-audio-jepa-plan.md`` for the design rationale and phase plan, and
``CLAUDE.md`` for the quick-reference invariants.

Phase 0 (this scaffold) provides: a codec embedding frontend, offline embedding
caching, an APC baseline, a log-mel frontend for the A-JEPA-comparable baseline,
and the data/manifest plumbing they share.
"""

__version__ = "0.0.1"
