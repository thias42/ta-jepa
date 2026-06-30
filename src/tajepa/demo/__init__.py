"""Web-demo builders (Gradio). Import lazily — the ``demo`` extra (gradio, matplotlib)
is only needed when actually building/launching a demo, not for core training/eval."""

from .anticipation import HEAD_JS, build_anticipation_demo

__all__ = ["HEAD_JS", "build_anticipation_demo"]
