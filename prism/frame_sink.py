"""Stable import path for the player and desk (`from prism.frame_sink import ...`)."""

from prism.cli.frame_sink import *  # noqa: F403
from prism.cli.frame_sink import BASE_NAME, NAME, FrameSink

__all__ = ["BASE_NAME", "NAME", "FrameSink"]
