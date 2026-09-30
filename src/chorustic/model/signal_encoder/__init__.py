"""Shared signal encoder utilities used by signal-level Chorus."""

from .dual_axis_signal_encoder import build_mantis_encoder, encode_with_mantis

__all__ = ["build_mantis_encoder", "encode_with_mantis"]
