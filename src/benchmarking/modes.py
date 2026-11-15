"""Scenariusze benchmarku: batch (cały X_test naraz) i online (każda próbka osobno)."""
from collections.abc import Sequence
from typing import Literal, cast, get_args

Mode = Literal["batch", "online"]
SUPPORTED_MODES: tuple[Mode, ...] = get_args(Mode)


def check_modes(modes: Sequence[str]) -> tuple[Mode, ...]:
    unknown = [m for m in modes if m not in SUPPORTED_MODES]
    if unknown:
        raise ValueError(f"unknown modes {unknown}, expected from {SUPPORTED_MODES}")
    if not modes:
        raise ValueError("no modes selected")
    return cast(tuple[Mode, ...], tuple(modes))
