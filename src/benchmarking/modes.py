"""Scenariusze benchmarku: batch (cały X_test naraz) i online (każda próbka osobno)."""
from collections.abc import Sequence
from typing import Literal, cast, get_args

Mode = Literal["batch", "online"]
SUPPORTED_MODES: tuple[Mode, ...] = get_args(Mode)


def check_modes(modes: Sequence[str]) -> tuple[Mode, ...]:
    nieznane = [m for m in modes if m not in SUPPORTED_MODES]
    if nieznane:
        raise ValueError(f"nieznane modes {nieznane}, dozwolone: {SUPPORTED_MODES}")
    if not modes:
        raise ValueError("nie wybrano żadnego scenariusza (modes)")
    return cast(tuple[Mode, ...], tuple(modes))
