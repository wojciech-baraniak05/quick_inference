"""Scenariusze benchmarku.

batch  - cały X_test naraz,
online - pierwsze n_online próbek X_test, każda osobno.
"""

SUPPORTED_MODES = ("batch", "online")


def check_modes(modes):
    nieznane = [m for m in modes if m not in SUPPORTED_MODES]
    if nieznane:
        raise ValueError(f"nieznane modes {nieznane}, dozwolone: {SUPPORTED_MODES}")
    if not modes:
        raise ValueError("nie wybrano żadnego scenariusza (modes)")
    return tuple(modes)
