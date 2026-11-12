"""Budowanie metod inferencji z bloków [[methods]] w config.toml."""
from collections.abc import Iterable, Mapping
from typing import Any

from .base import InferenceMethod
from .glee import Glee
from .hyper import Hyper
from .ib_pruning import IBPruning
from .margin import MarginEarlyStop
from .voting import FullVoting, SklearnRF

MethodSpec = Mapping[str, Any]      # np. {"type": "ib_pruning", "pi": 0.98, "enabled": True}

# rf_sklearn włącza się osobno, przez benchmark.sklearn_soft_voting
METHOD_TYPES: dict[str, type[InferenceMethod]] = {
    FullVoting.name: FullVoting,
    "ib_pruning": IBPruning,
    "hyper": Hyper,
    "glee": Glee,
    "margin": MarginEarlyStop,
}


def build_method(spec: MethodSpec) -> InferenceMethod:
    """Klucze poza type i enabled są argumentami konstruktora klasy METHOD_TYPES[type]."""
    spec = dict(spec)
    typ = spec.pop("type", None)
    spec.pop("enabled", None)
    if typ == SklearnRF.name:
        raise ValueError(f"'{SklearnRF.name}' włącza się przez benchmark.sklearn_soft_voting, "
                         f"nie przez [[methods]]")
    if typ not in METHOD_TYPES:
        raise ValueError(f"nieznany typ metody {typ!r}, dostępne: {list(METHOD_TYPES)}")
    try:
        return METHOD_TYPES[typ](**spec)
    except TypeError as e:
        raise ValueError(f"złe parametry metody {typ!r}: {e}") from None


def build_methods(specs: Iterable[MethodSpec],
                  sklearn_soft_voting: bool = True) -> list[InferenceMethod]:
    """Nowe (niedopasowane) metody według configu; rf_sklearn idzie na początek listy."""
    methods: list[InferenceMethod] = [SklearnRF()] if sklearn_soft_voting else []
    methods += [build_method(s) for s in specs if s.get("enabled", True)]
    names = [m.name for m in methods]
    duplikaty = sorted({n for n in names if names.count(n) > 1})
    if duplikaty:
        raise ValueError(f"powtórzone nazwy metod: {duplikaty} (ustaw im różne 'name')")
    if not methods:
        raise ValueError("nie wybrano żadnej metody do benchmarku")
    return methods
