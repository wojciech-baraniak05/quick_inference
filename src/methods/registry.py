"""Budowanie metod inferencji z bloków [[methods]] w config.toml."""
from collections.abc import Iterable, Mapping
from typing import Any

from .base import InferenceMethod, Task
from .glee import Glee
from .hyper import Hyper
from .ib_pruning import IBPruning
from .margin import MarginEarlyStop
from .own import Own, OwnRegression
from .regression import FullAveraging, GleeRegression, IBRegression, SklearnRFReg
from .voting import FullVoting, SklearnRF

MethodSpec = Mapping[str, Any]      # np. {"type": "ib_pruning", "pi": 0.98, "enabled": True}

# rf_sklearn włącza się osobno, przez benchmark.sklearn_soft_voting
METHOD_TYPES: dict[str, type[InferenceMethod]] = {
    FullVoting.name: FullVoting,
    "ib_pruning": IBPruning,
    "hyper": Hyper,
    "glee": Glee,
    "margin": MarginEarlyStop,
    "own": Own,
    # regresja
    FullAveraging.name: FullAveraging,
    "ib_reg": IBRegression,
    "glee_reg": GleeRegression,
    "own_reg": OwnRegression,
}


def build_method(spec: MethodSpec) -> InferenceMethod:
    """Klucze poza type i enabled są argumentami konstruktora klasy METHOD_TYPES[type]."""
    spec = dict(spec)
    method_type = spec.pop("type", None)
    spec.pop("enabled", None)
    if method_type == SklearnRF.name:
        raise ValueError(f"'{SklearnRF.name}' is enabled by benchmark.sklearn_soft_voting, "
                         f"not by [[methods]]")
    if method_type not in METHOD_TYPES:
        raise ValueError(f"unknown method type {method_type!r}, "
                         f"expected one of {list(METHOD_TYPES)}")
    try:
        return METHOD_TYPES[method_type](**spec)
    except TypeError as e:
        raise ValueError(f"bad parameters for {method_type!r}: {e}") from None


def build_methods(specs: Iterable[MethodSpec], sklearn_soft_voting: bool = True,
                  task: Task = "classification") -> list[InferenceMethod]:
    """Nowe (niedopasowane) metody według configu dla zadania task; bloki metod innego
    zadania są pomijane. rf_sklearn idzie na początek listy."""
    rf_sklearn: InferenceMethod = SklearnRFReg() if task == "regression" else SklearnRF()
    methods: list[InferenceMethod] = [rf_sklearn] if sklearn_soft_voting else []
    for s in specs:
        # typ decyduje o zadaniu, więc bloki innego zadania nie są nawet tworzone
        method_task = getattr(METHOD_TYPES.get(s.get("type")), "task", task)
        if s.get("enabled", True) and method_task == task:
            methods.append(build_method(s))
    names = [m.name for m in methods]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate method names: {duplicates} (give them different 'name')")
    if not methods:
        raise ValueError(f"no methods selected (task: {task})")
    return methods
