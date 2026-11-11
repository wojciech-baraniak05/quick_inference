"""Budowanie metod inferencji z opisów w config.toml.

Każdy wpis [[methods]] to słownik z kluczem type (nazwa z METHOD_TYPES),
opcjonalnym enabled (domyślnie true) i parametrami konstruktora danej klasy, np.

    {"type": "ib_pruning", "pi": 0.98}
    {"type": "hyper", "pi": 0.99}
    {"type": "glee", "pi": 0.99}
"""
from .glee import Glee
from .hyper import Hyper
from .ib_pruning import IBPruning
from .voting import FullVoting, SklearnRF

# typy metod dostępne w config.toml; rf_sklearn włącza się osobno
# przez benchmark.sklearn_soft_voting
METHOD_TYPES = {
    FullVoting.name: FullVoting,
    "ib_pruning": IBPruning,
    "hyper": Hyper,
    "glee": Glee,
}


def build_method(spec):
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


def build_methods(specs, sklearn_soft_voting=True):
    """Lista nowych (niedopasowanych) metod według configu.

    sklearn_soft_voting=True  - na początku listy jest rf_sklearn (soft voting),
    sklearn_soft_voting=False - tylko metody odpytujące pojedyncze drzewa.
    """
    methods = [SklearnRF()] if sklearn_soft_voting else []
    methods += [build_method(s) for s in specs if s.get("enabled", True)]
    names = [m.name for m in methods]
    duplikaty = sorted({n for n in names if names.count(n) > 1})
    if duplikaty:
        raise ValueError(f"powtórzone nazwy metod: {duplikaty} (ustaw im różne 'name')")
    if not methods:
        raise ValueError("nie wybrano żadnej metody do benchmarku")
    return methods
