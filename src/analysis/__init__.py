"""Analiza wyników benchmarku: metryki liczone z pliku wyników (benchmarking zapisuje tylko
dane surowe). Podsumowanie z src/: python -m analysis.summary"""
from .metrics import REFS, compute_metrics, method_metrics

__all__ = ["REFS", "compute_metrics", "method_metrics"]
