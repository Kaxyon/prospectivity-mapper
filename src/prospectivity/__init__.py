"""ML-based mineral prospectivity mapping with known deposits.

The package turns free public geoscience data into a prospectivity probability
map, a ranked target list, and SHAP explanations of both.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "config",
    "grid",
    "sources",
    "labels",
    "features",
    "terrain",
    "geology",
    "geophysics",
    "geochemistry",
    "model",
    "evaluate",
    "explain",
    "viz",
    "pipeline",
]
