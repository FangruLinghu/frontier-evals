"""Models package surface.

This module provides a lazy-loading import surface for the various components
of the diffusion/adaptation framework implemented in this repository. Access
to subpackages (e.g., models.diffusion_base, models.adaptor) will be resolved on
first access via a small __getattr__ hook, avoiding heavy import costs at
package import time.

Public submodules exposed:
- diffusion_base
- adaptor
- classifiers
- noise_optimization
- training
- evaluation
- datasets
- tools

"""

from importlib import import_module
from typing import Any

__all__ = [
    "diffusion_base",
    "adaptor",
    "classifiers",
    "noise_optimization",
    "training",
    "evaluation",
    "datasets",
    "tools",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    # Lazy-import supported submodules
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__() -> list:
    return sorted(list(globals().keys()) + __all__)
