"""Source package initializer for the diffusion-score project.

This module exposes the core building blocks used by downstream modules,
including the joint tokenizer and embeddings utilities. It also provides
easy access to the lightweight mask utilities via a submodule export.
"""

__version__ = "0.0.1"

# Re-export commonly used classes for convenient imports
from .tokenizer import JointTokenizer  # noqa: F401
from .embeddings import Embeddings  # noqa: F401

# Expose utility submodules for optional direct access
from . import mask_utils as mask_utils  # noqa: F401

__all__ = ["JointTokenizer", "Embeddings", "mask_utils", "__version__"]
