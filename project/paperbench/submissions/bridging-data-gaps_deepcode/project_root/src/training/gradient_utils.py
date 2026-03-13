# Gradient utilities for training with frozen backbones and trainable adaptors
"""gradient_utils.py

A small collection of utility functions to inspect and manipulate gradients
in the ANT training setup. These helpers are intentionally lightweight and only
depend on PyTorch, providing a stable API for components that need to clip
gradients, compute gradient norms, or freeze/unfreeze parameters during
training of the Houlsby-style adaptors.

Public API:
- clip_grad_norm_(parameters, max_norm, norm_type=2.0) -> float
- grad_norm(parameters, norm_type=2.0) -> float
- zero_grad(parameters) -> None
- set_requires_grad(module, requires_grad: bool) -> None
- freeze_parameters(module) -> None
- unfreeze_parameters(module) -> None
"""
from __future__ import annotations

from typing import Iterable, Union

import torch
import torch.nn as nn

# Import alias to avoid name collision with PyTorch's clip_grad_norm_ when exposed here
from torch.nn.utils import clip_grad_norm as _pt_clip_grad_norm



def _iter_parameters(params: Union[nn.Module, Iterable[torch.nn.Parameter]]):
    """Return an iterable over parameters.

    If a module is given, returns its parameters(); otherwise assumes an iterable
    of parameters.
    """
    if isinstance(params, nn.Module):
        return params.parameters()
    return params


def clip_grad_norm_(parameters: Union[nn.Module, Iterable[torch.nn.Parameter]], max_norm: float, norm_type: float = 2.0) -> float:
    """Clip gradient norms of the given parameters in-place and return the total norm.

    This is a thin wrapper around PyTorch's functional clip_grad_norm_ to provide a
    consistent API across the training codebase.

    Args:
        parameters: Module or iterable of parameters whose gradients will be clipped.
        max_norm: Maximum allowed norm of the gradients.
        norm_type: Type of the used norm (e.g., 2.0 for L2 norm).

    Returns:
        The total norm before clipping (as a scalar float).
    """
    paras = _iter_parameters(parameters)
    return _pt_clip_grad_norm(paras, max_norm, norm_type=norm_type)


def grad_norm(parameters: Union[nn.Module, Iterable[torch.nn.Parameter]], norm_type: float = 2.0) -> float:
    """Compute the gradient norm of the given parameters.

    If no gradients are present, returns 0.0. This helper treats param.grad as the
    quantity whose norm is measured.
    """
    paras = _iter_parameters(parameters)
    total_norm = 0.0
    for p in paras:
        if isinstance(p, torch.nn.Parameter) and p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    if total_norm == 0.0:
        return 0.0
    return total_norm ** (1.0 / norm_type)


def zero_grad(parameters: Union[nn.Module, Iterable[torch.nn.Parameter]]) -> None:
    """Zero-out gradients for the given parameters in-place."""
    paras = _iter_parameters(parameters)
    for p in paras:
        if isinstance(p, torch.nn.Parameter) and p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    """Set the requires_grad flag for all parameters in a module."""
    for p in module.parameters():
        p.requires_grad = requires_grad


def freeze_parameters(module: nn.Module) -> None:
    """Freeze all parameters of a module (set requires_grad=False)."""
    set_requires_grad(module, False)


def unfreeze_parameters(module: nn.Module) -> None:
    """Unfreeze all parameters of a module (set requires_grad=True)."""
    set_requires_grad(module, True)


__all__ = [
    "clip_grad_norm_",
    "grad_norm",
    "zero_grad",
    "set_requires_grad",
    "freeze_parameters",
    "unfreeze_parameters",
]
