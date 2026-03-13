"""
Simformer: All-in-one simulation-based inference

Implementation based on:
Gloeckler, M., Deistler, M., Weilbach, C., Wood, F., & Macke, J. H. (2024).
All-in-one simulation-based inference. ICML 2024.
"""

from simformer.models.simformer import Simformer
from simformer.models.transformer import TransformerEncoder, TransformerEncoderWithTime
from simformer.models.score_network import ScoreNetwork, ConditionalScoreNetwork
from simformer.diffusion.sde import VESDE, VPSDE, SDE
from simformer.diffusion.sampling import (
    sample_posterior,
    sample_likelihood,
    sample_joint,
    sample_arbitrary_conditional,
)
from simformer.training.trainer import Trainer, TrainingConfig
from simformer.training.losses import SimformerLoss, denoising_score_matching_loss

__version__ = "0.1.0"

__all__ = [
    # Main model
    "Simformer",
    # Model components
    "TransformerEncoder",
    "TransformerEncoderWithTime",
    "ScoreNetwork",
    "ConditionalScoreNetwork",
    # SDE
    "SDE",
    "VESDE",
    "VPSDE",
    # Sampling
    "sample_posterior",
    "sample_likelihood",
    "sample_joint",
    "sample_arbitrary_conditional",
    # Training
    "Trainer",
    "TrainingConfig",
    "SimformerLoss",
    "denoising_score_matching_loss",
]
