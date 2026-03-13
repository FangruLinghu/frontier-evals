# DPMs-ANT Algorithm Summaries

This document provides a concise, traceable summary of the core algorithms, equations, and training workflow implemented in the DPMs-ANT project. It ties the mathematical formulations to the corresponding code modules and files in the repository.

## Core Components and Equations
- L_sample (Eq. 1): Denoising score matching loss used to train the denoiser θ of the diffusion backbone. 
  - L_sample(θ) = E_{t, x0, ε} [ || ε − ε_θ(x_t, t) ||^2 ]
  - In practice, ε is predicted by the backbone given a noisy input x_t generated as x_t = sqrt(ᾱ_t) x0 + sqrt(1 − ᾱ_t) ε.
  - Implementations: src/models/diffusion_base/diffusion_utils.py and the backbone code in src/models/diffusion_base/ddpm.py and src/models/diffusion_base/ldm.py.

- Denoiser equation (μ_θ) and forward passes (Eq. 2, Eq. 4 context):
  - The forward diffusion step computes x_t from x0 and noise ε using time-step t and schedule terms ᾱ_t, β_t.
  - The backbones provide ε_θ(x_t, t) which is used to reconstruct x0 and propagate the denoising signal.
  - Implementations: diffusion_utils.py, schedulers.py, and the backbone definitions.

- Similarity loss (Eq. 5): L_sim for adaptor-guided alignment between source and target diffusion states.
  - L_sim = E_{t, x0 ∼ target} [ || ε_t − ε_θ(x_t, t) − σ_t^2 γ ∇_{x_t} log p_φ(y|x_t) ||^2 ]
  - γ > 0 weights the gradient signal from the domain classifier p_φ.
  - The adaptor ψ modifies intermediate representations to shift x_t towards the target domain before passing to θ.
  - Implementation: src/training/losses.py (L_sim) and the adaptor integration in src/adaptor/adaptor.py and src/training/ant_trainer.py.

- Adversarial noise inner-max (Eq. 6–Eq. 7) and worst-case ε⋆ (Eq. 8):
  - Inner maximization searches for ε⋆ that worsens denoising given current adaptor and backbone:
    - ε_{j+1} = Norm[ ε_j + ω ∇_{ε_j} (− loss(ε_j)) ]  (Eq. 7)
    - loss(ε) ≈ || ε − ε_θ(x_t(ε), t) − σ̂_t^2 γ ∇_{x_t} log p_φ(y|x_t) ||^2
  - The outer loss then uses ε⋆ to form the adversarial objective (Eq. 8).
  - Implementation: src/noise_optimization/adversarial_noise.py and the inner loop orchestrated by src/training/ant_trainer.py.

- Algorithm 1: Training DPMs with ANT
  - Outer loop updates adaptor ψ while θ is frozen.
  - Data flow per iteration:
    1) Sample x0 from target distribution (10-shot data).
    2) Sample t ∈ {1,..., T} and ε ∼ N(0, I); compute x_t.
    3) Compute ε⋆ via inner-max (AN module).
    4) Compute L(ψ) using ε⋆ and gradient signals ∇_{x_t} log p_φ(y|x_t).
    5) Backpropagate to update ψ; freeze θ.
  - Implementations: Algorithm described in src/training/ant_trainer.py with auxiliary components in the other modules.

- Data processing and 10-shot protocol
  - 10-shot target samples are diffused to create x_t states for each training step.
  - Data loaders and samplers are provided to generate batches for the ANT training loop (TenShotTargetDataset and data pipelines).
  - Implementations: src/training/losses.py, src/datasets/data_pipelines.py, and src/datasets/augmentation.py.

- Evaluation helpers
  - Intra-LPIPS and FID-like metrics are provided to assess diversity and fidelity.
  - Implementations: src/evaluation/intra_lpips.py (proxy LPIPS), src/evaluation/fid.py (FID proxy), and src/evaluation/visuals.py for visualization.

- Baselines and ablations
  - The project scaffolds several baselines (GAN-based, EWC, CDC, DCL, etc.) and ablations (ANT without AN, adaptor-only with frozen θ, etc.).
  - Validation and experiments are documented under experiments/ phase-docs.

## Mapping to Implementation Files
- DDPM and LDM backbones
  - src/models/diffusion_base/ddpm.py
  - src/models/diffusion_base/ldm.py
  - src/models/diffusion_base/schedulers.py
  - src/models/diffusion_base/diffusion_utils.py
- Adaptor framework
  - src/adaptor/adaptor.py
  - src/adaptor/adapters_config.py
  - bottleneck_configs in adapters_config.py
- Similarity classifier
  - src/classifiers/domain_classifier.py
  - src/classifiers/classifier_utils.py
- Adversarial noise optimization
  - src/noise_optimization/adversarial_noise.py
- Training loop and losses
  - src/training/ant_trainer.py
  - src/training/losses.py
  - src/training/gradient_utils.py
- Data handling
  - src/datasets/data_pipelines.py
  - src/datasets/augmentation.py
  - src/datasets/ten_shot_loader.py (conceptual; implemented as part of losses module in this repo)
- Evaluation and visuals
  - src/evaluation/fid.py
  - src/evaluation/intra_lpips.py
  - src/evaluation/visuals.py
- Utilities and packaging
  - src/tools/logging_utils.py
  - src/models/__init__.py, src/tools/logging_utils.py
- Configurations
  - project_root/configs/base_config.yaml
  - project_root/configs/ddpm_config.yaml
  - project_root/configs/ldm_config.yaml
  - project_root/setup_env.sh, project_root/README.md, project_root/requirements.txt

## Quick Reproduction Guidelines
- Ensure the environment is prepared (setup_env.sh or equivalent).
- Prepare a base configuration (configs/base_config.yaml) and backbone-specific overrides (configs/ddpm_config.yaml or configs/ldm_config.yaml).
- Run the training loop: python -m src.train_loop or a provided runner script.
- Validate with evaluation scripts: python -m src.eval_loop or the dedicated evaluation entrypoints.

> Note: This document is kept lightweight and high-level to avoid duplicating extensive docstrings embedded in the code. It serves as a map from equations to code modules and helps ensure reproducibility.
