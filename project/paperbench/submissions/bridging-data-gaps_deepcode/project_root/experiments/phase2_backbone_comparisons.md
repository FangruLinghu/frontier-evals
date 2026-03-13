# Phase 2: Backbone Comparisons (DDPM vs LDM) under DPMs-ANT

Objective
- Systematically compare how the two diffusion backbones (DDPM in pixel space versus Latent Diffusion Model in latent space) interact with the Houlsby-style adaptor (ψ) and the adversarial noise transfer (ANT) mechanism.
- Isolate backbone-specific effects on target-domain adaptation performance across 10-shot tasks, with and without Adversarial Noise (AN).

Experimental scope
- Backbones: DDPM (pixel-space) and LDM (latent-space).
- Adaptor: shared Houlsby-style ψ module with backbone-specific bottleneck sizes (per Adapters Config):
  - DDPM: c = 4, d = 8
  - LDM: c = 2, d = 8
- Settings: with ANT (AN inner-max) and without AN to assess contribution of adversarial noise.
- Datasets: source pretraining domains (FFHQ, LSUN Church); target domains include 10-shot variants such as Sunglasses, Babies, Raphael paintings, Sketches, Modigliani portraits, Haunted Houses, Landscape drawings; 3 random seeds per target.
- Evaluation: Intra-LPIPS for diversity, FID for fidelity (with caveats for small samples), qualitative sample galleries.

Backbone configurations
- DDPM backbone
  - Pre-trained on source data; θ frozen during adaptor training
  - Diffusion steps T and associated schedules as per base_config.yaml (assumed)
- LDM backbone
  - Latent diffusion with an encoder/decoder, θ frozen during adaptor training
  - Latent dimension and time-conditioning implemented in LDMBackbone placeholder

Adaptor construction
- For each diffusion layer l, adaptor ψ_l with bottleneck c × d, as per bottleneck_configs:
  - DDPM: c=4, d=8
  - LDM: c=2, d=8
- Zero-initialized parameters to ensure no initial shift; added residual connection: x_t^l → θ_l(x_t^{l−1}) + ψ_l(x_t^{l−1})
- Gradient updates only apply to ψ; θ remains frozen

Inner maximization (Adversarial Noise AN)
- If enabled, perform inner-max optimization over ε by following inner_adversarial_noise procedure (Eq. 6–8) with γ, ω, J as in Phase 1/Config
- Outer loss combines L_sim and AN component; we compare with and without AN

Training protocol (Algorithm 1)
- 10-shot target sampling per batch
- Sample t ∈ {1,...,T} and ε ∼ N(0, I)
- Compute x_t = sqrt(ᾱ_t) x_0 + sqrt(1 − ᾱ_t) ε
- Compute ε_θ(x_t, t) with frozen θ; compute ∇_{x_t} log p_φ(y|x_t)
- Compute L_sim and, if AN enabled, L_AN using ϵ⋆ from inner-max
- Update only ψ via optimizer (Adam/AdamW); θ remains fixed

Data handling and reproducibility
- 10-shot target protocol with consistent preprocessing across runs
- 3 seeds per target; report mean and standard deviation over seeds
- Identical preprocessing for source and target, including normalization settings from base_config.yaml

Evaluation plan
- Metrics: Intra-LPIPS, FID, qualitative galleries
- Ablations:
  - Phase 2a: DDPM with AN vs without AN
  - Phase 2b: LDM with AN vs without AN
  - Phase 2c: DDPM vs LDM without AN under identical adaptor settings
- Phase 2 convergence expectations: ANT-enabled setups converge faster and yield higher target-domain fidelity/diversity relative to baselines, with cross-backbone generalization insights

Baselines and references
- Compare against Phase 1 baselines (DDPM and LDM without ANT)
- Include references to TGAN/ADA, EWC, CDC, DCL, DDPM-PA, LDM-ANT where applicable in notes

Reproducibility and logs
- Record hyperparameters gamma, omega, J, T, adaptor lr, bottleneck variants per backbone
- Save convergence plots and qualitative galleries per target/backbone
- Seed control: seeds = [0, 1, 2] or [10, 20, 30] depending on run; document in logs

Notes for researchers
- This document provides a blueprint for robust Phase 2 experiments; actual run-time details depend on the concrete implementation in the codebase (ant_trainer.py, losses.py, data_pipelines, evaluation tools).
- Ensure that data_pipelines and dataset loaders align with the 10-shot sampling protocol and augmentation settings.

Appendix
- Cross-referenced modules and their responsibilities:
  - Diffusion: src/models/diffusion_base/ddpm.py, src/models/diffusion_base/ldm.py; schedulers.py; diffusion_utils.py
  - Adaptor: src/adaptor/adaptor.py, adapters_config.py
  - Classifier: src/classifiers/domain_classifier.py, classifier_utils.py
  - Adversarial Noise: src/noise_optimization/adversarial_noise.py
  - Training: src/training/ant_trainer.py, src/training/losses.py, src/training/gradient_utils.py
  - Datasets: src/datasets/data_pipelines.py, src/datasets/augmentation.py
  - Evaluation: src/evaluation/fid.py, src/evaluation/intra_lpips.py, src/evaluation/visuals.py
  - Configs: configs/base_config.yaml, configs/ddpm_config.yaml, configs/ldm_config.yaml

"}]}_affirmationSorry, I made a mistake in the previous content; I'll provide the final content directly.