# Phase 1 Baseline Reproduction: Phase 1 ANT Baselines with Dual Backbones (DDPM and LDM)

This document provides a compact, reproducible recipe to reproduce the Phase 1 baseline results described in the DPMs-ANT plan. It focuses on validating the plumbing of the diffusion backbones (DDPM and Latent Diffusion Model, LDM), the Houlsby-style adaptor, and the training loop when adversarial noise (AN) is not yet activated or when a minimal ANT setup is used for baseline comparisons. The goal is to establish a robust, end-to-end baseline using a single target domain and two backbones, before progressing to more complex ablations and cross-backbone studies.

Note: This document assumes the repository structure described in the reproduction plan and that the codebase components (diffusion backbones, adaptor, classifier, AN module, and training loop) are importable as indicated in the codebase.


## 1) Experimental Objective
- Reproduce a core baseline of DPMs-ANT training on a single 10-shot target domain using both backbones (DDPM and LDM).
- Verify basic data flow: small adaptor ψ layers (Houlsby-style) with a frozen backbone θ, using a similarity-based gradient signal from a domain classifier pϕ, and optionally a lightweight inner-max AN signal (depending on the baseline configuration).
- Validate end-to-end working training loop (Algorithm 1) and confirm that the code path from data loading to model updates is sound.
- Provide initial target-domain fidelity and diversity assessments (FID and a lightweight intra-LPIPS proxy) to establish a reference point.

## 2) Data, Targets, and Protocol
- Source data (pre-training): FFHQ and LSUN Church are used to pre-train the backbones.
- Target data: 10-shot samples from several target domains (examples listed in the plan). Targets are sampled deterministically using fixed seeds to ensure reproducibility.
- 10-shot protocol:
  - For each target task, sample 10 images and apply a random diffusion forward process to generate x_t with a random timestep t per sample.
  - Preprocess and normalize images consistently across source and target domains.
- Data splits and seeds:
  - Use 3 random seeds (e.g., seeds = [0, 1, 2]) for ablation stability.
  - Each seed yields an independent run for reporting mean and standard deviation.

## 3) Backbones and Adaptor Configuration
- Backbones:
  - DDPM (pixel-space diffusion) and Latent Diffusion Model (LDM, latent space diffusion).
  - θ (the denoiser) is frozen during adaptor training (no gradient updates).
- Adaptor ψ (Houlsby-style): per-layer bottlenecked residual adaptor inserted into the backbone by the following rule per layer l:
  - x_t^l → x_t^l + ψ_l(x_t^{l−1})
  - Bottleneck sizes (per backbone, via adapters_config):
    - DDPM: c=4, d=8
    - LDM:  c=2, d=8
  - Adaptor parameters initialized to zero to ensure zero initial shift.
- Similarity classifier pϕ:
  - A lightweight domain discriminator that outputs log probabilities for {source, target} given x_t.
  - Used to supply gradient ∇_{x_t} log pϕ(y|x_t) for the L_sim term. The classifier is kept fixed or lightly updated depending on the experimental setup.

## 4) Losses and Optimization (Algorithm 1 Foundations)
- Loss L_sim (Eq. 5): similarity-guided loss using the gradient signal ∇_{x_t} log pϕ(y|x_t) from the domain classifier.
- Adversarial Noise (AN): optional inner-max objective to identify worst-case noise ε⋆ as described in Eq. 6–Eq. 8, enabling robust transfer; for Phase 1 baselines, you may start without AN or with a minimal AN configuration to verify plumbing.
- Outer optimization:
  - Update only adaptor ψ; backbone θ is frozen.
  - Optimizer choice: Adam/AdamW with gradient clipping (norm 1.0) for ψ. Learning rates tuned per backbone (e.g., adaptor lr = 5e-5 to 1e-4).
- Hyperparameters (default values, to be confirmed during replication):
  - γ (similarity weight) ~ 5.0
  - ω (AN inner-max step) ~ 0.02
  - J (inner-max iterations) ~ 10
  - T (diffusion steps): as defined by the backbone schedule (DDPM or LDM).

## 5) Training Protocol and Orchestration
- Algorithm flow (high-level):
  1) Sample x0 from the 10-shot target domain (x0 ∈ R^{C×H×W}).
  2) Sample t ∈ {1..T} and ε ∼ N(0, I); compute x_t = sqrt(ᾱ_t) x0 + sqrt(1 − ᾱ_t) ε using the backbone's diffusion schedule.
  3) Compute εθ(x_t, t) with the frozen backbone θ.
  4) Compute ∇_{x_t} log pϕ(y|x_t) via the domain classifier; use in L_sim and, if AN is enabled, inner-max loss.
  5) Backpropagate the total loss w.r.t. adaptor ψ only; update ψ with Adam/AdamW, with gradient clipping.
  6) Repeat for the chosen number of iterations per target task (typical target: ~300 iterations for Phase 1 baselines).
- Logging:
  - Log per-iteration totals and individual loss components for traceability.
  - Save checkpoints per the base config (log_every, save_every).

## 6) Evaluation and Metrics
- Quantitative:
  - Intra-LPIPS: internal, lightweight diversity proxy across generated samples for the target style.
  - FID: fidelity proxy against the target distribution; note that FID with a small target set is unstable, so interpret with caution.
- Qualitative:
  - Save representative samples at intervals to build a qualitative gallery for the target style.
- Reporting:
  - For each seed, report mean and standard deviation across seeds for all metrics.

## 7) Repro Steps and Commands
- Environment setup:
  - python3 -m venv .venv
  - source .venv/bin/activate
  - pip install -r requirements.txt
- Data preparation:
  - Structure data under: data/source/{FFHQ, LSUN_Church} and data/target/{target_task_name} with 10-shot samples per target.
  - Optional: apply augmentation as per augmentation pipeline.
- Training:
  - Prepare config for the target backbone (either ddpm_config.yaml or ldm_config.yaml) and run the training script:
    - Example: python3 -u src/train_loop.py --config configs/base_config.yaml
  - Ensure seeds are fixed per run to enable reproducibility.
- Evaluation:
  - Run evaluation using eval_loop.py with the corresponding config for the 10-shot target task. Compute Intra-LPIPS and FID (with the lightweight evaluation utilities included in the repo).

## 8) Expected Outcomes and Baselines
- Phase 1 baseline should establish a robust end-to-end pipeline for 10-shot target transfer with DDPM and LDM backbones.
- Expect improved target-domain fidelity and diversity relative to non-ANT baselines, with reasonable convergence likely around the ~300 iterations mark (per the plan).
- The phase-1 baselines provide a reference point for Phase 2 ablations and cross-backbone comparisons.

## 9) References to Implementation and Data Locations
- Backbones: src/models/diffusion_base/ddpm.py, src/models/diffusion_base/ldm.py
- Adaptor: src/adaptor/adaptor.py, adapters_config.py
- Similarity Classifier: src/classifiers/domain_classifier.py
- Adversarial Noise: src/noise_optimization/adversarial_noise.py
- Training Loop: src/training/ant_trainer.py, src/training/losses.py
- Data: src/datasets/data_pipelines.py, src/datasets/augmentation.py
- Evaluation: src/evaluation/fid.py, src/evaluation/intra_lpips.py, src/evaluation/visuals.py
- Configs: configs/base_config.yaml, configs/ddpm_config.yaml, configs/ldm_config.yaml
- Utilities: src/tools/logging_utils.py, src/training/gradient_utils.py

## 10) Reproducibility Details
- Seeds: 3 seeds for stability; track mean and std of metrics across seeds.
- Determinism: enable deterministic ops where feasible in PyTorch; apply fixed random seeds for numpy and Python standard library.
- Checkpoints and logs: Save per-iteration and per-seed logs to the configured log directory; gather results in a compact summary report.

> This Phase 1 Baseline Repro document serves as a blueprint for researchers to replicate the baseline experiments across both backbones and establish comparability with the ANT extensions planned in subsequent phases.
