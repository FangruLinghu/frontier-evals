# Phase 3 Ablation Studies for DPMs-ANT

This document outlines the ablation study plan (Phase 3) to dissect contributions of core components in the DPMs-ANT framework. It builds on the Phase 1 baseline and Phase 2 backbone comparisons, focusing on how adversarial noise (AN), adaptor (ψ), and backbone choice (DDPM vs LDM) influence target-domain transfer efficiency, image fidelity, and diversity.

Authors: Research Team
Date: 2026-02-10

Note: All experiments assume 10-shot target data, fixed seeds (3–5 random seeds), identical preprocessing, and a shared diffusion schedule and optimizer configuration as described in the base configurations. Unless specified, we keep θ frozen (backbone) and train the Houlsby-style adaptor ψ.

## 0. Overview
- Objective: Systematically ablate components to quantify their impact on target-domain adaptation, convergence speed, and robustness.
- Core components under ablation:
  - Adversarial Noise (AN) component (on/off)
  - Houlsby adaptor ψ presence and bottleneck settings
  - Backbone type (DDPM vs LDM) under identical adaptor settings
  - Adaptor training regime (active vs frozen esp. θ)
- Evaluation metrics: FID, Intra-LPIPS (diversity), qualitative sample galleries, and convergence curves.
- Experimental cadence: 3–5 seeds per ablation, 300 iterations target transfer, with logging for reproducibility.

## 1. Ablation Scenarios
For each scenario, we run the same data setting and backbone configuration, only toggling the specified component. Where relevant, we compare across backbones (DDPM vs LDM) with the same adaptor settings.

### 1.1 Ablation A — Remove Adversarial Noise (AN) during outer loss
- Description: Disable Eq. 7/8 inner-max AN term; train with L_sim alone (Eq. 5).
- Hypothesis: AN accelerates convergence and improves target-domain robustness; removing AN will slow convergence and reduce target alignment.
- Variants: Per backbone, test with gamma=5, J=10, ω=0.02 disabled to observe impact on L_sim alone.

### 1.2 Ablation B — Adaptor ψ Disabled (Backbone-Only Transfer)
- Description: Remove adaptor contributions, i.e., set ψ = 0; the backbone θ is frozen and unused adaptor blocks do not contribute.
- Hypothesis: Without adaptor shifts, transfer relies solely on pre-trained priors; expect significantly reduced target alignment and diversity.

### 1.3 Ablation C — Backbone Comparison under Identical Adaptor (DDPM vs LDM)
- Description: Run Phase-2-like backbone comparison with identical adaptor architecture and AN settings, comparing DDPM vs LDM.
- Hypothesis: Cross-backbone differences in representation spaces affect transfer quality; measure whether LDM’s latent space provides advantages.
- Control: Bottleneck per-backbone as in adapters_config (DDPM c=4,d=8; LDM c=2,d=8).

### 1.4 Ablation D — Adaptor Bottleneck Variations per Backbone
- Description: Systematically vary per-backbone bottleneck sizes to quantify expressivity vs. parameter cost.
- Settings:
  - DDPM variants: (c,d) ∈ {(2,8), (4,8), (6,12)}
  - LDM variants: (c,d) ∈ {(2,8), (4,8)}
- Hypothesis: Larger bottlenecks yield better domain shifts up to a saturation point; measure impact on FID and Intra-LPIPS.

### 1.5 Ablation E — Adaptor ψ Training Regime (Zero Init vs Random Init)
- Description: Compare adaptor initialization strategies:
  - Zero initialization (as in Houlsby-style default)
  - Small random initialization (e.g., normal with small std)
- Hypothesis: Initialization affects early learning dynamics; test stability and convergence rate.

### 1.6 Ablation F — Backpropagation Scope (Whole Model vs Adaptor-Only)
- Description: Allow gradient flow not only through ψ but also through θ for a small subset of layers (e.g., last few blocks) to inspect sensitivity.
- Hypothesis: Minor gains possible if select layers benefit from joint optimization; more risk of destabilization.

### 1.7 Ablation G — γ, ω, J Sensitivity Sweep
- Description: Sweep hyperparameters to identify robustness margins:
  - γ ∈ {3, 5, 7}
  - ω ∈ {0.01, 0.02, 0.05}
  - J ∈ {5, 10, 20}
- Hypothesis: Moderate γ and ω balance stability and guidance; larger J accelerates inner maximization but with diminishing returns.

## 2. Experimental Protocol
- Seeds: Use 3–5 seeds per ablation scenario; report mean ± std.
- Data: 10-shot target samples per task; identical target sets across ablations for fair comparison.
- Backbones: Run both DDPM and LDM where applicable; keep source data and preprocessing identical.
- Training length: Target transfer of ~300 iterations (per plan) with early-stop metrics for convergence checks; log metrics every 50 iterations.
- Hardware: GPU-accelerated; maintain similar batch sizes (as per config) to ensure comparability.

## 3. Evaluation Plan
- Metrics: FID against target distribution, Intra-LPIPS, and qualitative galleries.
- Convergence: Compare convergence curves across ablations to identify which components drive faster alignment.
- Ablation-specific expectations:
  - AN-enabled runs should converge faster and exhibit higher target-domain fidelity/diversity.
  - Adaptor-disabled runs should undershoot target alignment dramatically.
  - Bottleneck variations reveal a sweet spot between expressivity and parameter efficiency.

## 4. Reproducibility and Logging
- Seeds: External RNG seeds; fixed across runs.
- Logging: Hyperparameters, seeds, and results logged to logs/experiments/phase3_ablation_studies.
- Checkpoints: Save best models and last-step adapters for analysis.

## 5. Deliverables and Reporting
- Experimental results tables for FID and Intra-LPIPS across ablations and backbones with mean ± std.
- Convergence plots and sample galleries per ablation scenario.
- Clear narrative comparing ablations to Phase 2 backbone comparisons and Phase 1 baselines.

## 6. Repro Steps
1. Prepare dataset: 10-shot target domains; ensure same seed initialization across ablations.
2. Configure model: choose backbone (DDPM or LDM); set bottleneck config per backbone as described.
3. Enable/disable components per ablation scenario as defined.
4. Run training: ANT training loop with the Phase-3 protocol.
5. Evaluate: compute FID and Intra-LPIPS on generated samples; save plots and galleries.
6. Compile results: aggregate across seeds and ablations for reporting.

## 7. Notes on Limitations
- The document outlines synthetic ablation experiments intended for a research setting. In the current repository, some components may require integration with actual data/hardware to reproduce results. The goal is to define a clear, repeatable plan and expected outcomes for subsequent execution.

---
This Phase 3 ablation plan complements the broader reproduction project and provides actionable guidance for investigating the contribution of AN, adaptor bottlenecks, and backbone choices in DPMs-ANT.
