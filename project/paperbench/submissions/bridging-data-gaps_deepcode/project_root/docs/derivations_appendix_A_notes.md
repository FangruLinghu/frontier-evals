# Appendix A: Derivations and Notation for DPMs-ANT

This appendix provides detailed derivations and notation that connect the mathematical formulation of the DPMs-ANT framework to the implemented code components. Equations are cited by their numbers as referenced in the main text and corresponding code (Eq. 1–Eq. 8, Algorithm 1).

A. Notation recap
- x0 ∈ R^{C×H×W}: clean image (or latent code for LDMs)
- t ∈ {1,...,T}: diffusion timestep (randomly drawn per sample)
- x_t: diffused/noisy state at timestep t
- ε ∼ N(0, I): standard Gaussian noise used in forward process
- θ: frozen denoiser (backbone) parameters (DDPM/LDM denoiser)
- ψ: Houlsby-style adaptor parameters (trainable)
- ψ_l: adaptor at layer l (per-layer adaptor)
- p_φ(y|x_t): binary domain classifier output (log-probabilities) with gradient ∇_{x_t} log p_φ(y|x_t)
- γ: similarity weight that scales the gradient term from the classifier
- α_t, ᾱ_t, β_t: diffusion schedule terms; ᾱ_t = ∏_{s=1}^t α_s (cumulative product)
- σ_t^2: per-step variance term, typically related to 1 − ᾱ_t in the denoising equation
- ˆσ_t: normalized or estimated noise scale used in loss formulations
- θ, ψ: as above, used in forward passes with the relation x_t = √ᾱ_t x0 + √(1 − ᾱ_t) ε

B. Forward process and L_sample (Eq. 1)
- The standard diffusion forward process draws ε ∼ N(0, I) and forms x_t as:
  x_t = √ᾱ_t x0 + √(1 − ᾱ_t) ε
- The primary training objective L_sample(θ) (Eq. 1) for the backbone denoiser is:
  L_sample(θ) = E_{t, x0, ε} [ || ε − ε_θ(x_t, t) ||^2 ]
  where ε_θ(x_t, t) is the model-predicted noise at state x_t and timestep t.
- In ANT, θ is frozen; training updates ψ to influence x_t through the adaptor and steer the denoiser indirectly via gradient signals.

C. Denoising and μ_θ (Eq. 2 context)
- The denoiser μ_θ represents the predicted mean of the reverse process; in DDPM notation:
  μ_θ(x_t, t) = (1/√α_t) (x_t − ((1 − α_t)/√(1 − ᾱ_t)) ε_θ(x_t, t))
- In code, the surrogate objective emphasizes matching ε with ε_θ rather than explicitly computing μ_θ, but the relationship underpins the denoising step used by the backbone.

D. Forward diffusion mechanics and x_t formulation (Eq. 1 context)
- As above, x_t depends on x0, ε, and the diffusion schedule. This forms the core data path through which adaptor ψ interacts with the diffusion backbone.
- The adaptor shifts intermediate representations: x_t^l → x_t^l+ = θ_l(x_t^{l−1}) + ψ_l(x_t^{l−1}) where ψ_l adds a low-rank residual to the backbone features.

E. Similarity loss L_sim (Eq. 5)
- The similarity-guided loss encourages alignment between the denoised state and a source-target similarity signal derived from p_φ:
  L_sim = E_{t, x0} [ || ε_t − ε_θ(x_t, t) − ˆσ_t^2 γ ∇_{x_t} log p_φ(y|x_t) ||^2 ]
- Here, ε_t is the true noise at time t, and the gradient term provides a direction to push x_t toward source-like representations to improve cross-domain similarity while adapting ψ.
- In the code, this loss is computed using the inner forward path with ψ embedded in the backbone, and ε_θ is evaluated by the frozen θ.

F. Inner-max Adversarial Noise (Eq. 6–Eq. 7) and ε⋆ (Eq. 8)
- The adversarial noise optimizer searches for a worst-case perturbation ε* that degrades denoising performance for a given (x0, t):
  - Initialize ε_0 ∼ N(0, I)
  - Iterate j = 0,...,J−1: ε_{j+1} = Normalize( ε_j + ω ∇_{ε_j} (− Loss(ε_j)) )
- The inner loss (loss(ε)) is defined as:
  Loss(ε) = E_t [ || ε − ε_θ(x_t(ε), t) − ˆσ_t^2 γ ∇_{x_t} log p_φ(y|x_t) ||^2 ]
  where x_t(ε) = √ᾱ_t x0 + √(1 − ᾱ_t) ε.
- After finding ε⋆, the outer objective L_AN contributes a robustification term, typically expressed as:
  L_AN = E_t [ || ε⋆ − ε_θ(x_t⋆, t) − ˆσ_t^2 γ ∇_{x_t} log p_φ(y|x_t⋆) ||^2 ]
- Eq. 8 formalizes the worst-case perturbation magnitude and its impact on denoising.
- In practice, ε⋆ is used to compute a second loss component that steers ψ toward reducing vulnerability to adversarial noise in diffusion steps.

G. Training loop: Algorithm 1 (outer adaptor training)
- The central training loop updates only ψ while θ remains fixed. Pseudocode:
  - For each iteration:
    - Sample x0 ∼ TargetData, t ∼ Uniform(1..T), ε ∼ N(0, I)
    - Compute x_t = √ᾱ_t x0 + √(1 − ᾱ_t) ε
    - Compute ε⋆ via inner maximization: ε⋆ = arg max_ε Loss(ε)
      (as per Eq. 6–Eq. 7 and inner optimization details)
    - Compute L_sim using ε_t, ε_θ(x_t, t), ∇ log p_φ(y|x_t)
    - Compute L_AN using ε⋆, ε_θ(x_t⋆, t), ∇ log p_φ(y|x_t⋆)
    - Total loss = L_sim + L_AN; backprop through ψ only; clip gradients as needed
    - Step optimizer for ψ (e.g., Adam/AdamW with gradient clipping)

H. Data flow and 10-shot protocol alignment (10-shot protocol)
- Data sampling yields a small target-domain batch; each sample x0 is processed through the forward diffusion to obtain x_t for a random t.
- The 10-shot protocol is handled in the data-loading layer and is reflected in ∇_{x_t} log p_φ(y|x_t) via p_φ.

I. Hyperparameters and notes
- γ = 5.0 (similarity control)
- ω = 0.02 (inner-max step size)
- J = 10 (inner-max iterations)
- T depends on backbone; typical values 1000 (DDPM) or specific to LDM
- Learning rates: adaptor ψ ≈ 5e-5 to 1e-4; θ is frozen
- Stability: gradient clipping with norm 1.0 to bound adaptor updates

Pseudo-algorithm (Algorithm 1 mapping to code)
1. Initialize ψ (adaptor) with zero initialization; freeze θ (backbone)
2. For each training step:
   a. Sample x0 ∼ target data; draw t ∈ {1..T}; Sample ε ∼ N(0, I)
   b. Form x_t = √ᾱ_t x0 + √(1 − ᾱ_t) ε
   c. Compute ε⋆ via inner_adversarial_noise (Eq. 6–7)
   d. Compute L_sim = || ε_t − ε_θ(x_t, t) − σ̂_t^2 γ ∇_{x_t} log p_φ(y|x_t) ||^2
   e. Compute L_AN using ε⋆ and x_t⋆ similarly
   f. Backprop total loss with respect to ψ; θ frozen
   g. Log metrics, step optimizer, clip gradients

Notes on numerical stability
- ε is normalized to zero mean and unit variance after each inner-max update to maintain stable gradients.
- The term ∇_{x_t} log p_φ(y|x_t) can be small; the γ term ensures enough gradient signal to guide adaptation.

Cross-reference to code mappings
- Eq. 1: L_sample objective in ddpm.py/ldm.py
- Eq. 2: μ_θ context and denoising optimizations
- Eq. 5: L_sim implementation in losses.py
- Eq. 6–7: inner adversarial noise control in adversarial_noise.py
- Eq. 8: ε⋆ worst-case perturbation mapping used in L_AN
- Algorithm 1: ant_trainer.py orchestration

Notes on reproducibility and interpretation
- This appendix helps map the math to practical, auditable code paths. The exact numerical instantiations may vary slightly depending on backbone and data but preserve the overall structure and objective.