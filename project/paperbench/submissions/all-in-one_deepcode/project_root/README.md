All-in-one diffusion-based SBI with SimFormer (Transformer-driven score diffusion)

Overview
- This repository implements a modular, config-driven diffusion-based probabilistic inference framework for jointly inferring parameters theta and observations x (p(theta, x)). It follows the approach described in the reproduction plan: a transformer-based model ( SimFormer ) learns the joint distribution p(theta, x) from simulator data and supports sampling of all conditionals (posterior, likelihood, and arbitrary conditioning).
- Core components include tokenization and embeddings, ME-enabled attention masks, a transformer encoder, forward diffusion models ( VPSDE / VESDE ), a score network, diffusion guidance, a forward/backward sampler, training loop with MC masking, and evaluation utilities. Data simulators cover toy and scientific tasks (Gaussian Linear, Two Moons, SLCP, Lotka-Volterra, SIR, Hodgkin-Huxley, Gravitational Waves, etc.).

What’s included
- src/tokenizer.py: Joint tokenizer for theta and x with per-variable conditioning flag (MC).
- src/embeddings.py: Lightweight deterministic embeddings provider for tokens.
- src/mask_utils.py: ME (multi-expert) attention mask utilities and dynamic adaptation.
- src/transformer_model.py: Lightweight NumPy-based Transformer encoder with optional ME-enabled attention.
- src/diffusion_core.py: VPSDE / VESDE forward diffusion, beta schedules, and time encoding utilities.
- src/score_network.py: Small deterministic score head with a simple time embedding.
- src/guidance.py: Diffusion guidance module for enforcing constraints during sampling.
- src/sampling.py: Reverse diffusion sampler (Euler–Maruyama) with conditioning support.
- src/training_loop.py: Minimal training runner wiring data generation, forward diffusion characteristics, and a simple score-matching loss.
- src/evaluation.py: Lightweight evaluation utilities for C2ST, NLL proxy, calibration, and posterior predictive checks.
- src/utilities.py, src/appendices_helpers.py: RNG, seeding, config helpers, and reproducibility utilities.
- src/data_samplers/: A suite of toy and science-task simulators (gaussian_linear.py, gaussian_mixture.py, two_moons.py, slcp.py, lotka_volterra.py, sir.py, hodgkin_huxley.py, gravitational_waves.py).
- configs/: YAML configurations per task and experiment (base.yaml, guided_diffusion.yaml, tasks/*, model_specs/simformer.yaml).
- scripts/: Orchestration and experiment pipelines (loading configs, running training, sampling, evaluation).
- results/: Outputs and logs from experiments.

Getting started
- Prerequisites
  - Python 3.10+ (3.11 recommended)
  - Basic scientific Python stack: numpy, scipy, matplotlib, etc.
  - YAML parsing (PyYAML) for config files.
  - Optional: JAX/Flax/Haiku and Optax if you replace the NumPy-based components with a JAX implementation. The current repo uses NumPy-based scaffolding suitable for quick demonstrations and unit tests.
- Environment setup (example)
  - Create a clean project env (conda or venv).
  - Install dependencies (numpy, scipy, matplotlib, pyyaml, tqdm, etc.). See guidance in docs or a provided environment.yml if available.
- Reproducibility
  - All simulations and model initializations support deterministic seeding via seed_everything in appendices_helpers.py and utilities.py. Use the seeds in the per-task YAML configs to reproduce runs.

Running a minimal, toy demo
- This repo is config-driven. Start from an existing per-task config (Gaussian Linear or Two Moons) under configs/tasks.
- Example quick-start path (toy tasks):
  - Gaussian Linear task: configs/tasks/gaussian_linear.yaml
  - Two Moons task: configs/tasks/two_moons.yaml
- The recommended workflow is to use the project’s experiment runner which loads the base.yaml, merges in a per-task YAML, and executes training, sampling, and evaluation steps. The exact script/entrypoint may vary in your checkout; look for a script under scripts/ that loads and runs a config file.

Config-driven workflow (high level)
- Base config (configs/base.yaml) contains global defaults for diffusion, training, and model setup.
- Task configs (configs/tasks/*.yaml) override and extend base defaults for a specific experiment (task_id, data samplers, diffusion settings, guidance, training hyperparameters, evaluation metrics, and runtime options).
- Model specs (configs/model_specs/simformer.yaml) define the transformer-based joint-score model (token dimensions, time embeddings, ME masking options, etc.).
- The config system wires: diffusion_core (SDE forward pass), sampling (reverse diffusion with conditioning), guidance (constraint-guided sampling), training_loop (MC masking and score-matching loss), and evaluation (C2ST, NLL proxy, calibration).

Extending and contributing
- To add a new task or a new data-sampler, implement a new module under src/data_samplers/ with the expected API: sample_prior, simulate, log_likelihood, get_mixture_weights. Mirror the interface used by existing samplers (gaussian_linear, two_moons, sir, lotka_volterra, hodgkin_huxley, etc.).
- To extend the diffusion model, update or add components in: src/diffusion_core.py, src/guidance.py, src/sampling.py, and src/transformer_model.py as needed. All modules are designed to be independent and testable.
- Update or add YAML configs under configs/ to configure new experiments. Follow the structure of existing files like guided_diffusion.yaml or slcp.yaml.

Testing and validation
- Unit tests exist under tests/ validating individual modules (tokenizer, embeddings, mask_utils, transformer_model, diffusion_core, score_network, sampling, guidance, training_loop, evaluation).
- Run tests with pytest to quickly verify components:
  - pytest -q tests/

Notes
- This repository emphasizes a minimal, test-friendly implementation suitable for unit tests and demonstration experiments. It intentionally skips heavy deep-learning frameworks in favor of approachable NumPy-based components for deterministic behavior and rapid iteration.

License
- This README provides a project overview and usage guidance. Replace with your preferred license header as needed.
