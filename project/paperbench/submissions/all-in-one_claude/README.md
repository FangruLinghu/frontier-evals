# Simformer: All-in-one Simulation-Based Inference

Implementation of the paper "All-in-one simulation-based inference" (Gloeckler et al., ICML 2024).

Simformer combines transformers with probabilistic diffusion models for amortized Bayesian inference from simulations.

## Features

- **Arbitrary Conditional Sampling**: Sample from any conditional p(θ_A, x_B | θ_C, x_D)
- **Structured Dependencies**: Exploit factorization structure via attention masks
- **Diffusion Guidance**: Incorporate constraints via interval guidance
- **9 Benchmark Tasks**: Comprehensive evaluation suite

## Installation

```bash
# Create conda environment
conda env create -f environment.yml
conda activate simformer

# Install package
pip install -e .
```

## Quick Start

### Training

```bash
# Train on Two Moons task
python scripts/train.py --task two_moons --epochs 100

# Train on Lotka-Volterra (complex task)
python scripts/train.py --task lotka_volterra --n_layers 8 --epochs 200
```

### Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint checkpoints/two_moons/final_model.pt \
    --task two_moons \
    --n_observations 100
```

### Run All Benchmarks

```bash
# Run all tasks
python scripts/run_benchmarks.py --all --epochs 100

# Run specific tasks
python scripts/run_benchmarks.py --tasks two_moons slcp tree
```

## Usage

### Posterior Inference

```python
import torch
from simformer import Simformer
from simformer.diffusion import VESDE, sample_posterior
from simformer.tasks import TwoMoonsTask

# Create task
task = TwoMoonsTask()

# Create model
model = Simformer(
    n_params=task.n_params,
    n_data=task.n_data,
    token_dim=50,
    n_layers=6,
    sde=VESDE(),
)

# Generate observation
theta_true = task.sample_prior(1)
x_obs = task.simulate(theta_true)

# Sample posterior
posterior_samples = sample_posterior(
    model, x_obs,
    n_samples=1000,
    n_steps=500,
)
```

### Arbitrary Conditional Sampling

```python
from simformer.diffusion import sample_arbitrary_conditional

# Define condition: fix θ_0 and x_0
condition_values = {"theta": {0: 1.5}, "x": {0: 0.3}}

# Sample remaining variables
samples = sample_arbitrary_conditional(
    model,
    condition_values,
    n_samples=1000,
)
```

### With Interval Guidance

```python
from simformer.diffusion.guidance import create_interval_guidance

# Constrain θ to [0, 2]
guidance = create_interval_guidance(
    param_bounds=torch.tensor([[0., 2.], [0., 2.]]),
    strength=1.0,
)

# Sample with guidance
samples = sample_posterior(
    model, x_obs,
    n_samples=1000,
    guidance=guidance,
)
```

## Architecture

### Model Components

- **SBI Tokenizer** (`simformer/tokenizer/`): Converts parameters and data to token sequences
- **Score Network** (`simformer/models/`): Time-conditioned transformer with adaptive layer norm
- **Attention Masks** (`simformer/masks/`): Encode dependency structure
- **Diffusion** (`simformer/diffusion/`): VESDE/VPSDE with reverse sampling

### Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Token dimension | 50 |
| Time embedding | 128-dim Gaussian Fourier |
| Transformer layers | 6 (simple) / 8 (complex) |
| Attention heads | 4 |
| MLP widening | 3x |
| VESDE σ_min | 0.0001 |
| VESDE σ_max | 15.0 |
| Batch size | 1000 |

## Benchmark Tasks

| Task | θ dim | x dim | Type |
|------|-------|-------|------|
| Gaussian Linear | 10 | 10 | Linear |
| Gaussian Mixture | 2 | 2 | Multi-modal |
| Two Moons | 2 | 2 | Non-linear |
| SLCP | 5 | 8 | Likelihood-free |
| Tree | 3 | 4 | Hierarchical |
| HMM | 10 | 10 | Sequential |
| Lotka-Volterra | 4 | 30 | ODE |
| SIRD | 22 | 30 | Function-valued |
| Hodgkin-Huxley | 7 | 8 | Complex ODE |

## Project Structure

```
simformer/
├── __init__.py
├── models/           # Neural network architectures
│   ├── transformer.py
│   ├── score_network.py
│   └── simformer.py
├── diffusion/        # SDE and sampling
│   ├── sde.py
│   ├── sampling.py
│   └── guidance.py
├── tokenizer/        # SBI tokenization
│   └── tokenizer.py
├── masks/            # Attention masks
│   └── attention_masks.py
├── training/         # Training infrastructure
│   ├── losses.py
│   └── trainer.py
├── evaluation/       # Metrics
│   ├── c2st.py
│   ├── coverage.py
│   └── metrics.py
├── tasks/            # Benchmark tasks
│   ├── base.py
│   ├── two_moons.py
│   └── ...
└── utils/            # Utilities
    └── helpers.py

scripts/
├── train.py          # Training script
├── evaluate.py       # Evaluation script
└── run_benchmarks.py # Run all benchmarks

configs/
└── default.yaml      # Default configuration
```

## Evaluation Metrics

- **C2ST**: Classifier Two-Sample Test (0.5 = perfect, 1.0 = distinguishable)
- **Expected Coverage**: Calibration of credible intervals
- **Calibration Error**: Mean |expected - empirical| coverage

## Hardware Requirements

- Tested on MacBook M3 Pro (MPS backend)
- Also supports CUDA and CPU
- Memory: ~8GB for training, ~4GB for inference

## Citation

```bibtex
@inproceedings{gloeckler2024allinone,
  title={All-in-one simulation-based inference},
  author={Gloeckler, Manuel and Deistler, Michael and Weilbach, Christian and Wood, Frank and Macke, Jakob H},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

## License

MIT License
