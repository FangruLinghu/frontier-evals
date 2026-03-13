# DPMs-ANT: Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning

Implementation of Wang et al., "Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning", ICML 2024.

## Method

DPMs-ANT addresses few-shot image generation by transferring pre-trained diffusion models to target domains with only 10 images. It introduces two strategies:

1. **Similarity-Guided Training**: Uses a binary classifier to estimate the gap between source and target domains, providing gradient guidance to the diffusion model during fine-tuning.

2. **Adversarial Noise Selection**: A min-max training process that dynamically selects the "worse-case" Gaussian noise, reducing required training iterations from ~5000 to ~300.

3. **Adaptor Layers**: Only fine-tunes small adaptor modules (1.3% of parameters) added to each U-Net layer, keeping the pre-trained model frozen.

## Installation

```bash
conda env create -f environment.yml
conda activate dpms_ant
pip install -e .
```

## Quick Start

### Toy 2D Experiment (Section 5.1)

```bash
python scripts/toy_experiment.py
```

This runs the 2D Gaussian transfer experiment and produces visualizations.

### Full Image Generation Pipeline

```bash
# 1. Download pre-trained models
python scripts/download_models.py --output-dir checkpoints/pretrained

# 2. Place 10-shot target images in datasets/<target>/
#    e.g., datasets/sunglasses/ with 10 sunglasses images

# 3. Train DPMs-ANT
python scripts/train.py \
    --target-dir datasets/sunglasses \
    --source-dir datasets/ffhq \
    --pretrained-model checkpoints/pretrained/ffhq_ddpm.pt \
    --method ant \
    --iterations 300

# 4. Evaluate
python scripts/evaluate.py \
    --model-checkpoint results/sunglasses_ant_*/checkpoints/final.pt \
    --target-dir datasets/sunglasses \
    --n-generated 1000
```

## Key Hyperparameters

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Guidance scale | gamma | 5.0 | Similarity guidance strength |
| AN steps | J | 10 | Adversarial noise gradient ascent steps |
| AN step size | omega | 0.02 | Adversarial noise learning rate |
| Learning rate | eta | 5e-5 (DDPM) / 1e-5 (LDM) | Adaptor learning rate |
| Iterations | - | 300 | Training iterations |
| Batch size | - | 40 | Training batch size |
| Adaptor c | c | 4 (DDPM) / 2 (LDM) | Spatial downscale factor |
| Adaptor d | d | 8 | Bottleneck dimension |

## Paper Results (10-shot)

### Intra-LPIPS (higher = more diversity)

| Method | Babies | Sunglasses | Raphael | Haunted | Landscape |
|--------|--------|------------|---------|---------|-----------|
| CDC | 0.583 | 0.581 | 0.564 | 0.620 | 0.674 |
| DCL | 0.579 | 0.574 | 0.558 | 0.616 | 0.626 |
| DDPM-PA | 0.599 | 0.604 | 0.581 | 0.628 | 0.706 |
| **DDPM-ANT** | **0.592** | **0.613** | **0.621** | **0.648** | **0.723** |
| **LDM-ANT** | **0.601** | **0.613** | **0.592** | **0.653** | **0.738** |

### FID (lower = better quality)

| Method | Babies | Sunglasses |
|--------|--------|------------|
| CDC | 74.39 | 42.13 |
| DCL | 52.56 | 38.01 |
| DDPM-PA | 48.92 | 34.75 |
| **ANT** | **46.70** | **20.06** |

## Project Structure

```
dpms_ant/
    models/
        unet.py                 # U-Net architecture (Dhariwal & Nichol 2021)
    adaptor/
        adaptor.py              # Adaptor layers (Houlsby et al. 2019)
    classifier/
        noise_classifier.py     # Binary classifier for similarity guidance
    diffusion/
        gaussian_diffusion.py   # DDPM/DDIM diffusion process
        adversarial_noise.py    # Adversarial noise selection (Eq 7)
    evaluation/
        metrics.py              # FID, Intra-LPIPS
    data/
        datasets.py             # Dataset loading utilities
    training.py                 # DPMs-ANT training loop (Algorithm 1)

scripts/
    toy_experiment.py           # 2D Gaussian experiment (Section 5.1)
    train.py                    # Main training script
    evaluate.py                 # Evaluation script
    download_models.py          # Download pre-trained models

configs/
    default.yaml                # Default configuration
```

## Algorithm (Algorithm 1 from paper)

```
Require: binary classifier p_phi, pre-trained DPMs eps_theta, learning rate eta
repeat
    x0 ~ q(x0)                          # Sample from target
    t ~ Uniform({1,...,T})                # Random timestep
    eps ~ N(0, I)                         # Initial noise
    for j = 0,...,J-1 do
        Update eps via Eq (7)             # Adversarial noise selection
    end for
    Compute L(psi) with eps* via Eq (8)   # Similarity-guided loss
    psi = psi - eta * grad_psi L(psi)     # Update adaptor only
until converged
```

## Citation

```bibtex
@inproceedings{wang2024bridging,
  title={Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning},
  author={Wang, Xiyu and Lin, Baijiong and Liu, Daochang and Chen, Ying-Cong and Xu, Chang},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
