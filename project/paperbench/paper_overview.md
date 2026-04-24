# PaperBench Paper Overview

## Summary

23 papers total in the benchmark. Categorized by research focus below.

## Papers by Category

### Model Improvement (Efficiency / Robustness / Adaptation)

| Paper ID | Title | Framework | Sub-tasks | Points | GitHub |
|----------|-------|-----------|-----------|--------|--------|
| adaptive-pruning | APT: Adaptive Pruning and Tuning Pretrained Language Models for Efficient Training and Inference | PyTorch | 171 | 221 | [ROIM1998/APT](https://github.com/ROIM1998/APT) |
| bbox | BBox-Adapter: Lightweight Adapting for Black-Box Large Language Models | PyTorch | 421 | 452 | [haotiansun14/BBox-Adapter](https://github.com/haotiansun14/BBox-Adapter) |
| bridging-data-gaps | Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning | PyTorch | 206 | 252 | [ShinyGua/DPMs-ANT](https://github.com/ShinyGua/DPMs-ANT) |
| robust-clip | Robust CLIP: Unsupervised Adversarial Fine-Tuning of Vision Embeddings for Robust Large Vision-Language Models | PyTorch | 145 | 164 | [chs20/RobustVLM](https://github.com/chs20/RobustVLM) |
| sample-specific-masks | Sample-specific Masks for Visual Reprogramming-based Prompting | PyTorch | 395 | 470 | [tmlr-group/SMM](https://github.com/tmlr-group/SMM) |
| test-time-model-adaptation | Test-Time Model Adaptation with Only Forward Passes | PyTorch | 235 | 271 | [mr-eggplant/FOA](https://github.com/mr-eggplant/FOA) |

### Continual Learning / Forgetting

| Paper ID | Title | Framework | Sub-tasks | Points | GitHub |
|----------|-------|-----------|-----------|--------|--------|
| ftrl | Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem | PyTorch | 232 | 256 | [BartekCupial/finetuning-RL-as-CL](https://github.com/BartekCupial/finetuning-RL-as-CL) |
| self-expansion | Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning | TensorFlow | 362 | 393 | *(no public repo)* |
| self-composing-policies | Self-Composing Policies for Scalable Continual Reinforcement Learning | unknown | — | — | — |
| what-will-my-model-forget | What Will My Model Forget? Forecasting Forgotten Examples in Language Model Refinement | likely PyTorch | 1,145 | 1,162 | [AuCson/icml-24-wwmf-temp](https://github.com/AuCson/icml-24-wwmf-temp) |

### Analysis / Interpretability / Benchmarking

| Paper ID | Title | Framework | Sub-tasks | Points | GitHub |
|----------|-------|-----------|-----------|--------|--------|
| mechanistic-understanding | A Mechanistic Understanding of Alignment Algorithms: A Case Study on DPO and Toxicity | PyTorch | 127 | 152 | [ajyl/dpo_toxic](https://github.com/ajyl/dpo_toxic) |
| lca-on-the-line | LCA-on-the-Line: Benchmarking Out-of-Distribution Generalization with Class Taxonomies | unknown | — | — | — |
| pinn | Challenges in Training PINNs: A Loss Landscape Perspective | unknown | — | — | — |

### Reinforcement Learning Methods

| Paper ID | Title | Framework | Sub-tasks | Points | GitHub |
|----------|-------|-----------|-----------|--------|--------|
| fre | Unsupervised Zero-Shot Reinforcement Learning via Functional Reward Encodings | unknown | — | — | — |
| rice | RICE: Breaking Through the Training Bottlenecks of Reinforcement Learning with Explanation | unknown | — | — | — |
| sapg | SAPG: Split and Aggregate Policy Gradients | unknown | — | — | — |

### LLM Reasoning / Generation

| Paper ID | Title | Framework | Sub-tasks | Points | GitHub |
|----------|-------|-----------|-----------|--------|--------|
| semantic-self-consistency | Semantic Self-Consistency: Enhancing Language Model Reasoning via Semantic Weighting | unknown | 99 | 126 | *(no public repo)* |
| stay-on-topic-with-classifier-free-guidance | Stay on Topic with Classifier-Free Guidance | PyTorch-adjacent (HF) | 185 | 227 | [Vermeille/lm-evaluation-harness-cfg](https://github.com/Vermeille/lm-evaluation-harness-cfg) |

### Statistical Inference / Generative Modeling

| Paper ID | Title | Framework | Sub-tasks | Points | GitHub |
|----------|-------|-----------|-----------|--------|--------|
| all-in-one | All-in-one simulation-based inference | unknown | — | — | — |
| bam | Batch and Match: black-box variational inference with a score-based divergence | unknown | — | — | — |
| sequential-neural-score-estimation | Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models | unknown | — | — | — |
| stochastic-interpolants | Stochastic Interpolants with Data-Dependent Couplings | unknown | — | — | — |

### Data Selection

| Paper ID | Title | Framework | Sub-tasks | Points | GitHub |
|----------|-------|-----------|-----------|--------|--------|
| lbcs | Refined Coreset Selection: Towards Minimal Coreset Size under Model Performance Constraints | unknown | — | — | — |

## Reproducibility Quick Reference (Model Improvement papers, sorted by difficulty)

1. **robust-clip** — 145 tasks, 164 pts — adversarial fine-tuning of CLIP for VLM robustness (needs large GPU)
2. **adaptive-pruning** — 171 tasks, 221 pts — pruning/tuning RoBERTa and T5 for efficiency
3. **bridging-data-gaps** — 206 tasks, 252 pts — transfer learning for diffusion models
4. **test-time-model-adaptation** — 235 tasks, 271 pts — forward-pass-only adaptation at test time
5. **sample-specific-masks** — 395 tasks, 470 pts — visual reprogramming masks
6. **bbox** — 421 tasks, 452 pts — black-box LLM adaptation
