from setuptools import setup, find_packages

setup(
    name="simformer",
    version="0.1.0",
    description="All-in-one simulation-based inference with Simformer",
    author="Implementation based on Gloeckler et al. (2024)",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy",
        "scipy",
        "matplotlib",
        "seaborn",
        "pandas",
        "scikit-learn",
        "tqdm",
        "pyyaml",
        "einops",
        "hydra-core",
        "omegaconf",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
            "flake8",
        ],
        "wandb": ["wandb"],
    },
)
