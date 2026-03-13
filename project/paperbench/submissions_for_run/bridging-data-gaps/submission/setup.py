from setuptools import setup, find_packages

setup(
    name="dpms_ant",
    version="0.1.0",
    description="DPMs-ANT: Bridging Data Gaps in Diffusion Models with Adversarial Noise-Based Transfer Learning",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "Pillow>=10.0.0",
        "tqdm>=4.65.0",
        "lpips>=0.1.4",
        "matplotlib>=3.7.0",
        "blobfile>=2.0.0",
        "PyYAML>=6.0",
        "scikit-learn>=1.3.0",
        "diffusers>=0.25.0",
        "accelerate>=0.25.0",
        "transformers>=4.36.0",
    ],
)
