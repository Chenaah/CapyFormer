from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="capyformer",
    version="0.1.0",
    author="Chen Yu",
    description="A lightweight Transformer library for locomotion control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Chenaah/CapyFormer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "gymnasium>=0.26.0",
        "stable-baselines3>=2.0.0",
        "omegaconf>=2.1.0",
        "imageio>=2.9.0",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
        "ray": [
            "ray>=2.0.0",
        ],
        "wandb": [
            "wandb>=0.12.0",
        ],
    },
)
