"""
CapyFormer - A lightweight Transformer library for decision transformers
"""

__version__ = "0.1.0"

from capyformer.model import Transformer, MaskedCausalAttention, Block
from capyformer.data import TrajectoryDataset, ModuleTrajectoryDataset

__all__ = [
    "Transformer",
    "MaskedCausalAttention",
    "Block",
    "TrajectoryDataset",
    "ModuleTrajectoryDataset",
]
