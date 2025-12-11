"""
CapyFormer - A lightweight Transformer library for decision transformers
"""

__version__ = "0.1.0"

from capyformer.model import Transformer, TransformerInference, MaskedCausalAttention, Block
from capyformer.data import TrajectoryDataset, ModuleTrajectoryDataset
from capyformer.rnn_model import RNNModel
from capyformer.rnn_trainer import RNNTrainer

__all__ = [
    "Transformer",
    "TransformerInference",
    "MaskedCausalAttention",
    "Block",
    "TrajectoryDataset",
    "ModuleTrajectoryDataset",
    "RNNModel",
    "RNNTrainer",
]
