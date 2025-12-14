"""
CapyFormer - A lightweight Transformer library for decision transformers
"""

__version__ = "0.1.0"

from capyformer.model import Transformer, TransformerInference, MaskedCausalAttention, Block
from capyformer.data import TrajectoryDataset, ModuleTrajectoryDataset
from capyformer.trainer import Trainer
from capyformer.rnn_model import RNNModel
from capyformer.rnn_trainer import RNNTrainer

# HuggingFace trainer (optional, requires transformers package)
try:
    from capyformer.hf_trainer import HFTrainer, HFTrajectoryModel, HFTransformerInference
    HAS_HF_TRAINER = True
except ImportError:
    HAS_HF_TRAINER = False

__all__ = [
    "Transformer",
    "TransformerInference",
    "MaskedCausalAttention",
    "Block",
    "TrajectoryDataset",
    "ModuleTrajectoryDataset",
    "Trainer",
    "RNNModel",
    "RNNTrainer",
    # HuggingFace trainer (conditional)
    "HFTrainer",
    "HFTrajectoryModel",
    "HFTransformerInference",
    "HAS_HF_TRAINER",
]
