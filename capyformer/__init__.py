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
    from capyformer.hf_trainer import (
        HFTrainer, 
        HFTrajectoryModel, 
        HFTransformerInference,
        HFFlowMatchingInference,
        FlowMatchingHead,
        # Action Chunking (non-autoregressive, MPC-style like pi0)
        HFActionChunkingTrainer,
        HFActionChunkingModel,
        ActionChunkingInference,
        ActionChunkingHead,
    )
    HAS_HF_TRAINER = True
except ImportError:
    HAS_HF_TRAINER = False

# Sanity check utilities (optional import to avoid dependencies)
try:
    from capyformer.sanity_check import (
        ToyLinearDataset,
        ToyModularDataset,
        ToyPeriodicDataset,
        ToyEnvironment,
        ToyModularEnvironment,
    )
    HAS_SANITY_CHECK = True
except ImportError:
    HAS_SANITY_CHECK = False

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
    "HFFlowMatchingInference",
    "FlowMatchingHead",
    # Action Chunking (non-autoregressive, MPC-style)
    "HFActionChunkingTrainer",
    "HFActionChunkingModel",
    "ActionChunkingInference",
    "ActionChunkingHead",
    "HAS_HF_TRAINER",
    # Sanity check utilities
    "ToyLinearDataset",
    "ToyModularDataset",
    "ToyPeriodicDataset",
    "ToyEnvironment",
    "ToyModularEnvironment",
    "HAS_SANITY_CHECK",
]
