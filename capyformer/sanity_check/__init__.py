"""
Sanity check utilities for CapyFormer transformer policy training.

This module provides toy datasets and environments to verify that the
transformer policy training pipeline is working correctly.

Usage:
    from capyformer.sanity_check import (
        ToyLinearDataset,
        ToyModularDataset,
        ToyEnvironment,
        ToyModularEnvironment,
    )
"""

from .toy_datasets import (
    ToyLinearDataset,
    ToyModularDataset,
    ToyPeriodicDataset,
)
from .toy_environments import (
    ToyEnvironment,
    ToyModularEnvironment,
)

__all__ = [
    "ToyLinearDataset",
    "ToyModularDataset",
    "ToyPeriodicDataset",
    "ToyEnvironment",
    "ToyModularEnvironment",
]

