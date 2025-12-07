# CapyFormer

**Status:** 🚧 Under Development

CapyFormer is a lightweight Transformer library originally developed for the [ModularLegs project](https://modularlegs.github.io/) by Chen Yu. The code is now being separated from the main codebase ([modularlegs GitHub repository](https://github.com/Chenaah/modularlegs)) to create an easy-to-use Transformer library for decision transformers.

## Installation

### Install from source

```bash
# Clone the repository
git clone https://github.com/Chenaah/CapyFormer.git
cd CapyFormer

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[ray,wandb,dev]"
```

### Install from PyPI (coming soon)

```bash
pip install capyformer
```

## Quick Start

```python
from capyformer import Transformer, TrajectoryDataset

# Create a transformer model
model = Transformer(
    state_dim=17,
    act_dim=6,
    n_blocks=3,
    h_dim=128,
    context_len=20,
    n_heads=1,
    drop_p=0.1,
)

# Load trajectory data
dataset = TrajectoryDataset(...)

# Train your model
# See examples/ for more details
```

## Features

- **Lightweight**: Simple and easy to understand implementation
- **Flexible**: Support for various decision transformer architectures
- **Modular**: Easy to extend and customize

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- Gymnasium >= 0.26.0
- Stable-Baselines3 >= 2.0.0

## Documentation

Documentation is under development. Please check the `examples/` directory for usage examples.

## License

MIT License (add LICENSE file as needed)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
