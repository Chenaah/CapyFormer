"""
HuggingFace Transformer Trainer for Decision Transformer style training.

This module provides a trainer that uses HuggingFace's pretrained models (like Gemma, LLaMA, etc.)
as the backbone for trajectory prediction tasks, while maintaining the same API as the regular Trainer.

The key idea is to:
1. Project continuous input tokens (states) into the HF model's embedding space
2. Use the HF model's transformer layers for sequence modeling
3. Project the output back to continuous action/target space

Usage is identical to the regular Trainer:

```python
class MyDataset(TrajectoryDataset):
    def _setup_dataset(self, dataset_config):
        self.trajectories = load_my_data()
        self.input_keys = ["position", "velocity"]
        self.target_key = "actions"

dataset = MyDataset({"val_split": 0.1}, context_len=20)

trainer = HFTrainer(
    dataset,
    model_name="google/gemma-2b",  # or any HF causal LM
    log_dir="./logs",
    use_lora=True,  # Efficient fine-tuning
)
trainer.learn(n_epochs=1000)

# Inference
policy = trainer.get_inference()
action = policy.step({"position": pos, "velocity": vel})
```
"""

import copy
import csv
import os
import random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

try:
    from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
    from transformers import BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

from capyformer.data import TrajectoryDataset


class ContinuousEmbedding(nn.Module):
    """
    Projects continuous values to the embedding space of the HF model.
    
    Unlike discrete token embeddings, this learns a linear projection
    from continuous input dimensions to the model's hidden dimension.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, input_dim)
        Returns:
            Embedded tensor of shape (B, T, hidden_dim)
        """
        return self.norm(self.proj(x))


class ContinuousHead(nn.Module):
    """
    Projects hidden states back to continuous output space.
    """
    
    def __init__(self, hidden_dim: int, output_dim: int, use_tanh: bool = False):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, output_dim)
        self.use_tanh = use_tanh
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Hidden states of shape (B, T, hidden_dim)
        Returns:
            Output tensor of shape (B, T, output_dim)
        """
        out = self.proj(x)
        if self.use_tanh:
            out = torch.tanh(out)
        return out


class HFTrajectoryModel(nn.Module):
    """
    Wrapper around HuggingFace causal language models for trajectory prediction.
    
    Architecture:
    - Input projection layers: project each continuous input token to embedding space
    - HF backbone: pretrained transformer (optionally with LoRA)
    - Output projection: project hidden states to target dimension
    
    Sequence format (auto-regressive):
    [input1_t0, input2_t0, ..., target_t0, input1_t1, input2_t1, ..., target_t1, ...]
    
    The model predicts the target at each timestep based on the input tokens.
    """
    
    input_token_names: List[str]
    
    def __init__(
        self,
        hf_model: nn.Module,
        input_token_dims: List[int],
        input_token_names: List[str],
        target_dim: int,
        hidden_dim: int,
        use_action_tanh: bool = False,
        state_mean: Optional[Dict[str, torch.Tensor]] = None,
        state_std: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        
        self.hf_model = hf_model
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.input_token_names = input_token_names
        self.num_input_tokens = len(input_token_dims)
        
        # Store dimensions as buffer for serialization
        self.register_buffer('input_token_dims', torch.tensor(input_token_dims, dtype=torch.long))
        
        # Input embeddings for each token type
        self.input_embeddings = nn.ModuleDict()
        for name, dim in zip(input_token_names, input_token_dims):
            self.input_embeddings[name] = ContinuousEmbedding(dim, hidden_dim)
        
        # Embedding for previous target (fed back auto-regressively)
        self.target_embedding = ContinuousEmbedding(target_dim, hidden_dim)
        
        # Output projection
        self.output_head = ContinuousHead(hidden_dim, target_dim, use_tanh=use_action_tanh)
        
        # Store normalization stats
        self._state_mean_dict = None
        self._state_std_dict = None
        
        if state_mean is not None:
            self._state_mean_dict = {
                key: torch.as_tensor(value, dtype=torch.float32) 
                for key, value in state_mean.items()
            }
        
        if state_std is not None:
            self._state_std_dict = {
                key: torch.as_tensor(value, dtype=torch.float32)
                for key, value in state_std.items()
            }
    
    @property
    def state_mean(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._state_mean_dict
    
    @property
    def state_std(self) -> Optional[Dict[str, torch.Tensor]]:
        return self._state_std_dict
    
    @property
    def act_dim(self) -> int:
        """Alias for backward compatibility."""
        return self.target_dim
    
    @property
    def state_token_dims(self) -> List[int]:
        """Alias for backward compatibility."""
        return [int(d.item()) for d in self.input_token_dims]
    
    @property
    def state_token_names(self) -> List[str]:
        """Alias for backward compatibility."""
        return self.input_token_names
    
    def forward(
        self,
        timesteps: torch.Tensor,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        state_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass matching the original Transformer API.
        
        Args:
            timesteps: (B, T) timestep indices (unused, kept for API compatibility)
            states: Dict[str, (B, T, dim)] input state tokens
            actions: (B, T, target_dim) previous actions (for auto-regressive)
            state_mask: Optional Dict[str, (B, T)] mask for missing tokens
        
        Returns:
            Tuple of (None, action_predictions, None) to match original API
        """
        B, T, _ = actions.shape
        device = actions.device
        
        # Build sequence: [input1, input2, ..., target] for each timestep
        # Total tokens per timestep = num_input_tokens + 1 (for target)
        tokens_per_step = self.num_input_tokens + 1
        
        # Embed all input tokens
        embeddings_list = []
        attention_masks = []
        
        for t in range(T):
            # Embed each input token for this timestep
            for name in self.input_token_names:
                state_t = states[name][:, t:t+1, :]  # (B, 1, dim)
                emb = self.input_embeddings[name](state_t)  # (B, 1, hidden)
                embeddings_list.append(emb)
                
                # Handle masking for missing tokens
                if state_mask is not None and name in state_mask:
                    mask_t = state_mask[name][:, t:t+1]  # (B, 1)
                else:
                    mask_t = torch.ones(B, 1, device=device, dtype=torch.bool)
                attention_masks.append(mask_t)
            
            # Embed previous action (zero for first timestep)
            action_t = actions[:, t:t+1, :].nan_to_num(0)  # (B, 1, target_dim)
            action_emb = self.target_embedding(action_t)  # (B, 1, hidden)
            embeddings_list.append(action_emb)
            attention_masks.append(torch.ones(B, 1, device=device, dtype=torch.bool))
        
        # Concatenate all embeddings: (B, T * tokens_per_step, hidden)
        hidden_states = torch.cat(embeddings_list, dim=1)
        attention_mask = torch.cat(attention_masks, dim=1).float()  # (B, T * tokens_per_step)
        
        # Pass through HF model
        # Most HF models expect input_ids, but we're using inputs_embeds
        outputs = self.hf_model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get last hidden states
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
        else:
            # Some models return hidden_states tuple
            last_hidden = outputs.hidden_states[-1]
        
        # Extract hidden states at positions just before target tokens
        # Target prediction happens after seeing all input tokens
        # Positions: [inp1, inp2, ..., TARGET, inp1, inp2, ..., TARGET, ...]
        # We predict from position (num_input_tokens - 1) for each timestep
        target_positions = []
        for t in range(T):
            # Position of last input token before target
            pos = t * tokens_per_step + (self.num_input_tokens - 1)
            target_positions.append(pos)
        
        # Gather hidden states at target positions
        target_hidden = last_hidden[:, target_positions, :]  # (B, T, hidden)
        
        # Project to action space
        action_preds = self.output_head(target_hidden)  # (B, T, target_dim)
        
        return None, action_preds, None
    
    def get_inference(self, device: str = None, context_len: int = None) -> 'HFTransformerInference':
        """Create an inference wrapper for step-by-step prediction."""
        if device is None:
            device = next(self.parameters()).device
        return HFTransformerInference(self, device=str(device), context_len=context_len)


class HFTransformerInference:
    """
    Inference wrapper for HFTrajectoryModel.
    
    Manages context window and state history automatically.
    Compatible with the original TransformerInference API.
    """
    
    def __init__(self, model: HFTrajectoryModel, device: str = 'cpu', context_len: int = None):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.input_token_names = model.input_token_names
        self.input_token_dims = [int(d.item()) for d in model.input_token_dims]
        self.target_dim = model.target_dim
        
        self.state_mean = model.state_mean
        self.state_std = model.state_std
        
        # Default context length if not specified
        self.context_len = context_len if context_len is not None else 20
        
        self.reset()
    
    def reset(self):
        """Reset history for new episode."""
        self.timestep = 0
        self.state_history = {name: [] for name in self.input_token_names}
        self.mask_history = {name: [] for name in self.input_token_names}
        self.action_history = []
    
    def _normalize_state(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Normalize state using stored statistics."""
        if self.state_mean is not None and self.state_std is not None:
            if name in self.state_mean and name in self.state_std:
                mean = self.state_mean[name].to(value.device)
                std = self.state_std[name].to(value.device)
                return (value - mean) / std
        return value
    
    @torch.no_grad()
    def step(self, current_state: dict, return_numpy: bool = True):
        """
        Process current state and return predicted output.
        
        Args:
            current_state: Dict mapping input token names to values (numpy or tensor)
            return_numpy: If True, return numpy array
        
        Returns:
            Predicted target (e.g., action) as numpy array or tensor
        """
        # Process and store current state
        for idx, name in enumerate(self.input_token_names):
            is_present = name in current_state and current_state[name] is not None
            
            if is_present:
                value = current_state[name]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float()
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32)
                
                value = self._normalize_state(name, value)
                self.mask_history[name].append(torch.tensor(True))
            else:
                token_dim = self.input_token_dims[idx]
                value = torch.zeros(token_dim, dtype=torch.float32)
                self.mask_history[name].append(torch.tensor(False))
            
            self.state_history[name].append(value)
        
        # Prepare context
        n_steps = min(len(self.action_history) + 1, self.context_len)
        
        # Build states dict
        states = {}
        state_mask = {}
        for name in self.input_token_names:
            state_list = self.state_history[name][-n_steps:]
            mask_list = self.mask_history[name][-n_steps:]
            states[name] = torch.stack(state_list, dim=0).unsqueeze(0).to(self.device)
            state_mask[name] = torch.stack(mask_list, dim=0).unsqueeze(0).to(self.device)
        
        # Build actions (previous predictions)
        if len(self.action_history) >= n_steps:
            actions_list = self.action_history[-n_steps:]
        else:
            actions_list = self.action_history.copy()
            while len(actions_list) < n_steps:
                actions_list.insert(0, torch.zeros(self.target_dim))
        
        actions = torch.stack(actions_list, dim=0).unsqueeze(0).to(self.device)
        timesteps = torch.arange(n_steps).unsqueeze(0).to(self.device)
        
        # Forward pass
        _, action_preds, _ = self.model.forward(timesteps, states, actions, state_mask)
        
        # Get prediction for current timestep
        predicted = action_preds[0, -1]
        
        # Store for next step
        self.action_history.append(predicted.cpu())
        self.timestep += 1
        
        # Trim history
        max_history = self.context_len * 2
        if len(self.action_history) > max_history:
            self.action_history = self.action_history[-max_history:]
            for name in self.input_token_names:
                self.state_history[name] = self.state_history[name][-max_history:]
                self.mask_history[name] = self.mask_history[name][-max_history:]
        
        if return_numpy:
            return predicted.cpu().numpy()
        return predicted


class HFTrainer:
    """
    Trainer using HuggingFace pretrained models for trajectory prediction.
    
    This trainer wraps HuggingFace causal language models (like Gemma, LLaMA)
    and adapts them for continuous state/action prediction while maintaining
    the same API as the regular Trainer.
    
    Key features:
    - Uses pretrained HF models as backbone
    - Supports LoRA for efficient fine-tuning
    - Same API as regular Trainer (input_keys, target_key)
    - Auto-regressive sequence modeling
    
    Args:
        dataset: TrajectoryDataset instance
        model_name: HuggingFace model name (e.g., "google/gemma-2b")
        log_dir: Directory for logs and checkpoints
        use_lora: Whether to use LoRA for efficient fine-tuning
        lora_r: LoRA rank
        lora_alpha: LoRA alpha parameter
        freeze_backbone: Whether to freeze the HF model backbone
        load_in_8bit: Whether to load model in 8-bit precision
        use_action_tanh: Whether to apply tanh to outputs
        batch_size: Training batch size
        device: Training device
        learning_rate: Learning rate
        wt_decay: Weight decay
        warmup_steps: Number of warmup steps
        validation_freq: Validation frequency (epochs)
        action_is_velocity: Whether actions represent velocities
        dt: Time step for velocity integration
    """
    
    def __init__(
        self,
        dataset: TrajectoryDataset,
        model_name: str = "google/gemma-2b",
        log_dir: str = "./logs",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_backbone: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_action_tanh: bool = False,
        batch_size: int = 32,
        device: str = "cuda:0",
        learning_rate: float = 1e-4,
        wt_decay: float = 0.01,
        warmup_steps: int = 1000,
        seed: int = 0,
        validation_freq: int = 100,
        validation_trajectories: int = 10,
        action_is_velocity: bool = True,
        dt: float = 0.02,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "HuggingFace Transformers not installed. "
                "Please install: pip install transformers"
            )
        
        if use_lora and not HAS_PEFT:
            raise ImportError(
                "PEFT not installed but use_lora=True. "
                "Please install: pip install peft"
            )
        
        self.traj_dataset = dataset
        self.model_name = model_name
        self.log_dir = log_dir
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.freeze_backbone = freeze_backbone
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_action_tanh = use_action_tanh
        self.batch_size = batch_size
        self.device = device
        self.learning_rate = learning_rate
        self.wt_decay = wt_decay
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.validation_freq = validation_freq
        self.validation_trajectories = validation_trajectories
        self.action_is_velocity = action_is_velocity
        self.dt = dt
        
        # Get dataset properties
        self.act_dim = dataset.act_dim
        self.state_token_dims = dataset.state_token_dims
        self.state_token_names = dataset.state_token_names
        self.context_len = dataset.context_len
        
        self.model = None
        
        os.makedirs(self.log_dir, exist_ok=True)
    
    def _create_model(self) -> HFTrajectoryModel:
        """Create the HF-based trajectory model."""
        print(f"Loading HuggingFace model: {self.model_name}")
        
        # Configure quantization if requested
        quantization_config = None
        if self.load_in_8bit or self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        # Load HF model



        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if (self.load_in_8bit or self.load_in_4bit) else torch.float32,
            device_map="auto" if (self.load_in_8bit or self.load_in_4bit) else None,
            trust_remote_code=True,
        )
        
        # Get hidden dimension from config
        config = hf_model.config
        hidden_dim = config.hidden_size
        print(f"HF model hidden dimension: {hidden_dim}")
        
        # Apply LoRA if requested
        if self.use_lora:
            print(f"Applying LoRA (r={self.lora_r}, alpha={self.lora_alpha})")
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Common attention modules
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            hf_model = get_peft_model(hf_model, lora_config)
            hf_model.print_trainable_parameters()
        elif self.freeze_backbone:
            # Freeze all backbone parameters
            for param in hf_model.parameters():
                param.requires_grad = False
        
        # Get normalization stats
        state_mean, state_std = self.traj_dataset.get_state_stats()
        
        # Create wrapper model
        model = HFTrajectoryModel(
            hf_model=hf_model,
            input_token_dims=self.state_token_dims,
            input_token_names=self.state_token_names,
            target_dim=self.act_dim,
            hidden_dim=hidden_dim,
            use_action_tanh=self.use_action_tanh,
            state_mean=state_mean,
            state_std=state_std,
        )
        
        return model
    
    def get_inference(self, device: str = None):
        """Get inference wrapper for step-by-step prediction."""
        if self.model is None:
            raise RuntimeError("No model available. Call learn() first.")
        
        if device is None:
            device = self.device
        
        return self.model.get_inference(device=device, context_len=self.context_len)
    
    def save(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model available. Call learn() first.")
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        if path.endswith(".pt"):
            path = path.rsplit(".", 1)[0]
        
        checkpoint_path = f"{path}.pt"
        
        # Prepare state mean/std for saving
        state_mean = self.model.state_mean
        state_std = self.model.state_std
        
        if state_mean is not None:
            state_mean = {k: v.cpu() for k, v in state_mean.items()}
        if state_std is not None:
            state_std = {k: v.cpu() for k, v in state_std.items()}
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "model_name": self.model_name,
                "input_token_dims": self.state_token_dims,
                "input_token_names": self.state_token_names,
                "target_dim": self.act_dim,
                "context_len": self.context_len,
                "use_action_tanh": self.use_action_tanh,
                "state_mean": state_mean,
                "state_std": state_std,
            },
            "trainer_config": {
                "use_lora": self.use_lora,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "action_is_velocity": self.action_is_velocity,
                "dt": self.dt,
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load(self, path: str, device: str = None):
        """Load model from checkpoint."""
        if device is None:
            device = self.device
        
        if not path.endswith(".pt"):
            path = f"{path}.pt"
        
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=device)
        
        # Recreate model
        config = checkpoint["model_config"]
        self.model_name = config["model_name"]
        
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        
        # Update trainer config
        trainer_config = checkpoint.get("trainer_config", {})
        self.action_is_velocity = trainer_config.get("action_is_velocity", True)
        self.dt = trainer_config.get("dt", 0.02)
        
        print(f"Model loaded successfully")
        return self
    
    def validate_rollout(self, model, device):
        """Validate by performing trajectory rollouts."""
        model.eval()
        total_mse = 0.0
        num_predictions = 0
        
        inference = model.get_inference(device=device, context_len=self.context_len)
        
        with torch.no_grad():
            for _ in range(self.validation_trajectories):
                traj_idx = random.randint(0, len(self.traj_dataset.val_trajectories) - 1)
                traj = self.traj_dataset.val_trajectories[traj_idx]
                
                inputs = self.traj_dataset.get_inputs(traj)
                target_gt = self.traj_dataset.get_target(traj)
                traj_len = min(self.traj_dataset.get_traj_len(traj), 200)
                
                if traj_len < 2:
                    continue
                
                inference.reset()
                
                predicted_actions = []
                for t in range(traj_len - 1):
                    current_state = {}
                    for key in inputs.keys():
                        state_value = inputs[key][t]
                        if np.all(np.isnan(state_value)):
                            current_state[key] = None
                        else:
                            current_state[key] = np.nan_to_num(state_value, nan=0.0)
                    
                    pred_action = inference.step(current_state, return_numpy=False)
                    predicted_actions.append(pred_action.cpu())
                
                if len(predicted_actions) > 0:
                    predicted_actions = torch.stack(predicted_actions)
                    actual_targets = torch.from_numpy(target_gt[:len(predicted_actions)]).float()
                    
                    valid_mask = ~torch.isnan(actual_targets)
                    if valid_mask.any():
                        pred_masked = predicted_actions[valid_mask]
                        target_masked = actual_targets[valid_mask]
                        mse = F.mse_loss(pred_masked, target_masked, reduction='sum')
                        
                        total_mse += mse.item()
                        num_predictions += valid_mask.sum().item()
        
        model.train()
        
        if num_predictions > 0:
            return total_mse / num_predictions
        return 0.0
    
    def learn(self, n_epochs: int = None, save_final: bool = True):
        """
        Train the model.
        
        Args:
            n_epochs: Number of training epochs
            save_final: Whether to save checkpoint at end
        """
        device = torch.device(self.device)
        
        # Setup logging
        log_csv_path = os.path.join(self.log_dir, "log.csv")
        csv_writer = csv.writer(open(log_csv_path, 'a', 1))
        csv_writer.writerow(["loss", "validation_mse"])
        
        print("=" * 60)
        print(f"HFTrainer - {self.model_name}")
        print(f"Start time: {datetime.now().strftime('%y-%m-%d-%H-%M-%S')}")
        print("=" * 60)
        print(f"Device: {device}")
        print(f"Log dir: {self.log_dir}")
        
        # Create data loader
        traj_data_loader = DataLoader(
            self.traj_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        
        n_epochs = n_epochs or int(1e6 / len(traj_data_loader))
        num_updates_per_iter = len(traj_data_loader)
        
        if num_updates_per_iter == 0:
            raise ValueError("Dataset too small for batch size")
        
        # Create model
        model = self._create_model()
        
        # Move to device (handle quantized models)
        if not (self.load_in_8bit or self.load_in_4bit):
            model = model.to(device)
        
        # Setup optimizer - only optimize trainable parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.wt_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / self.warmup_steps, 1)
        )
        
        # Set seeds
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        inner_bar = tqdm(range(num_updates_per_iter), leave=False)
        
        for epoch in trange(n_epochs):
            log_losses = []
            model.train()
            
            for batch_data in iter(traj_data_loader):
                timesteps, states, actions, traj_mask, state_mask = batch_data
                
                timesteps = timesteps.to(device)
                
                if isinstance(states, dict):
                    states = {k: v.to(device) for k, v in states.items()}
                else:
                    states = states.to(device)
                
                if state_mask is not None and isinstance(state_mask, dict):
                    state_mask = {k: v.to(device) for k, v in state_mask.items()}
                
                actions = actions.to(device)
                traj_mask = traj_mask.to(device)
                action_target = torch.clone(actions).detach()
                
                # Forward pass
                _, action_preds, _ = model.forward(
                    timesteps=timesteps,
                    states=states,
                    actions=actions,
                    state_mask=state_mask,
                )
                
                # Compute loss
                action_preds = action_preds.view(-1, self.act_dim)[traj_mask.view(-1) > 0]
                action_target = action_target.view(-1, self.act_dim)[traj_mask.view(-1) > 0]
                
                valid_mask = ~torch.isnan(action_target)
                if valid_mask.all():
                    loss = F.mse_loss(action_preds, action_target)
                else:
                    loss = F.mse_loss(action_preds[valid_mask], action_target[valid_mask])
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                
                log_losses.append(loss.item())
                inner_bar.update(1)
            
            inner_bar.reset()
            
            # Validation
            validation_mse = 0.0
            if epoch % self.validation_freq == 0:
                tqdm.write(f"Epoch {epoch}: Validating...")
                validation_mse = self.validate_rollout(model, device)
                tqdm.write(f"Validation MSE: {validation_mse:.6f}")
            
            avg_loss = np.mean(log_losses) if log_losses else 0.0
            csv_writer.writerow([avg_loss, validation_mse])
        
        self.model = model
        
        tqdm.write("=" * 60)
        tqdm.write("Finished training!")
        tqdm.write("=" * 60)
        
        if save_final:
            self.save(os.path.join(self.log_dir, "final"))
        
        return self


# Convenience function to check available HF models
def list_recommended_models():
    """List recommended HuggingFace models for trajectory prediction."""
    models = [
        ("google/gemma-2b", "2B params, good balance of size/performance"),
        ("google/gemma-7b", "7B params, better performance"),
        ("meta-llama/Llama-2-7b-hf", "7B params, requires HF token"),
        ("mistralai/Mistral-7B-v0.1", "7B params, efficient"),
        ("microsoft/phi-2", "2.7B params, very efficient"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "1.1B params, lightweight"),
    ]
    
    print("Recommended HuggingFace models for trajectory prediction:")
    print("-" * 60)
    for name, desc in models:
        print(f"  {name}")
        print(f"    {desc}")
    print("-" * 60)
    print("\nNote: Larger models generally perform better but require more GPU memory.")
    print("Use load_in_8bit=True or load_in_4bit=True for memory efficiency.")


if __name__ == "__main__":
    list_recommended_models()

