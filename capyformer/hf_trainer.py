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


class FlowMatchingHead(nn.Module):
    """
    Flow Matching action head inspired by pi0.
    
    Instead of directly predicting actions, this head learns a velocity field
    that transforms noise into actions through an ODE. At inference time,
    we integrate the ODE using Euler steps.
    
    Flow Matching formulation:
    - Training: Given clean action x_0 and noise x_1, create x_t = (1-t)*x_0 + t*x_1
                Learn to predict velocity v = dx/dt = x_1 - x_0
    - Inference: Start from x_1 (noise), integrate ODE to get x_0 (clean action)
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        action_dim: int,
        time_embed_dim: int = 64,
        mlp_hidden_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding using sinusoidal encoding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.GELU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        
        # Noisy action embedding
        self.action_embed = nn.Linear(action_dim, hidden_dim)
        
        # MLP to predict velocity field
        # Input: hidden_state + time_embed + noisy_action_embed
        total_dim = hidden_dim + time_embed_dim + hidden_dim
        self.velocity_net = nn.Sequential(
            nn.Linear(total_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, action_dim),
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        noisy_actions: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict velocity field for flow matching.
        
        Args:
            hidden_states: (B, T, hidden_dim) - transformer output
            noisy_actions: (B, T, action_dim) - noisy/interpolated actions
            timesteps: (B,) or (B, T) - diffusion timesteps in [0, 1]
        
        Returns:
            velocity: (B, T, action_dim) - predicted velocity field
        """
        B, T, _ = hidden_states.shape
        
        # Ensure timesteps have right shape
        if timesteps.dim() == 1:
            timesteps = timesteps.unsqueeze(1).expand(B, T)  # (B, T)
        
        # Embed time
        time_emb = self.time_embed(timesteps.reshape(-1))  # (B*T, time_embed_dim)
        time_emb = time_emb.reshape(B, T, -1)  # (B, T, time_embed_dim)
        
        # Embed noisy actions
        action_emb = self.action_embed(noisy_actions)  # (B, T, hidden_dim)
        
        # Concatenate all features
        features = torch.cat([hidden_states, time_emb, action_emb], dim=-1)
        
        # Predict velocity
        velocity = self.velocity_net(features)  # (B, T, action_dim)
        
        return velocity


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) tensor of timesteps
        Returns:
            (B, dim) sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


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
    
    Supports two action head types:
    - Direct regression (default): Directly predict actions
    - Flow Matching (use_flow_matching=True): Learn velocity field like pi0
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
        use_flow_matching: bool = False,
        flow_matching_steps: int = 10,
        state_mean: Optional[Dict[str, torch.Tensor]] = None,
        state_std: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        
        self.hf_model = hf_model
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.input_token_names = input_token_names
        self.num_input_tokens = len(input_token_dims)
        self.use_flow_matching = use_flow_matching
        self.flow_matching_steps = flow_matching_steps
        
        # Store dimensions as buffer for serialization
        self.register_buffer('input_token_dims', torch.tensor(input_token_dims, dtype=torch.long))
        
        # Input embeddings for each token type
        self.input_embeddings = nn.ModuleDict()
        for name, dim in zip(input_token_names, input_token_dims):
            self.input_embeddings[name] = ContinuousEmbedding(dim, hidden_dim)
        
        # Embedding for previous target (fed back auto-regressively)
        self.target_embedding = ContinuousEmbedding(target_dim, hidden_dim)
        
        # Output projection - either direct or flow matching
        if use_flow_matching:
            self.output_head = FlowMatchingHead(
                hidden_dim=hidden_dim,
                action_dim=target_dim,
                time_embed_dim=64,
                mlp_hidden_dim=256,
            )
        else:
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
    
    def _get_transformer_hidden_states(
        self,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        state_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Run inputs through transformer and get hidden states at target positions.
        
        Args:
            states: Dict[str, (B, T, dim)] input state tokens
            actions: (B, T, target_dim) previous actions (for auto-regressive)
            state_mask: Optional Dict[str, (B, T)] mask for missing tokens
        
        Returns:
            target_hidden: (B, T, hidden_dim) hidden states at action prediction positions
        """
        B, T, _ = actions.shape
        device = actions.device
        
        # Build sequence: [input1, input2, ..., target] for each timestep
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
        attention_mask = torch.cat(attention_masks, dim=1).float()
        
        # Pass through HF model
        outputs = self.hf_model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Get last hidden states
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden = outputs.last_hidden_state
        else:
            last_hidden = outputs.hidden_states[-1]
        
        # Extract hidden states at positions just before target tokens
        target_positions = []
        for t in range(T):
            pos = t * tokens_per_step + (self.num_input_tokens - 1)
            target_positions.append(pos)
        
        target_hidden = last_hidden[:, target_positions, :]  # (B, T, hidden)
        return target_hidden
    
    def forward(
        self,
        timesteps: torch.Tensor,
        states: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        state_mask: Optional[Dict[str, torch.Tensor]] = None,
        flow_timesteps: Optional[torch.Tensor] = None,
        noisy_actions: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass matching the original Transformer API.
        
        Args:
            timesteps: (B, T) timestep indices (unused, kept for API compatibility)
            states: Dict[str, (B, T, dim)] input state tokens
            actions: (B, T, target_dim) previous/target actions
            state_mask: Optional Dict[str, (B, T)] mask for missing tokens
            flow_timesteps: (B,) flow matching timesteps in [0, 1] (only for flow matching)
            noisy_actions: (B, T, target_dim) noisy actions x_t (only for flow matching training)
        
        Returns:
            For direct regression: (None, action_predictions, None)
            For flow matching: (None, predicted_velocity, None)
        """
        # Get transformer hidden states
        target_hidden = self._get_transformer_hidden_states(states, actions, state_mask)
        
        if self.use_flow_matching:
            if noisy_actions is None:
                raise ValueError("noisy_actions required for flow matching forward pass")
            if flow_timesteps is None:
                raise ValueError("flow_timesteps required for flow matching forward pass")
            
            # Predict velocity field
            velocity_preds = self.output_head(target_hidden, noisy_actions, flow_timesteps)
            return None, velocity_preds, None
        else:
            # Direct regression
            action_preds = self.output_head(target_hidden)
            return None, action_preds, None
    
    @torch.no_grad()
    def sample_actions(
        self,
        states: Dict[str, torch.Tensor],
        prev_actions: torch.Tensor,
        state_mask: Optional[Dict[str, torch.Tensor]] = None,
        num_steps: int = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample actions using flow matching ODE integration (like pi0).
        
        Args:
            states: Dict[str, (B, T, dim)] input state tokens
            prev_actions: (B, T, target_dim) previous actions for context
            state_mask: Optional mask for missing tokens
            num_steps: Number of Euler steps (default: self.flow_matching_steps)
            noise: Optional (B, T, target_dim) starting noise
        
        Returns:
            actions: (B, T, target_dim) sampled clean actions
        """
        if not self.use_flow_matching:
            raise RuntimeError("sample_actions only available when use_flow_matching=True")
        
        B, T, _ = prev_actions.shape
        device = prev_actions.device
        
        if num_steps is None:
            num_steps = self.flow_matching_steps
        
        if noise is None:
            noise = torch.randn(B, T, self.target_dim, device=device)
        
        # Get transformer hidden states (cached for all denoising steps)
        target_hidden = self._get_transformer_hidden_states(states, prev_actions, state_mask)
        
        # Flow matching integration from t=1 (noise) to t=0 (clean action)
        dt = -1.0 / num_steps
        x_t = noise
        time = torch.ones(B, device=device)  # Start at t=1
        
        for _ in range(num_steps):
            # Predict velocity at current timestep
            v_t = self.output_head(target_hidden, x_t, time)
            
            # Euler step: x_t = x_t + dt * v_t
            x_t = x_t + dt * v_t
            time = time + dt
        
        return x_t
    
    def get_inference(self, device: str = None, context_len: int = None) -> 'HFTransformerInference':
        """Create an inference wrapper for step-by-step prediction."""
        if device is None:
            device = next(self.parameters()).device
        
        if self.use_flow_matching:
            return HFFlowMatchingInference(self, device=str(device), context_len=context_len)
        else:
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


class HFFlowMatchingInference:
    """
    Inference wrapper for HFTrajectoryModel with Flow Matching.
    
    Uses iterative ODE integration to sample actions from noise,
    similar to pi0's approach.
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
        
        self.context_len = context_len if context_len is not None else 20
        self.num_denoise_steps = model.flow_matching_steps
        
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
    def step(self, current_state: dict, return_numpy: bool = True, num_steps: int = None):
        """
        Process current state and return predicted output using flow matching.
        
        Args:
            current_state: Dict mapping input token names to values (numpy or tensor)
            return_numpy: If True, return numpy array
            num_steps: Override number of denoising steps
        
        Returns:
            Predicted action sampled via flow matching
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
        
        prev_actions = torch.stack(actions_list, dim=0).unsqueeze(0).to(self.device)
        
        # Sample action using flow matching
        num_steps = num_steps or self.num_denoise_steps
        sampled_actions = self.model.sample_actions(
            states=states,
            prev_actions=prev_actions,
            state_mask=state_mask,
            num_steps=num_steps,
        )
        
        # Get prediction for current timestep (last position)
        predicted = sampled_actions[0, -1]
        
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


# =============================================================================
# NON-AUTOREGRESSIVE ACTION CHUNKING MODEL (like pi0's MPC-style control)
# =============================================================================

class ActionChunkingHead(nn.Module):
    """
    Action chunking head that predicts multiple future actions at once.
    
    Unlike autoregressive prediction, this head takes the transformer's
    hidden state and directly outputs an action horizon [a_t, a_t+1, ..., a_t+H].
    
    Supports both direct regression and flow matching.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        action_horizon: int,
        use_flow_matching: bool = False,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.use_flow_matching = use_flow_matching
        
        # Learnable action queries (one per timestep in horizon)
        self.action_queries = nn.Parameter(torch.randn(1, action_horizon, hidden_dim) * 0.02)
        
        # Cross-attention: action queries attend to context hidden states
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(hidden_dim)
        
        # MLP for final projection
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        if use_flow_matching:
            # Flow matching components
            self.time_embed = nn.Sequential(
                SinusoidalPosEmb(time_embed_dim),
                nn.Linear(time_embed_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.action_embed = nn.Linear(action_dim, hidden_dim)
            self.velocity_proj = nn.Linear(hidden_dim * 2, action_dim)
        else:
            # Direct regression
            self.action_proj = nn.Linear(hidden_dim, action_dim)
    
    def forward(
        self,
        context_hidden: torch.Tensor,
        noisy_actions: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict action chunk.
        
        Args:
            context_hidden: (B, T, hidden_dim) - context from transformer
            noisy_actions: (B, H, action_dim) - noisy actions for flow matching
            timesteps: (B,) - flow matching timesteps
        
        Returns:
            For direct: (B, H, action_dim) - predicted action chunk
            For flow matching: (B, H, action_dim) - predicted velocity
        """
        B = context_hidden.shape[0]
        
        # Expand action queries for batch
        queries = self.action_queries.expand(B, -1, -1)  # (B, H, hidden)
        
        # Cross-attention: queries attend to context
        attn_out, _ = self.cross_attention(
            query=queries,
            key=context_hidden,
            value=context_hidden,
        )
        queries = self.ln1(queries + attn_out)
        
        # MLP
        mlp_out = self.mlp(queries)
        queries = self.ln2(queries + mlp_out)  # (B, H, hidden)
        
        if self.use_flow_matching:
            # Flow matching: predict velocity
            time_emb = self.time_embed(timesteps)[:, None, :].expand(-1, self.action_horizon, -1)
            action_emb = self.action_embed(noisy_actions)
            combined = torch.cat([queries + time_emb, action_emb], dim=-1)
            velocity = self.velocity_proj(combined)
            return velocity
        else:
            # Direct regression
            actions = self.action_proj(queries)
            return actions


class HFActionChunkingModel(nn.Module):
    """
    Non-autoregressive HuggingFace model with Action Chunking (like pi0's MPC).
    
    Key differences from HFTrajectoryModel:
    - NO previous actions in input sequence (non-autoregressive)
    - Input: only state history [s_t-k, ..., s_t]
    - Output: action chunk [a_t, a_t+1, ..., a_t+H]
    - At inference: execute only a_t, then replan (MPC style)
    
    This is the architecture used by pi0 and other modern robot learning methods.
    
    Sequence format:
    [input1_t-k, input2_t-k, ..., input1_t-k+1, ..., input1_t, input2_t, ...]
    -> [action_t, action_t+1, ..., action_t+H]
    """
    
    input_token_names: List[str]
    
    def __init__(
        self,
        hf_model: nn.Module,
        input_token_dims: List[int],
        input_token_names: List[str],
        target_dim: int,
        hidden_dim: int,
        action_horizon: int = 16,
        use_flow_matching: bool = False,
        flow_matching_steps: int = 10,
        state_mean: Optional[Dict[str, torch.Tensor]] = None,
        state_std: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        
        self.hf_model = hf_model
        self.hidden_dim = hidden_dim
        self.target_dim = target_dim
        self.input_token_names = input_token_names
        self.num_input_tokens = len(input_token_dims)
        self.action_horizon = action_horizon
        self.use_flow_matching = use_flow_matching
        self.flow_matching_steps = flow_matching_steps
        
        # Store dimensions
        self.register_buffer('input_token_dims', torch.tensor(input_token_dims, dtype=torch.long))
        
        # Input embeddings for each token type
        self.input_embeddings = nn.ModuleDict()
        for name, dim in zip(input_token_names, input_token_dims):
            self.input_embeddings[name] = ContinuousEmbedding(dim, hidden_dim)
        
        # Action chunking head
        self.action_head = ActionChunkingHead(
            hidden_dim=hidden_dim,
            action_dim=target_dim,
            action_horizon=action_horizon,
            use_flow_matching=use_flow_matching,
        )
        
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
        return self.target_dim
    
    @property
    def state_token_dims(self) -> List[int]:
        return [int(d.item()) for d in self.input_token_dims]
    
    @property
    def state_token_names(self) -> List[str]:
        return self.input_token_names
    
    def _get_context_hidden(
        self,
        states: Dict[str, torch.Tensor],
        state_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Process input states through transformer (no actions in sequence).
        
        Args:
            states: Dict[str, (B, T, dim)] - input state tokens
            state_mask: Optional mask for missing tokens
        
        Returns:
            context_hidden: (B, T*num_tokens, hidden) - transformer output
        """
        B = list(states.values())[0].shape[0]
        T = list(states.values())[0].shape[1]
        device = list(states.values())[0].device
        
        # Build sequence: [input1, input2, ...] for each timestep
        # NO action tokens in the sequence (non-autoregressive)
        embeddings_list = []
        attention_masks = []
        
        for t in range(T):
            for name in self.input_token_names:
                state_t = states[name][:, t:t+1, :]  # (B, 1, dim)
                emb = self.input_embeddings[name](state_t)  # (B, 1, hidden)
                embeddings_list.append(emb)
                
                if state_mask is not None and name in state_mask:
                    mask_t = state_mask[name][:, t:t+1]
                else:
                    mask_t = torch.ones(B, 1, device=device, dtype=torch.bool)
                attention_masks.append(mask_t)
        
        # Concatenate: (B, T * num_input_tokens, hidden)
        hidden_states = torch.cat(embeddings_list, dim=1)
        attention_mask = torch.cat(attention_masks, dim=1).float()
        
        # Pass through HF model
        outputs = self.hf_model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        if hasattr(outputs, 'last_hidden_state'):
            context_hidden = outputs.last_hidden_state
        else:
            context_hidden = outputs.hidden_states[-1]
        
        return context_hidden
    
    def forward(
        self,
        states: Dict[str, torch.Tensor],
        target_actions: Optional[torch.Tensor] = None,
        state_mask: Optional[Dict[str, torch.Tensor]] = None,
        flow_timesteps: Optional[torch.Tensor] = None,
        noisy_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for action chunking.
        
        Args:
            states: Dict[str, (B, T, dim)] - input state tokens
            target_actions: (B, H, target_dim) - ground truth actions for training
            state_mask: Optional mask for missing tokens
            flow_timesteps: (B,) - flow matching timesteps
            noisy_actions: (B, H, target_dim) - noisy actions for flow matching
        
        Returns:
            For direct: (B, H, target_dim) - predicted action chunk
            For flow matching: (B, H, target_dim) - predicted velocity
        """
        # Get context from transformer
        context_hidden = self._get_context_hidden(states, state_mask)
        
        # Predict action chunk
        if self.use_flow_matching:
            if noisy_actions is None or flow_timesteps is None:
                raise ValueError("noisy_actions and flow_timesteps required for flow matching")
            return self.action_head(context_hidden, noisy_actions, flow_timesteps)
        else:
            return self.action_head(context_hidden)
    
    @torch.no_grad()
    def sample_action_chunk(
        self,
        states: Dict[str, torch.Tensor],
        state_mask: Optional[Dict[str, torch.Tensor]] = None,
        num_steps: int = None,
    ) -> torch.Tensor:
        """
        Sample action chunk (for flow matching).
        
        Args:
            states: Dict[str, (B, T, dim)] - input state tokens
            state_mask: Optional mask
            num_steps: ODE integration steps
        
        Returns:
            actions: (B, H, target_dim) - sampled action chunk
        """
        if not self.use_flow_matching:
            # Direct regression - just forward pass
            return self.action_head(self._get_context_hidden(states, state_mask))
        
        B = list(states.values())[0].shape[0]
        device = list(states.values())[0].device
        
        if num_steps is None:
            num_steps = self.flow_matching_steps
        
        # Get context hidden states
        context_hidden = self._get_context_hidden(states, state_mask)
        
        # Sample from noise
        noise = torch.randn(B, self.action_horizon, self.target_dim, device=device)
        
        # ODE integration
        dt = -1.0 / num_steps
        x_t = noise
        time = torch.ones(B, device=device)
        
        for _ in range(num_steps):
            v_t = self.action_head(context_hidden, x_t, time)
            x_t = x_t + dt * v_t
            time = time + dt
        
        return x_t
    
    def get_inference(self, device: str = None, context_len: int = None) -> 'ActionChunkingInference':
        """Create inference wrapper."""
        if device is None:
            device = next(self.parameters()).device
        return ActionChunkingInference(self, device=str(device), context_len=context_len)


class ActionChunkingInference:
    """
    MPC-style inference for Action Chunking model.
    
    At each step:
    1. Take state history as input
    2. Predict action chunk [a_t, a_t+1, ..., a_t+H]
    3. Execute only a_t
    4. Optionally: execute multiple actions before replanning
    
    This matches pi0's inference pattern.
    """
    
    def __init__(
        self,
        model: HFActionChunkingModel,
        device: str = 'cpu',
        context_len: int = None,
        execute_horizon: int = 1,  # How many actions to execute before replanning
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        self.input_token_names = model.input_token_names
        self.input_token_dims = [int(d.item()) for d in model.input_token_dims]
        self.target_dim = model.target_dim
        self.action_horizon = model.action_horizon
        
        self.state_mean = model.state_mean
        self.state_std = model.state_std
        
        self.context_len = context_len if context_len is not None else 20
        self.execute_horizon = execute_horizon  # Execute this many actions before replanning
        
        self.reset()
    
    def reset(self):
        """Reset for new episode."""
        self.timestep = 0
        self.state_history = {name: [] for name in self.input_token_names}
        self.mask_history = {name: [] for name in self.input_token_names}
        
        # Cached action chunk for multi-step execution
        self.cached_actions = None
        self.cached_action_idx = 0
    
    def _normalize_state(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """Normalize state using stored statistics."""
        if self.state_mean is not None and self.state_std is not None:
            if name in self.state_mean and name in self.state_std:
                mean = self.state_mean[name].to(value.device)
                std = self.state_std[name].to(value.device)
                return (value - mean) / std
        return value
    
    @torch.no_grad()
    def step(self, current_state: dict, return_numpy: bool = True, force_replan: bool = False):
        """
        Get action for current state (MPC style).
        
        Args:
            current_state: Dict mapping input token names to values
            return_numpy: If True, return numpy array
            force_replan: If True, always recompute action chunk
        
        Returns:
            action: Single action for current timestep
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
        
        # Check if we need to replan
        need_replan = (
            force_replan or
            self.cached_actions is None or
            self.cached_action_idx >= self.execute_horizon
        )
        
        if need_replan:
            # Build context from state history
            n_steps = min(len(self.state_history[self.input_token_names[0]]), self.context_len)
            
            states = {}
            state_mask = {}
            for name in self.input_token_names:
                state_list = self.state_history[name][-n_steps:]
                mask_list = self.mask_history[name][-n_steps:]
                states[name] = torch.stack(state_list, dim=0).unsqueeze(0).to(self.device)
                state_mask[name] = torch.stack(mask_list, dim=0).unsqueeze(0).to(self.device)
            
            # Predict action chunk
            if self.model.use_flow_matching:
                action_chunk = self.model.sample_action_chunk(states, state_mask)
            else:
                action_chunk = self.model.forward(states, state_mask=state_mask)
            
            self.cached_actions = action_chunk[0]  # (H, action_dim)
            self.cached_action_idx = 0
        
        # Get action for current timestep
        action = self.cached_actions[self.cached_action_idx]
        self.cached_action_idx += 1
        self.timestep += 1
        
        # Trim history
        max_history = self.context_len * 2
        if len(self.state_history[self.input_token_names[0]]) > max_history:
            for name in self.input_token_names:
                self.state_history[name] = self.state_history[name][-max_history:]
                self.mask_history[name] = self.mask_history[name][-max_history:]
        
        if return_numpy:
            return action.cpu().numpy()
        return action
    
    def get_full_action_chunk(self, return_numpy: bool = True):
        """
        Get the full predicted action chunk.
        
        Useful for visualization or debugging.
        
        Returns:
            (H, action_dim) - full action horizon
        """
        if self.cached_actions is None:
            return None
        
        if return_numpy:
            return self.cached_actions.cpu().numpy()
        return self.cached_actions


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
    - **Flow Matching action head** (like pi0) for improved action generation
    
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
        use_flow_matching: Whether to use flow matching action head (like pi0)
        flow_matching_steps: Number of ODE integration steps for flow matching
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
        model_name: str = "google/gemma-3-270m",
        log_dir: str = "./logs",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_backbone: bool = True,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_action_tanh: bool = False,
        use_flow_matching: bool = False,
        flow_matching_steps: int = 10,
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
        validation_metric_fn: callable = None,
    ):
        """
        validation_metric_fn: Optional custom validation metric function.
            Should take (predictions: np.ndarray, targets: np.ndarray) and return
            a tuple (metric_value: float, metric_name: str). If provided, this 
            metric will be displayed during validation instead of MSE.
        """
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
        self.use_flow_matching = use_flow_matching
        self.flow_matching_steps = flow_matching_steps
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
        self.validation_metric_fn = validation_metric_fn
        
        # Get dataset properties
        self.act_dim = dataset.act_dim
        self.state_token_dims = dataset.state_token_dims
        self.state_token_names = dataset.state_token_names
        self.context_len = dataset.context_len
        
        self.model = None
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        if use_flow_matching:
            print(f"🌊 Flow Matching enabled with {flow_matching_steps} denoising steps")
    
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
            use_flow_matching=self.use_flow_matching,
            flow_matching_steps=self.flow_matching_steps,
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
                "use_flow_matching": self.use_flow_matching,
                "flow_matching_steps": self.flow_matching_steps,
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
        self.use_flow_matching = config.get("use_flow_matching", False)
        self.flow_matching_steps = config.get("flow_matching_steps", 10)
        
        self.model = self._create_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()
        
        # Update trainer config
        trainer_config = checkpoint.get("trainer_config", {})
        self.action_is_velocity = trainer_config.get("action_is_velocity", True)
        self.dt = trainer_config.get("dt", 0.02)
        
        print(f"Model loaded successfully")
        if self.use_flow_matching:
            print(f"  Flow Matching enabled ({self.flow_matching_steps} steps)")
        return self
    
    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda:0"):
        """
        Load an HFTrainer from a checkpoint file (class method - no instance needed).
        
        This is the recommended way to load a model for inference without training.
        
        Args:
            path: Path to the checkpoint file (.pt).
            device: Device to load the model to.
        
        Returns:
            HFTrainer instance with loaded model, ready for inference.
        
        Example:
            trainer = HFTrainer.from_checkpoint("./models/my_model.pt")
            inference = trainer.get_inference()
            
            for t in range(episode_length):
                state = {'position': pos, 'velocity': vel}
                prediction = inference.step(state)
        """
        if not path.endswith(".pt"):
            path = f"{path}.pt"
        
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=device)
        
        config = checkpoint["model_config"]
        trainer_config = checkpoint.get("trainer_config", {})
        
        # Get state stats from checkpoint
        state_mean = config.get("state_mean")
        state_std = config.get("state_std")
        
        if state_mean is None or state_std is None:
            raise ValueError(
                "Checkpoint missing state_mean/state_std. "
                "Use trainer.load() with a dataset instead."
            )
        
        # Create a minimal HFTrainer instance without a dataset
        trainer = object.__new__(cls)
        
        # Set essential attributes from checkpoint
        trainer.traj_dataset = None
        trainer.model_name = config["model_name"]
        trainer.log_dir = os.path.dirname(path) or "."
        trainer.act_dim = config["target_dim"]
        trainer.state_token_dims = config["input_token_dims"]
        trainer.state_token_names = config["input_token_names"]
        trainer.context_len = config["context_len"]
        trainer.use_action_tanh = config.get("use_action_tanh", False)
        trainer.use_flow_matching = config.get("use_flow_matching", False)
        trainer.flow_matching_steps = config.get("flow_matching_steps", 10)
        trainer.device = device
        
        # Set trainer config
        trainer.action_is_velocity = trainer_config.get("action_is_velocity", True)
        trainer.dt = trainer_config.get("dt", 0.02)
        trainer.use_lora = trainer_config.get("use_lora", False)
        trainer.lora_r = trainer_config.get("lora_r", 16)
        trainer.lora_alpha = trainer_config.get("lora_alpha", 32)
        
        # Set other attributes to defaults (not needed for inference)
        trainer.lora_dropout = 0.05
        trainer.freeze_backbone = True
        trainer.load_in_8bit = False
        trainer.load_in_4bit = False
        trainer.batch_size = 32
        trainer.learning_rate = 1e-4
        trainer.wt_decay = 0.01
        trainer.warmup_steps = 1000
        trainer.seed = 0
        trainer.validation_freq = 100
        trainer.validation_trajectories = 10
        
        # Load HF model backbone
        print(f"Loading HuggingFace model: {trainer.model_name}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            trainer.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        
        hidden_dim = hf_model.config.hidden_size
        
        # Create wrapper model
        trainer.model = HFTrajectoryModel(
            hf_model=hf_model,
            input_token_dims=trainer.state_token_dims,
            input_token_names=trainer.state_token_names,
            target_dim=trainer.act_dim,
            hidden_dim=hidden_dim,
            use_action_tanh=trainer.use_action_tanh,
            use_flow_matching=trainer.use_flow_matching,
            flow_matching_steps=trainer.flow_matching_steps,
            state_mean=state_mean,
            state_std=state_std,
        )
        
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.to(device)
        trainer.model.eval()
        
        print(f"Model loaded successfully from {path}")
        if trainer.use_flow_matching:
            print(f"  Flow Matching enabled ({trainer.flow_matching_steps} steps)")
        
        return trainer
    
    def validate_rollout(self, model, device):
        """Validate by performing trajectory rollouts.
        
        Returns:
            tuple: (metric_value, metric_name) if validation_metric_fn is set,
                   otherwise (mse_value, "MSE")
        """
        model.eval()
        all_predictions = []
        all_targets = []
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
                        
                        # Collect for custom metric
                        all_predictions.append(pred_masked.numpy())
                        all_targets.append(target_masked.numpy())
                        
                        mse = F.mse_loss(pred_masked, target_masked, reduction='sum')
                        total_mse += mse.item()
                        num_predictions += valid_mask.sum().item()
        
        model.train()
        
        # Use custom metric if provided
        if self.validation_metric_fn is not None and len(all_predictions) > 0:
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            metric_value, metric_name = self.validation_metric_fn(all_predictions, all_targets)
            return metric_value, metric_name
        
        # Default: return MSE
        if num_predictions > 0:
            return total_mse / num_predictions, "MSE"
        return 0.0, "MSE"
    
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
                
                if self.use_flow_matching:
                    # Flow Matching training
                    B, T, _ = actions.shape
                    
                    # Sample random timesteps t ~ U[0, 1]
                    flow_t = torch.rand(B, device=device)
                    
                    # Sample noise
                    noise = torch.randn_like(actions)
                    
                    # Create interpolated actions x_t = (1-t)*x_0 + t*x_1
                    # where x_0 is clean action, x_1 is noise
                    t_expanded = flow_t[:, None, None].expand(-1, T, self.act_dim)
                    noisy_actions = (1 - t_expanded) * actions + t_expanded * noise
                    
                    # Target velocity: v = x_1 - x_0 = noise - actions
                    target_velocity = noise - actions
                    
                    # Forward pass predicts velocity
                    _, velocity_preds, _ = model.forward(
                        timesteps=timesteps,
                        states=states,
                        actions=actions,
                        state_mask=state_mask,
                        flow_timesteps=flow_t,
                        noisy_actions=noisy_actions,
                    )
                    
                    # Compute flow matching loss
                    velocity_preds = velocity_preds.view(-1, self.act_dim)[traj_mask.view(-1) > 0]
                    target_velocity = target_velocity.view(-1, self.act_dim)[traj_mask.view(-1) > 0]
                    
                    valid_mask = ~torch.isnan(target_velocity)
                    if valid_mask.all():
                        loss = F.mse_loss(velocity_preds, target_velocity)
                    else:
                        loss = F.mse_loss(velocity_preds[valid_mask], target_velocity[valid_mask])
                else:
                    # Direct regression training
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
            validation_metric = 0.0
            metric_name = "MSE"
            if epoch % self.validation_freq == 0:
                tqdm.write(f"Epoch {epoch}: Validating...")
                validation_metric, metric_name = self.validate_rollout(model, device)
                tqdm.write(f"Validation {metric_name}: {validation_metric:.6f}")
            
            avg_loss = np.mean(log_losses) if log_losses else 0.0
            csv_writer.writerow([avg_loss, validation_metric])
        
        self.model = model
        
        tqdm.write("=" * 60)
        tqdm.write("Finished training!")
        tqdm.write("=" * 60)
        
        if save_final:
            self.save(os.path.join(self.log_dir, "final"))
        
        return self


class HFActionChunkingTrainer:
    """
    Trainer for non-autoregressive Action Chunking model (like pi0's MPC).
    
    Key differences from HFTrainer:
    - Non-autoregressive: no previous actions in input sequence
    - Action chunking: predicts multiple future actions at once
    - MPC-style inference: predict horizon, execute first action
    
    This architecture matches pi0 and modern robot learning methods.
    
    Args:
        dataset: TrajectoryDataset instance
        model_name: HuggingFace model name
        action_horizon: Number of future actions to predict (default: 16)
        execute_horizon: How many actions to execute before replanning (default: 1)
        use_flow_matching: Whether to use flow matching
        ... (other args same as HFTrainer)
    """
    
    def __init__(
        self,
        dataset: TrajectoryDataset,
        model_name: str = "google/gemma-3-270m",
        log_dir: str = "./logs",
        action_horizon: int = 16,
        execute_horizon: int = 1,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_backbone: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        use_flow_matching: bool = False,
        flow_matching_steps: int = 10,
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
        validation_callback=None,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError("HuggingFace Transformers not installed.")
        
        if use_lora and not HAS_PEFT:
            raise ImportError("PEFT not installed but use_lora=True.")
        
        self.traj_dataset = dataset
        self.model_name = model_name
        self.log_dir = log_dir
        self.action_horizon = action_horizon
        self.execute_horizon = execute_horizon
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.freeze_backbone = freeze_backbone
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.use_flow_matching = use_flow_matching
        self.flow_matching_steps = flow_matching_steps
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
        self.validation_callback = validation_callback
        
        self.act_dim = dataset.act_dim
        self.state_token_dims = dataset.state_token_dims
        self.state_token_names = dataset.state_token_names
        self.context_len = dataset.context_len
        
        self.model = None
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"🎯 Action Chunking Model (MPC-style like pi0)")
        print(f"   Action horizon: {action_horizon}")
        print(f"   Execute horizon: {execute_horizon}")
        if use_flow_matching:
            print(f"   🌊 Flow Matching: {flow_matching_steps} steps")
    
    def _create_model(self) -> HFActionChunkingModel:
        """Create the Action Chunking model."""
        print(f"Loading HuggingFace model: {self.model_name}")
        
        quantization_config = None
        if self.load_in_8bit or self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=self.load_in_8bit,
                load_in_4bit=self.load_in_4bit,
                bnb_4bit_compute_dtype=torch.float16,
            )
        
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if (self.load_in_8bit or self.load_in_4bit) else torch.float32,
            device_map="auto" if (self.load_in_8bit or self.load_in_4bit) else None,
            trust_remote_code=True,
        )
        
        config = hf_model.config
        hidden_dim = config.hidden_size
        print(f"HF model hidden dimension: {hidden_dim}")
        
        if self.use_lora:
            print(f"Applying LoRA (r={self.lora_r}, alpha={self.lora_alpha})")
            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            hf_model = get_peft_model(hf_model, lora_config)
            hf_model.print_trainable_parameters()
        elif self.freeze_backbone:
            for param in hf_model.parameters():
                param.requires_grad = False
        
        state_mean, state_std = self.traj_dataset.get_state_stats()
        
        model = HFActionChunkingModel(
            hf_model=hf_model,
            input_token_dims=self.state_token_dims,
            input_token_names=self.state_token_names,
            target_dim=self.act_dim,
            hidden_dim=hidden_dim,
            action_horizon=self.action_horizon,
            use_flow_matching=self.use_flow_matching,
            flow_matching_steps=self.flow_matching_steps,
            state_mean=state_mean,
            state_std=state_std,
        )
        
        return model
    
    def get_inference(self, device: str = None):
        """Get MPC-style inference wrapper."""
        if self.model is None:
            raise RuntimeError("No model available. Call learn() first.")
        
        if device is None:
            device = self.device
        
        inference = self.model.get_inference(device=device, context_len=self.context_len)
        inference.execute_horizon = self.execute_horizon
        return inference
    
    def save(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise RuntimeError("No model available.")
        
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        if path.endswith(".pt"):
            path = path.rsplit(".", 1)[0]
        
        checkpoint_path = f"{path}.pt"
        
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
                "action_horizon": self.action_horizon,
                "use_flow_matching": self.use_flow_matching,
                "flow_matching_steps": self.flow_matching_steps,
                "state_mean": state_mean,
                "state_std": state_std,
            },
            "trainer_config": {
                "execute_horizon": self.execute_horizon,
                "action_is_velocity": self.action_is_velocity,
                "dt": self.dt,
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def load(self, path: str, device: str = None):
        """
        Load a saved model checkpoint.
        
        Args:
            path: Path to checkpoint (with or without .pt extension)
            device: Device to load model to (defaults to cuda if available)
        """
        if not path.endswith(".pt"):
            path = f"{path}.pt"
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(path, map_location=device)
        
        model_config = checkpoint["model_config"]
        trainer_config = checkpoint.get("trainer_config", {})
        
        # Update trainer config
        self.execute_horizon = trainer_config.get("execute_horizon", self.execute_horizon)
        self.action_is_velocity = trainer_config.get("action_is_velocity", self.action_is_velocity)
        self.dt = trainer_config.get("dt", self.dt)
        
        # Update model config
        self.use_flow_matching = model_config.get("use_flow_matching", False)
        self.flow_matching_steps = model_config.get("flow_matching_steps", 10)
        self.action_horizon = model_config.get("action_horizon", self.action_horizon)
        self.context_len = model_config.get("context_len", self.context_len)
        
        # Recreate model with saved config
        state_mean = model_config.get("state_mean")
        state_std = model_config.get("state_std")
        
        # Get backbone
        backbone = AutoModelForCausalLM.from_pretrained(
            model_config["model_name"],
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        
        # Get hidden_dim from backbone config
        hidden_dim = backbone.config.hidden_size
        
        model = HFActionChunkingModel(
            hf_model=backbone,
            input_token_dims=model_config["input_token_dims"],
            input_token_names=model_config["input_token_names"],
            target_dim=model_config["target_dim"],
            hidden_dim=hidden_dim,
            action_horizon=model_config["action_horizon"],
            use_flow_matching=model_config.get("use_flow_matching", False),
            flow_matching_steps=model_config.get("flow_matching_steps", 10),
            state_mean=state_mean,
            state_std=state_std,
        )
        
        # Load weights
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        self.model = model
        
        print(f"Model loaded from {path}")
        print(f"  Action horizon: {self.action_horizon}")
        print(f"  Execute horizon: {self.execute_horizon}")
        if self.use_flow_matching:
            print(f"  Flow Matching enabled ({self.flow_matching_steps} steps)")
        
        return self
    
    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cuda:0"):
        """
        Load an HFActionChunkingTrainer from a checkpoint file (class method - no instance needed).
        
        This is the recommended way to load a model for inference without training.
        
        Args:
            path: Path to the checkpoint file (.pt).
            device: Device to load the model to.
        
        Returns:
            HFActionChunkingTrainer instance with loaded model, ready for inference.
        
        Example:
            trainer = HFActionChunkingTrainer.from_checkpoint("./models/my_model.pt")
            inference = trainer.get_inference()
            
            for t in range(episode_length):
                state = {'position': pos, 'velocity': vel}
                action = inference.step(state)
        """
        if not path.endswith(".pt"):
            path = f"{path}.pt"
        
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=device)
        
        model_config = checkpoint["model_config"]
        trainer_config = checkpoint.get("trainer_config", {})
        
        # Get state stats from checkpoint
        state_mean = model_config.get("state_mean")
        state_std = model_config.get("state_std")
        
        if state_mean is None or state_std is None:
            raise ValueError(
                "Checkpoint missing state_mean/state_std. "
                "Use trainer.load() with a dataset instead."
            )
        
        # Create a minimal HFActionChunkingTrainer instance without a dataset
        trainer = object.__new__(cls)
        
        # Set essential attributes from checkpoint
        trainer.traj_dataset = None
        trainer.model_name = model_config["model_name"]
        trainer.log_dir = os.path.dirname(path) or "."
        trainer.act_dim = model_config["target_dim"]
        trainer.state_token_dims = model_config["input_token_dims"]
        trainer.state_token_names = model_config["input_token_names"]
        trainer.context_len = model_config["context_len"]
        trainer.action_horizon = model_config["action_horizon"]
        trainer.use_flow_matching = model_config.get("use_flow_matching", False)
        trainer.flow_matching_steps = model_config.get("flow_matching_steps", 10)
        trainer.device = device
        
        # Set trainer config
        trainer.execute_horizon = trainer_config.get("execute_horizon", 1)
        trainer.action_is_velocity = trainer_config.get("action_is_velocity", True)
        trainer.dt = trainer_config.get("dt", 0.02)
        
        # Set other attributes to defaults (not needed for inference)
        trainer.use_lora = False
        trainer.lora_r = 16
        trainer.lora_alpha = 32
        trainer.lora_dropout = 0.05
        trainer.freeze_backbone = True
        trainer.load_in_8bit = False
        trainer.load_in_4bit = False
        trainer.batch_size = 32
        trainer.learning_rate = 1e-4
        trainer.wt_decay = 0.01
        trainer.warmup_steps = 1000
        trainer.seed = 0
        trainer.validation_freq = 100
        trainer.validation_trajectories = 10
        
        # Load HF model backbone
        print(f"Loading HuggingFace model: {trainer.model_name}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            trainer.model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        
        hidden_dim = hf_model.config.hidden_size
        
        # Create Action Chunking model
        trainer.model = HFActionChunkingModel(
            hf_model=hf_model,
            input_token_dims=trainer.state_token_dims,
            input_token_names=trainer.state_token_names,
            target_dim=trainer.act_dim,
            hidden_dim=hidden_dim,
            action_horizon=trainer.action_horizon,
            use_flow_matching=trainer.use_flow_matching,
            flow_matching_steps=trainer.flow_matching_steps,
            state_mean=state_mean,
            state_std=state_std,
        )
        
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.model.to(device)
        trainer.model.eval()
        
        print(f"Model loaded successfully from {path}")
        print(f"  Action horizon: {trainer.action_horizon}")
        print(f"  Execute horizon: {trainer.execute_horizon}")
        if trainer.use_flow_matching:
            print(f"  Flow Matching enabled ({trainer.flow_matching_steps} steps)")
        
        return trainer
    
    def validate_rollout(self, model, device):
        """
        Validate by performing trajectory rollouts using ActionChunkingInference.
        
        This uses the real inference pipeline (MPC-style) to ensure validation
        matches actual usage. At each step:
        1. Feed current state to inference
        2. Get action (first action from predicted chunk)
        3. Compare with ground truth
        
        Returns:
            Average MSE across all validation trajectories
        """
        model.eval()
        total_mse = 0.0
        num_predictions = 0
        
        # Create inference wrapper - matches real usage
        inference = ActionChunkingInference(
            model, 
            device=str(device), 
            context_len=self.context_len,
            execute_horizon=self.execute_horizon,
        )
        
        with torch.no_grad():
            for traj_i in range(self.validation_trajectories):
                if len(self.traj_dataset.val_trajectories) == 0:
                    break
                    
                traj_idx = random.randint(0, len(self.traj_dataset.val_trajectories) - 1)
                traj = self.traj_dataset.val_trajectories[traj_idx]
                
                # Get trajectory data
                inputs = self.traj_dataset.get_inputs(traj)
                target_gt = self.traj_dataset.get_target(traj)
                traj_len = min(self.traj_dataset.get_traj_len(traj), 100)  # Limit for speed
                
                if not isinstance(inputs, dict):
                    continue
                
                if traj_len < 2:
                    continue
                
                # Reset inference for new trajectory
                inference.reset()
                
                # Run inference step by step (MPC style)
                predicted_actions = []
                for t in range(traj_len - 1):
                    # Build current state dict
                    current_state = {}
                    for key in inputs.keys():
                        state_value = inputs[key][t]
                        if np.all(np.isnan(state_value)):
                            current_state[key] = None
                        else:
                            current_state[key] = np.nan_to_num(state_value, nan=0.0)
                    
                    # Get action using inference (this tests the full pipeline!)
                    # force_replan=True to test action chunking each step
                    pred_action = inference.step(current_state, return_numpy=False, force_replan=True)
                    predicted_actions.append(pred_action.cpu())
                
                # Compute MSE
                if len(predicted_actions) > 0:
                    predicted_actions = torch.stack(predicted_actions)
                    actual_targets = torch.from_numpy(target_gt[:len(predicted_actions)]).float()
                    
                    # Mask out NaN values
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
        """Train the Action Chunking model."""
        device = torch.device(self.device)
        
        log_csv_path = os.path.join(self.log_dir, "log.csv")
        csv_writer = csv.writer(open(log_csv_path, 'a', 1))
        csv_writer.writerow(["loss", "validation_mse"])
        
        print("=" * 60)
        print(f"HFActionChunkingTrainer - {self.model_name}")
        print(f"Start time: {datetime.now().strftime('%y-%m-%d-%H-%M-%S')}")
        print("=" * 60)
        
        traj_data_loader = DataLoader(
            self.traj_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
        
        n_epochs = n_epochs or int(1e6 / len(traj_data_loader))
        
        model = self._create_model()
        
        self.model = model
        
        if not (self.load_in_8bit or self.load_in_4bit):
            model = model.to(device)
        
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
        
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        inner_bar = tqdm(range(len(traj_data_loader)), leave=False)
        
        for epoch in trange(n_epochs):
            log_losses = []
            model.train()
            
            for batch_data in iter(traj_data_loader):
                timesteps, states, actions, traj_mask, state_mask = batch_data
                
                if isinstance(states, dict):
                    states = {k: v.to(device) for k, v in states.items()}
                
                if state_mask is not None and isinstance(state_mask, dict):
                    state_mask = {k: v.to(device) for k, v in state_mask.items()}
                
                actions = actions.to(device)
                traj_mask = traj_mask.to(device)
                
                B, T, action_dim = actions.shape
                
                # Debug: print on first batch of first epoch
                if epoch == 0 and len(log_losses) == 0:
                    print(f"\n[DEBUG] Batch shapes: B={B}, T={T}, action_dim={action_dim}")
                    print(f"[DEBUG] action_horizon={self.action_horizon}, context_len={self.context_len}")
                
                # For action chunking, we need TEMPORAL ALIGNMENT:
                # - State context: [s_0, ..., s_t] (history up to prediction point)
                # - Action target: [a_t, a_t+1, ..., a_t+H] (future actions to predict)
                
                # Need enough length for action horizon
                if T < self.action_horizon + 1:
                    if epoch == 0:
                        print(f"[DEBUG] Skipping batch: T={T} < action_horizon+1={self.action_horizon + 1}")
                    inner_bar.update(1)
                    continue
                
                # Sample prediction start points
                # pred_start is where we start predicting: [a_pred_start, ..., a_pred_start+H]
                # Context will be states before pred_start
                min_pred_start = 1  # Need at least 1 state as context
                max_pred_start = T - self.action_horizon  # Leave room for action horizon
                
                if max_pred_start < min_pred_start:
                    if epoch == 0:
                        print(f"[DEBUG] Skipping batch: max_pred_start={max_pred_start} < min_pred_start={min_pred_start}")
                    inner_bar.update(1)
                    continue
                
                # Use max_pred_start + 1 because randint is exclusive on the right
                pred_start_indices = torch.randint(min_pred_start, max_pred_start + 1, (B,), device=device)
                
                # Extract aligned state context and action chunks for each batch item
                # State context: states UP TO pred_start (the context the model sees)
                # Action chunk: actions FROM pred_start onwards
                
                aligned_states = {}
                aligned_masks = {}
                context_len_actual = self.context_len
                
                for key in states.keys():
                    state_chunks = []
                    for b in range(B):
                        # Get context ending at pred_start (exclusive)
                        end_idx = pred_start_indices[b].item()
                        start_idx = max(0, end_idx - context_len_actual)
                        
                        # Extract context [start_idx, end_idx)
                        ctx = states[key][b, start_idx:end_idx, :]  # (ctx_len, dim)
                        
                        # Pad to context_len if needed
                        if ctx.shape[0] < context_len_actual:
                            pad_len = context_len_actual - ctx.shape[0]
                            pad = torch.zeros(pad_len, ctx.shape[1], device=device, dtype=ctx.dtype)
                            ctx = torch.cat([pad, ctx], dim=0)
                        
                        state_chunks.append(ctx)
                    
                    aligned_states[key] = torch.stack(state_chunks, dim=0)  # (B, context_len, dim)
                
                # Handle state masks similarly
                if state_mask is not None:
                    for key in state_mask.keys():
                        mask_chunks = []
                        for b in range(B):
                            end_idx = pred_start_indices[b].item()
                            start_idx = max(0, end_idx - context_len_actual)
                            
                            msk = state_mask[key][b, start_idx:end_idx]
                            
                            if msk.shape[0] < context_len_actual:
                                pad_len = context_len_actual - msk.shape[0]
                                pad = torch.zeros(pad_len, device=device, dtype=msk.dtype)
                                msk = torch.cat([pad, msk], dim=0)
                            
                            mask_chunks.append(msk)
                        
                        aligned_masks[key] = torch.stack(mask_chunks, dim=0)
                else:
                    aligned_masks = None
                
                # Extract action chunks: [a_pred_start, ..., a_pred_start+H]
                action_chunks = torch.stack([
                    actions[b, pred_start_indices[b]:pred_start_indices[b]+self.action_horizon]
                    for b in range(B)
                ])
                
                if self.use_flow_matching:
                    # Flow matching training
                    flow_t = torch.rand(B, device=device)
                    noise = torch.randn_like(action_chunks)
                    
                    t_expanded = flow_t[:, None, None].expand(-1, self.action_horizon, self.act_dim)
                    noisy_actions = (1 - t_expanded) * action_chunks + t_expanded * noise
                    target_velocity = noise - action_chunks
                    
                    velocity_preds = model.forward(
                        states=aligned_states,
                        state_mask=aligned_masks,
                        flow_timesteps=flow_t,
                        noisy_actions=noisy_actions,
                    )
                    
                    loss = F.mse_loss(velocity_preds, target_velocity)
                else:
                    # Direct regression
                    action_preds = model.forward(
                        states=aligned_states,
                        state_mask=aligned_masks,
                    )
                    
                    loss = F.mse_loss(action_preds, action_chunks)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                scheduler.step()
                
                log_losses.append(loss.item())
                inner_bar.update(1)
            
            inner_bar.reset()
            
            # Debug: check if any batches were processed
            if epoch == 0:
                print(f"\n[DEBUG] Epoch 0: Processed {len(log_losses)} batches")
                if len(log_losses) > 0:
                    print(f"[DEBUG] First loss: {log_losses[0]:.6f}, Last loss: {log_losses[-1]:.6f}")
            
            # Real validation using ActionChunkingInference
            validation_mse = 0.0
            if epoch % self.validation_freq == 0:
                avg_loss = np.mean(log_losses) if log_losses else 0.0
                tqdm.write(f"Epoch {epoch}: Loss = {avg_loss:.6f}, Validating...")
                validation_mse = self.validate_rollout(model, device)
                tqdm.write(f"  Validation MSE: {validation_mse:.6f}")
                
                 # Call validation callback if provided
                if self.validation_callback is not None:
                    self.validation_callback(self, epoch)

            
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

