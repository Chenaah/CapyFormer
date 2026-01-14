"""
this extremely minimal Decision Transformer model is based on
the following causal transformer (GPT) implementation:

Misha Laskin's tweet:
https://twitter.com/MishaLaskin/status/1481767788775628801?cxt=HHwWgoCzmYD9pZApAAAA

and its corresponding notebook:
https://colab.research.google.com/drive/1NUBqyboDcGte5qAJKOl8gaJC28V_73Iv?usp=sharing

** the above colab notebook has a bug while applying masked_fill 
which is fixed in the following code
"""

import math
import pdb
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class MaskedCausalAttention(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        ones = torch.ones((max_T, max_T))
        mask = torch.tril(ones).view(1, 1, max_T, max_T)

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer('mask',mask)

    def forward(self, x, token_mask: Optional[torch.Tensor] = None):
        """
        Forward pass with optional token masking.
        
        Args:
            x: Input tensor of shape (B, T, C)
            token_mask: Optional mask of shape (B, T) where True/1 indicates valid tokens
                       and False/0 indicates tokens to mask out (missing tokens).
                       If None, all tokens are considered valid.
        """
        B, T, C = x.shape # batch size, seq length, h_dim * n_heads

        N, D = self.n_heads, C // self.n_heads # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1,2)
        k = self.k_net(x).view(B, T, N, D).transpose(1,2)
        v = self.v_net(x).view(B, T, N, D).transpose(1,2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2,3) / math.sqrt(D)
        # causal mask applied to weights
        weights = weights.masked_fill(self.mask[...,:T,:T] == 0, float('-inf'))
        
        # Apply token mask if provided (mask out attention to missing tokens)
        if token_mask is not None:
            # token_mask: (B, T) -> expand to (B, 1, 1, T) for broadcasting
            # We want to mask the keys/values (columns), so missing tokens can't be attended to
            token_mask_expanded = token_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            weights = weights.masked_fill(token_mask_expanded == 0, float('-inf'))
        
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))

        assert not torch.isnan(out).any(), "Output contains NaN"
        

        return out


class Block(nn.Module):
    def __init__(self, h_dim, max_T, n_heads, drop_p):
        super().__init__()
        self.attention = MaskedCausalAttention(h_dim, max_T, n_heads, drop_p)
        self.mlp = nn.Sequential(
                nn.Linear(h_dim, 4*h_dim),
                nn.GELU(),
                nn.Linear(4*h_dim, h_dim),
                nn.Dropout(drop_p),
            )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x, token_mask: Optional[torch.Tensor] = None):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x, token_mask) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class Transformer(nn.Module):
    # Class attribute annotations for TorchScript
    state_token_names: List[str]
    _state_mean_dict: Optional[Dict[str, torch.Tensor]]
    _state_std_dict: Optional[Dict[str, torch.Tensor]]
    
    def __init__(self, state_token_dims, act_dim, n_blocks, h_dim, context_len,
                 n_heads, drop_p, max_timestep=4096, state_mean=None, state_std=None,
                 use_action_tanh=False, shared_state_embedding=True, state_token_names=None):
        super().__init__()

        self.act_dim = act_dim
        self.h_dim = h_dim
        self.max_timestep = max_timestep
        self.shared_state_embedding = shared_state_embedding
        
        # Store state token names (keys for the state dictionary)
        if state_token_names is None:
            # Default names if not provided
            state_token_names = [f"token_{i}" for i in range(len(state_token_dims))]
        self.state_token_names: List[str] = state_token_names
        
        # Register as buffer for TorchScript compatibility
        self.register_buffer('state_token_dims', torch.tensor(state_token_dims, dtype=torch.long))
        self.num_state_tokens = len(state_token_dims)
        
        # Calculate input sequence length based on token configuration
        input_seq_len = (self.num_state_tokens + 1) * context_len  # +1 for action token

        ### transformer blocks (use ModuleList to support passing mask through)
        self.transformer = nn.ModuleList([Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)])

        ### projection heads (project to embedding)
        self.embed_ln = nn.LayerNorm(h_dim)
        self.embed_timestep = nn.Embedding(max_timestep, h_dim)
        self.embed_rtg = torch.nn.Linear(1, h_dim)
        
        # Create state embedding layers based on token dimensions
        if self.shared_state_embedding:
            # All tokens share the same embedding layer
            unique_dims = torch.unique(self.state_token_dims)
            if len(unique_dims) > 1:
                raise ValueError("When using shared_state_embedding=True, all state tokens must have the same dimension. "
                               f"Got dimensions: {self.state_token_dims}")
            self.shared_state_embed = nn.Linear(int(self.state_token_dims[0].item()), h_dim)
            # Create empty ModuleDict for TorchScript compatibility
            self.state_embedding_layers = nn.ModuleDict()
        else:
            # Each token has its own embedding layer (using dict with token names as keys)
            self.state_embedding_layers = nn.ModuleDict()
            for i, name in enumerate(self.state_token_names):
                dim = int(self.state_token_dims[i].item())
                self.state_embedding_layers[name] = nn.Linear(dim, h_dim)
            # Create a dummy layer for TorchScript compatibility
            self.shared_state_embed = nn.Linear(1, h_dim)  # Dummy layer, won't be used

        # # discrete actions
        # self.embed_action = torch.nn.Embedding(act_dim, h_dim)
        # use_action_tanh = False # False for discrete actions

        # continuous actions
        self.embed_action = torch.nn.Linear(act_dim, h_dim)

        ### prediction heads
        # self.predict_rtg = torch.nn.Linear(h_dim, 1)
        # self.predict_state = torch.nn.Linear(h_dim, self.state_dim)
        # self.predict_action = nn.Linear(h_dim, act_dim)
        self.predict_action = nn.Sequential(
            *([nn.Linear(h_dim, act_dim)] + ([nn.Tanh()] if use_action_tanh else []))
        )

        # Handle state_mean and state_std - register as buffers for TorchScript compatibility
        # Initialize dict versions (for non-JIT use and checkpoint saving)
        self._state_mean_dict = None
        self._state_std_dict = None
        
        if state_mean is not None:
            if isinstance(state_mean, dict):
                # Store dict version for non-JIT use and for checkpoint saving
                self._state_mean_dict = {
                    key: torch.as_tensor(value, dtype=torch.float32) for key, value in state_mean.items()
                }
                # Register concatenated buffer for TorchScript
                mean_tensors = [self._state_mean_dict[name] for name in self.state_token_names]
                self.register_buffer('_state_mean_concat', torch.cat(mean_tensors))
            else:
                self.register_buffer('_state_mean_concat', torch.as_tensor(state_mean, dtype=torch.float32))
        else:
            self.register_buffer('_state_mean_concat', torch.zeros(1))  # Dummy buffer
            
        if state_std is not None:
            if isinstance(state_std, dict):
                self._state_std_dict = {
                    key: torch.as_tensor(value, dtype=torch.float32) for key, value in state_std.items()
                }
                std_tensors = [self._state_std_dict[name] for name in self.state_token_names]
                self.register_buffer('_state_std_concat', torch.cat(std_tensors))
            else:
                self.register_buffer('_state_std_concat', torch.as_tensor(state_std, dtype=torch.float32))
        else:
            self.register_buffer('_state_std_concat', torch.ones(1))  # Dummy buffer
    
    @property
    def state_mean(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return state_mean as dict for backward compatibility."""
        return self._state_mean_dict
    
    @property  
    def state_std(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return state_std as dict for backward compatibility."""
        return self._state_std_dict


    def forward(self, timesteps: torch.Tensor, states: Dict[str, torch.Tensor], actions: torch.Tensor, 
                state_mask: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the Decision Transformer.
        
        Args:
            timesteps: Tensor of shape (B, T) with timestep indices
            states: Dictionary where keys are state token names (e.g., 'position', 'velocity')
                   and values are tensors of shape (B, T, token_dim) where token_dim can vary per token.
                   Missing tokens should still be present in the dict but can contain zeros (padding).
            actions: Tensor of shape (B, T, act_dim)
            state_mask: Optional dictionary with same keys as states, where values are boolean tensors
                       of shape (B, T). True/1 indicates the token is present (valid), 
                       False/0 indicates the token is missing (will be masked in attention).
                       If None, all tokens are considered present.
                       
        Example with missing tokens:
            # Token 'velocity' is missing at timesteps 2,3 for batch item 0
            states = {
                'position': torch.randn(2, 5, 3),  # (B=2, T=5, dim=3)
                'velocity': torch.randn(2, 5, 3),  # padded with zeros where missing
            }
            state_mask = {
                'position': torch.ones(2, 5, dtype=torch.bool),   # all present
                'velocity': torch.tensor([[1,1,0,0,1], [1,1,1,1,1]], dtype=torch.bool),  # some missing
            }
        """
        
        # Get batch size and sequence length from the first state token
        first_token_name = self.state_token_names[0]
        first_token = states[first_token_name]
        B, T, _ = first_token.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        action_embeddings = self.embed_action(actions.nan_to_num()) + time_embeddings

        # Process each state token and build token-level mask
        state_embeddings: List[torch.Tensor] = []
        token_masks: List[torch.Tensor] = []
        
        if self.shared_state_embedding:
            # Use shared embedding for all tokens
            for token_name in self.state_token_names:
                state_token = states[token_name]  # (B, T, token_dim)
                embedding = self.shared_state_embed(state_token) + time_embeddings
                state_embeddings.append(embedding)
                
                # Get mask for this token (default to all True if not provided)
                if state_mask is not None and token_name in state_mask:
                    token_masks.append(state_mask[token_name])
                else:
                    token_masks.append(torch.ones(B, T, dtype=torch.bool, device=first_token.device))
        else:
            # Use separate embeddings - enumerate ModuleDict for TorchScript compatibility
            for token_name, embed_layer in self.state_embedding_layers.items():
                state_token = states[token_name]  # (B, T, token_dim)
                embedding = embed_layer(state_token) + time_embeddings
                state_embeddings.append(embedding)
                
                # Get mask for this token
                if state_mask is not None and token_name in state_mask:
                    token_masks.append(state_mask[token_name])
                else:
                    token_masks.append(torch.ones(B, T, dtype=torch.bool, device=first_token.device))
        
        # Stack all embeddings: state tokens + action token
        state_embeddings.append(action_embeddings)
        # Action tokens are always present
        token_masks.append(torch.ones(B, T, dtype=torch.bool, device=first_token.device))
        
        num_tokens = len(state_embeddings)
        # h shape: (B, num_tokens, T, h_dim) -> (B, T, num_tokens, h_dim) -> (B, T*num_tokens, h_dim)
        # But we want interleaved: for each timestep, all tokens in order
        # Current: stack gives (B, num_tokens, T, h_dim)
        # We want: (B, T, num_tokens, h_dim) then reshape to (B, T*num_tokens, h_dim)
        h = torch.stack(state_embeddings, dim=1).permute(0, 2, 1, 3).reshape(B, num_tokens * T, self.h_dim)
        
        # Build the flattened attention mask: (B, num_tokens * T)
        # Same interleaving as embeddings
        if state_mask is not None:
            # Stack masks: (num_tokens, B, T) -> (B, T, num_tokens) -> (B, T*num_tokens)
            stacked_masks = torch.stack(token_masks, dim=0)  # (num_tokens, B, T)
            stacked_masks = stacked_masks.permute(1, 2, 0)   # (B, T, num_tokens)
            attention_mask = stacked_masks.reshape(B, T * num_tokens)  # (B, T*num_tokens)
        else:
            attention_mask = None

        h = self.embed_ln(h)

        # transformer and prediction (pass mask through each block)
        for block in self.transformer:
            h = block(h, attention_mask)

        # Reshape to get output embeddings
        num_tokens = self.num_state_tokens + 1  # +1 for action token
        h = h.reshape(B, T, num_tokens, self.h_dim).permute(0, 2, 1, 3)
        
        # Predict action using the embedding from the last state token
        # (before the action token in the sequence)
        action_preds = self.predict_action(h[:, -2])  # -2 because -1 is action, -2 is last state token

        return None, action_preds, None


    # def _setup_state_tokens(self, independent_module_tokens, num_state_tokens, state_token_dims):
    #     """
    #     Setup state token configuration.
        
    #     Args:
    #         independent_module_tokens: Legacy boolean flag for backward compatibility
    #         num_state_tokens: Number of state tokens to expect
    #         state_token_dims: List of dimensions for each state token
    #     """
    #     if num_state_tokens is not None and state_token_dims is not None:
    #         # Use new flexible configuration
    #         if len(state_token_dims) != num_state_tokens:
    #             raise ValueError(f"Length of state_token_dims ({len(state_token_dims)}) must match num_state_tokens ({num_state_tokens})")
            
    #         self.num_state_tokens = num_state_tokens
    #         self.state_token_dims = state_token_dims
                    
    #     elif independent_module_tokens:
    #         # Legacy behavior: split state into equal modules
    #         self.num_modules = 5  # Default value, can be made configurable
    #         module_dim = self.state_dim // self.num_modules
    #         self.num_state_tokens = self.num_modules
    #         self.state_token_dims = [module_dim] * self.num_modules
    #     else:
    #         # Single token for entire state
    #         self.num_state_tokens = 1
    #         self.state_token_dims = [self.state_dim]
    
    def _create_state_embeddings(self):
        """Create embedding layers for each state token."""
        self.state_embedding_layers = nn.ModuleList()
        for dim in self.state_token_dims:
            self.state_embedding_layers.append(nn.Linear(dim, self.h_dim))
    
    def get_inference(self, device: str = None, context_len: int = None) -> 'TransformerInference':
        """
        Create an inference wrapper for easy step-by-step prediction.
        
        The inference wrapper handles context window management automatically,
        so you can simply feed in the current state at each timestep.
        
        Args:
            device: Device to run inference on ('cpu' or 'cuda'). 
                   If None, uses the device of the model's parameters.
            context_len: Context length. If None, inferred from model architecture.
        
        Returns:
            TransformerInference instance
        
        Example:
            model = Transformer(...)
            model.load_state_dict(torch.load('model.pt'))
            
            inference = model.get_inference(device='cuda')
            
            for t in range(episode_length):
                state = {'position': pos, 'velocity': vel}
                prediction = inference.step(state)
                # use prediction...
            
            inference.reset()  # for new episode
        """
        if device is None:
            # Get device from model parameters
            device = next(self.parameters()).device
        
        return TransformerInference(self, device=str(device), context_len=context_len)


class TransformerInference:
    """
    Inference wrapper for Transformer that automatically manages state/action history.
    
    This class handles the context window management so you can simply feed in
    the current state at each timestep and get the predicted output.
    
    Note: This wrapper automatically normalizes input states using the model's
    stored normalization statistics. Predictions are returned in the original
    (un-normalized) scale since targets are not normalized during training.
    
    Example usage:
        # Load trained model
        model = Transformer(...)
        model.load_state_dict(torch.load('model.pt'))
        
        # Create inference wrapper
        inference = TransformerInference(model, device='cuda')
        
        # At each timestep, just call step() with current state
        for t in range(episode_length):
            current_state = {
                'position': get_position(),  # numpy array or tensor of shape (dim,)
                'velocity': get_velocity(),
            }
            predicted_output = inference.step(current_state)
            
            # Use predicted_output for control...
            
        # Reset for new episode
        inference.reset()
    """
    
    def __init__(self, model: Transformer, device: str = 'cpu', context_len: int = None):
        """
        Initialize the inference wrapper.
        
        Args:
            model: Trained Transformer model
            device: Device to run inference on ('cpu' or 'cuda')
            context_len: Context length (if None, inferred from model)
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Get model configuration
        self.state_token_names = model.state_token_names
        self.state_token_dims = [int(d.item()) for d in model.state_token_dims]
        self.act_dim = model.act_dim
        
        # Get normalization statistics from model
        self.state_mean = model.state_mean  # Dict[str, Tensor] or None
        self.state_std = model.state_std    # Dict[str, Tensor] or None
        
        # Infer context length from model's transformer block
        if context_len is not None:
            self.context_len = context_len
        else:
            # Try to infer from the attention mask size
            # input_seq_len = (num_state_tokens + 1) * context_len
            num_tokens = model.num_state_tokens + 1
            input_seq_len = model.transformer[0].attention.max_T
            self.context_len = input_seq_len // num_tokens
        
        # Initialize history buffers
        self.reset()
    
    def reset(self):
        """Reset the history buffers for a new episode."""
        self.timestep = 0
        
        # Initialize state history as dict of lists
        self.state_history = {
            name: [] for name in self.state_token_names
        }
        
        # Initialize mask history (tracks which tokens are present at each timestep)
        self.mask_history = {
            name: [] for name in self.state_token_names
        }
        
        # Initialize action history (predictions)
        self.action_history = []
    
    def _normalize_state(self, name: str, value: torch.Tensor) -> torch.Tensor:
        """
        Normalize a state tensor using stored statistics.
        
        Args:
            name: State token name
            value: Raw state tensor
            
        Returns:
            Normalized state tensor
        """
        if self.state_mean is not None and self.state_std is not None:
            if name in self.state_mean and name in self.state_std:
                mean = self.state_mean[name]
                std = self.state_std[name]
                
                # Move to same device as value if needed
                if mean.device != value.device:
                    mean = mean.to(value.device)
                    std = std.to(value.device)
                try:
                    value - mean
                except:
                    import pdb; pdb.set_trace()
                return (value - mean) / std
        
        # No normalization available
        return value
    
    def _prepare_context(self):
        """Prepare the context tensors for model forward pass."""
        # Determine how many timesteps we have
        n_steps = len(self.action_history) + 1  # +1 for current state without action yet
        context_steps = min(n_steps, self.context_len)
        
        # Prepare states dict and mask dict
        states = {}
        state_mask = {}
        has_any_missing = False
        
        for name in self.state_token_names:
            # Get last context_steps states
            state_list = self.state_history[name][-context_steps:]
            mask_list = self.mask_history[name][-context_steps:]
            
            # Stack and add batch dimension: (context_steps, dim) -> (1, context_steps, dim)
            states[name] = torch.stack(state_list, dim=0).unsqueeze(0).to(self.device)
            
            # Stack masks: (context_steps,) -> (1, context_steps)
            state_mask[name] = torch.stack(mask_list, dim=0).unsqueeze(0).to(self.device)
            
            if not state_mask[name].all():
                has_any_missing = True
        
        # Prepare actions - pad with zeros for current timestep (we don't have action yet)
        if len(self.action_history) >= context_steps:
            actions_list = self.action_history[-(context_steps):]
        else:
            actions_list = self.action_history.copy()
        
        # Pad actions to match state length (add zero action for current timestep)
        while len(actions_list) < context_steps:
            actions_list.append(torch.zeros(self.act_dim))
        
        actions = torch.stack(actions_list, dim=0).unsqueeze(0).to(self.device)
        
        # Prepare timesteps
        start_t = max(0, self.timestep - context_steps + 1)
        timesteps = torch.arange(start_t, start_t + context_steps, dtype=torch.long)
        timesteps = timesteps.unsqueeze(0).to(self.device)
        
        # Only return mask if there are missing tokens (optimization)
        if not has_any_missing:
            state_mask = None

        
        return timesteps, states, actions, context_steps, state_mask
    
    @torch.no_grad()
    def step(self, current_state: dict, return_numpy: bool = True):
        """
        Process current state and return predicted output.
        
        Supports missing tokens - if a token is not in current_state or is None,
        it will be padded with zeros and masked out in attention.
        
        Args:
            current_state: Dictionary mapping state token names to their values.
                          Values can be numpy arrays or tensors of shape (dim,).
                          Missing tokens can be omitted from the dict or set to None.
            return_numpy: If True, return numpy array; otherwise return tensor.
        
        Returns:
            Predicted output (e.g., action, velocity, position) as numpy array or tensor
            of shape (target_dim,)
            
        Example with missing tokens:
            # Token 'velocity' is missing at this timestep
            current_state = {
                'position': np.array([1.0, 2.0, 3.0]),
                # 'velocity' is omitted or can be set to None
            }
            prediction = inference.step(current_state)
        """
        # Convert current state to tensors, normalize, and add to history
        for idx, name in enumerate(self.state_token_names):
            # Check if token is present
            is_present = name in current_state and current_state[name] is not None
            
            if is_present:
                value = current_state[name]
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value).float()
                elif not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32)
                
                # Normalize the input state
                value = self._normalize_state(name, value)
                self.mask_history[name].append(torch.tensor(True))
            else:
                print(f"Token '{name}' is missing at timestep {self.timestep}, padding with zeros.")
                # Token is missing - pad with zeros
                token_dim = self.state_token_dims[idx]
                value = torch.zeros(token_dim, dtype=torch.float32)
                self.mask_history[name].append(torch.tensor(False))
            
            self.state_history[name].append(value)
        
        # Prepare context for model
        timesteps, states, actions, context_steps, state_mask = self._prepare_context()
        
        # Forward pass with optional mask
        _, action_preds, _ = self.model.forward(timesteps, states, actions, state_mask)
        
        # Get prediction for current timestep (last position in sequence)
        predicted = action_preds[0, -1]  # Remove batch dim, get last timestep
        
        # Store predicted action in history for next step
        self.action_history.append(predicted.cpu())
        
        # Increment timestep
        self.timestep += 1
        
        # Trim history to avoid memory growth (keep 2x context_len for safety)
        max_history = self.context_len * 2
        if len(self.action_history) > max_history:
            self.action_history = self.action_history[-max_history:]
            for name in self.state_token_names:
                self.state_history[name] = self.state_history[name][-max_history:]
                self.mask_history[name] = self.mask_history[name][-max_history:]
        
        if return_numpy:
            return predicted.cpu().numpy()
        return predicted
    
    def get_history(self):
        """
        Get the full history of states, masks, and predictions.
        
        Returns:
            dict with 'states' (dict of lists), 'masks' (dict of lists), and 'actions' (list of tensors)
        """
        return {
            'states': {name: [s.numpy() for s in states] 
                      for name, states in self.state_history.items()},
            'masks': {name: [m.item() for m in masks]
                     for name, masks in self.mask_history.items()},
            'actions': [a.numpy() for a in self.action_history]
        }


if __name__ == "__main__":

    # Example with dictionary states
    model = Transformer(
        state_token_dims=[8,8,8,8,8], 
        state_token_names=['module_0', 'module_1', 'module_2', 'module_3', 'module_4'],
        act_dim=5,
        n_blocks=1,
        h_dim=128,
        context_len=60,
        n_heads=1,
        drop_p=0.1,
        shared_state_embedding=True
    )
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(0.23)

    model.state_dict()

    # Create states as a dictionary
    states_dict = {
        'module_0': torch.zeros((256, 60, 8)),
        'module_1': torch.zeros((256, 60, 8)),
        'module_2': torch.zeros((256, 60, 8)),
        'module_3': torch.zeros((256, 60, 8)),
        'module_4': torch.zeros((256, 60, 8)),
    }

    out = model.forward(
        timesteps=torch.zeros((256, 60), dtype=torch.int64),  # 60 timesteps
        states=states_dict,  # Dictionary of state tokens
        actions=torch.zeros((256, 60, 5))     # action dimension is 5
    )

    print(out[1].shape)

    # ============================================
    # Example: Missing Tokens with state_mask
    # ============================================
    print("\n--- Missing Tokens Example (Training) ---")
    
    # Create a mask where module_2 is missing at some timesteps
    # Pattern: A,B,_,D,E repeats (module_2 missing)
    B, T = 4, 10
    state_mask = {
        'module_0': torch.ones(B, T, dtype=torch.bool),
        'module_1': torch.ones(B, T, dtype=torch.bool),
        'module_2': torch.zeros(B, T, dtype=torch.bool),  # Always missing
        'module_3': torch.ones(B, T, dtype=torch.bool),
        'module_4': torch.ones(B, T, dtype=torch.bool),
    }
    # Make module_2 present at some timesteps for variety
    state_mask['module_2'][:, 0:3] = True  # Present at first 3 timesteps
    
    states_with_missing = {
        'module_0': torch.randn(B, T, 8),
        'module_1': torch.randn(B, T, 8),
        'module_2': torch.randn(B, T, 8),  # Contains data, but masked where missing
        'module_3': torch.randn(B, T, 8),
        'module_4': torch.randn(B, T, 8),
    }
    
    out_with_mask = model.forward(
        timesteps=torch.arange(T).unsqueeze(0).expand(B, -1),
        states=states_with_missing,
        actions=torch.randn(B, T, 5),
        state_mask=state_mask  # Pass the mask!
    )
    print(f"Output shape with missing tokens: {out_with_mask[1].shape}")
    print(f"Module_2 mask pattern: {state_mask['module_2'][0]}")

    # ============================================
    # Example: Using TransformerInference wrapper
    # ============================================
    print("\n--- TransformerInference Example ---")
    
    # Create a simple model for inference demo
    inference_model = Transformer(
        state_token_dims=[2, 2],  # position and velocity, each 2D
        state_token_names=['position', 'velocity'],
        act_dim=2,
        n_blocks=1,
        h_dim=64,
        context_len=10,
        n_heads=1,
        drop_p=0.0,
        shared_state_embedding=False
    )
    
    # Get inference wrapper directly from model - much cleaner!
    inference = inference_model.get_inference(device='cpu')
    
    # Simulate a few timesteps of inference
    print(f"Context length: {inference.context_len}")
    print(f"State token names: {inference.state_token_names}")
    
    for t in range(15):
        # Simulate getting current state (e.g., from sensors)
        current_state = {
            'position': np.random.randn(2),  # Can use numpy arrays
            'velocity': np.random.randn(2),
        }
        
        # Get prediction - history is managed automatically!
        prediction = inference.step(current_state)
        
        if t < 3 or t >= 13:
            print(f"  t={t}: prediction shape = {prediction.shape}, value = {prediction}")
        elif t == 3:
            print("  ...")
    
    # Reset for new episode
    inference.reset()
    print("Reset for new episode - history cleared")
    
    # ============================================
    # Example: Inference with Missing Tokens
    # ============================================
    print("\n--- Inference with Missing Tokens ---")
    
    inference2 = inference_model.get_inference(device='cpu')
    
    for t in range(10):
        # Simulate sensor dropout - velocity missing at odd timesteps
        if t % 2 == 0:
            current_state = {
                'position': np.random.randn(2),
                'velocity': np.random.randn(2),  # Present
            }
        else:
            current_state = {
                'position': np.random.randn(2),
                # 'velocity' is missing (not in dict)
            }
        
        prediction = inference2.step(current_state)
        
        velocity_present = 'velocity' in current_state
        print(f"  t={t}: velocity={'present' if velocity_present else 'MISSING'}, "
              f"prediction={prediction}")
    
    # Check the mask history
    history = inference2.get_history()
    print(f"\nVelocity mask history: {history['masks']['velocity']}")


    # model2 = DecisionTransformerOld(
    #     state_dim=40,
    #     act_dim=5,
    #     n_blocks=1,
    #     h_dim=128,
    #     context_len=60,
    #     n_heads=1,
    #     drop_p=0.1,
    #     independent_module_tokens=True
    # )

    # with torch.no_grad():
    #     for param in model2.parameters():
    #         param.fill_(0.23)
    # model2.state_dict()
    # out2 = model2.forward(
    #     timesteps=torch.zeros((256, 60), dtype=torch.int64),  # 60 timesteps
    #     states=torch.zeros((256, 60, 40)),  # 5 tokens, each with dimension 8
    #     actions=torch.zeros((256, 60, 5))     # action dimension is 5
    # )

    # pdb.set_trace()