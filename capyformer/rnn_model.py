"""
RNN baseline model for sequence prediction.
This is a simpler baseline compared to the Transformer model.
"""

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNModel(nn.Module):
    """
    RNN baseline model that processes sequential state observations 
    and predicts actions (e.g., velocities).
    
    Args:
        state_token_dims: List of dimensions for each state token type
        act_dim: Dimension of action space
        h_dim: Hidden dimension of RNN
        n_layers: Number of RNN layers
        rnn_type: Type of RNN ('LSTM', 'GRU', or 'RNN')
        drop_p: Dropout probability
        state_mean: Mean values for state normalization (dict or tensor)
        state_std: Standard deviation values for state normalization (dict or tensor)
        shared_state_embedding: Whether to share embedding across all state tokens
        state_token_names: Names of state tokens (keys for state dictionary)
    """
    
    def __init__(self, state_token_dims, act_dim, h_dim=256, n_layers=2, 
                 rnn_type='LSTM', drop_p=0.1, state_mean=None, state_std=None,
                 shared_state_embedding=True, state_token_names=None):
        super().__init__()
        
        self.act_dim = act_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.shared_state_embedding = shared_state_embedding
        
        # Store state token names (keys for the state dictionary)
        if state_token_names is None:
            state_token_names = [f"token_{i}" for i in range(len(state_token_dims))]
        self.state_token_names = state_token_names
        
        # Register as buffer for compatibility
        self.register_buffer('state_token_dims', torch.tensor(state_token_dims, dtype=torch.long))
        self.num_state_tokens = len(state_token_dims)
        
        # Create state embedding layers
        if self.shared_state_embedding:
            # All tokens share the same embedding layer
            unique_dims = torch.unique(self.state_token_dims)
            if len(unique_dims) > 1:
                raise ValueError("When using shared_state_embedding=True, all state tokens must have the same dimension. "
                               f"Got dimensions: {self.state_token_dims}")
            state_dim = int(self.state_token_dims[0].item())
            self.shared_state_embed = nn.Linear(state_dim, h_dim)
            self.state_embedding_layers = nn.ModuleDict()
        else:
            # Each token has its own embedding layer
            self.state_embedding_layers = nn.ModuleDict()
            for i, name in enumerate(self.state_token_names):
                dim = int(self.state_token_dims[i].item())
                self.state_embedding_layers[name] = nn.Linear(dim, h_dim)
            self.shared_state_embed = nn.Linear(1, h_dim)  # Dummy layer
        
        # Input projection: concatenate all embedded state tokens
        # After embedding, each token becomes h_dim, so total input is num_state_tokens * h_dim
        rnn_input_dim = self.num_state_tokens * h_dim
        
        # RNN layers
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                rnn_input_dim, 
                h_dim, 
                n_layers, 
                batch_first=True,
                dropout=drop_p if n_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                rnn_input_dim, 
                h_dim, 
                n_layers, 
                batch_first=True,
                dropout=drop_p if n_layers > 1 else 0
            )
        elif rnn_type == 'RNN':
            self.rnn = nn.RNN(
                rnn_input_dim, 
                h_dim, 
                n_layers, 
                batch_first=True,
                dropout=drop_p if n_layers > 1 else 0,
                nonlinearity='relu'
            )
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output layers
        self.dropout = nn.Dropout(drop_p)
        self.ln = nn.LayerNorm(h_dim)
        self.predict_action = nn.Linear(h_dim, act_dim)
        
        # Handle both dictionary and tensor formats for state_mean and state_std
        if state_mean is not None:
            if isinstance(state_mean, dict):
                self.state_mean = {key: torch.tensor(value) for key, value in state_mean.items()}
            else:
                self.state_mean = torch.tensor(state_mean)
        else:
            self.state_mean = None
            
        if state_std is not None:
            if isinstance(state_std, dict):
                self.state_std = {key: torch.tensor(value) for key, value in state_std.items()}
            else:
                self.state_std = torch.tensor(state_std)
        else:
            self.state_std = None
    
    def forward(self, states, hidden=None):
        """
        Forward pass of the RNN model.
        
        Args:
            states: Dictionary where keys are state token names (e.g., 'imu_quaternion', 'velocity')
                   and values are tensors of shape (B, T, token_dim)
            hidden: Optional initial hidden state for the RNN
                   - For LSTM: tuple of (h_0, c_0), each of shape (n_layers, B, h_dim)
                   - For GRU/RNN: tensor of shape (n_layers, B, h_dim)
        
        Returns:
            action_preds: Predicted actions of shape (B, T, act_dim)
            hidden: Final hidden state of the RNN
        """
        
        # Check if states is a dictionary
        if not isinstance(states, dict):
            raise ValueError("states must be a dictionary with state token names as keys")
        
        # Verify all expected state tokens are present
        if set(states.keys()) != set(self.state_token_names):
            raise ValueError(f"Expected state keys {self.state_token_names}, got {list(states.keys())}")
        
        # Get batch size and sequence length from the first state token
        first_token = states[self.state_token_names[0]]
        B, T = first_token.shape[0], first_token.shape[1]
        
        # Embed each state token
        embedded_tokens = []
        for token_name in self.state_token_names:
            state_token = states[token_name]
            
            # Embed the token
            if self.shared_state_embedding:
                embedded = self.shared_state_embed(state_token)
            else:
                embedded = self.state_embedding_layers[token_name](state_token)
            
            embedded_tokens.append(embedded)
        
        # Concatenate all embedded tokens along the feature dimension
        # Each token is (B, T, h_dim), concatenating gives (B, T, num_tokens * h_dim)
        state_embeddings = torch.cat(embedded_tokens, dim=-1)
        
        # Pass through RNN
        if self.rnn_type == 'LSTM':
            if hidden is None:
                rnn_out, (h_n, c_n) = self.rnn(state_embeddings)
                hidden = (h_n, c_n)
            else:
                rnn_out, (h_n, c_n) = self.rnn(state_embeddings, hidden)
                hidden = (h_n, c_n)
        else:  # GRU or RNN
            if hidden is None:
                rnn_out, h_n = self.rnn(state_embeddings)
                hidden = h_n
            else:
                rnn_out, h_n = self.rnn(state_embeddings, hidden)
                hidden = h_n
        
        # Apply layer norm and dropout
        rnn_out = self.ln(rnn_out)
        rnn_out = self.dropout(rnn_out)
        
        # Predict actions
        action_preds = self.predict_action(rnn_out)
        
        return action_preds, hidden
