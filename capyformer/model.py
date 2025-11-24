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

    def forward(self, x):
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
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B,T,N*D)

        out = self.proj_drop(self.proj_net(attention))
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

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x) # residual
        x = self.ln1(x)
        x = x + self.mlp(x) # residual
        x = self.ln2(x)
        return x


class Transformer(nn.Module):
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
        self.state_token_names = state_token_names
        
        # Register as buffer for TorchScript compatibility
        self.register_buffer('state_token_dims', torch.tensor(state_token_dims, dtype=torch.long))
        self.num_state_tokens = len(state_token_dims)
        
        # Calculate input sequence length based on token configuration
        input_seq_len = (self.num_state_tokens + 1) * context_len  # +1 for action token

        ### transformer blocks
        blocks = [Block(h_dim, input_seq_len, n_heads, drop_p) for _ in range(n_blocks)]
        self.transformer = nn.Sequential(*blocks)

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


    def forward(self, timesteps, states, actions):
        """
        Forward pass of the Decision Transformer.
        
        Args:
            timesteps: Tensor of shape (B, T) with timestep indices
            states: Dictionary where keys are state token names (e.g., 'position', 'velocity')
                   and values are tensors of shape (B, T, token_dim) where token_dim can vary per token
            actions: Tensor of shape (B, T, act_dim)
        """
        
        # Check if states is a dictionary
        if not isinstance(states, dict):
            raise ValueError("states must be a dictionary with state token names as keys")
        
        # Verify all expected state tokens are present
        if set(states.keys()) != set(self.state_token_names):
            raise ValueError(f"Expected state keys {self.state_token_names}, got {list(states.keys())}")
        
        # Get batch size and sequence length from the first state token
        first_token = states[self.state_token_names[0]]
        B, T, _ = first_token.shape

        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings

        # Process each state token
        state_embeddings = []
        if self.shared_state_embedding:
            # Use shared embedding for all tokens
            for token_name in self.state_token_names:
                state_token = states[token_name]  # (B, T, token_dim)
                embedding = self.shared_state_embed(state_token) + time_embeddings
                state_embeddings.append(embedding)
        else:
            # Use separate embeddings
            for token_name in self.state_token_names:
                state_token = states[token_name]  # (B, T, token_dim)
                embedding = self.state_embedding_layers[token_name](state_token) + time_embeddings
                state_embeddings.append(embedding)
        
        # Stack all embeddings: state tokens + action token
        h = torch.stack(
            state_embeddings + [action_embeddings], dim=1
        ).permute(0, 2, 1, 3).reshape(B, len(state_embeddings + [action_embeddings]) * T, self.h_dim)

        h = self.embed_ln(h)

        # transformer and prediction
        h = self.transformer(h)

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