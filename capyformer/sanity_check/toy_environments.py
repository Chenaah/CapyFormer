"""
Toy Environments for Sanity Checking Transformer Policy Inference.

These environments have simple dynamics that match the toy datasets,
allowing us to verify that the trained policy works correctly during
step-by-step inference.

Environments:
1. ToyEnvironment: Simple single-observation environment
2. ToyModularEnvironment: Mimics modular robot structure (5 modules)

Copyright 2025 Chen Yu
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, Callable
import gymnasium as gym
from gymnasium import spaces


class ToyEnvironment(gym.Env):
    """
    Simple toy environment for testing policy inference.
    
    Dynamics:
        obs[t+1] = obs[t] + action[t] * dt + noise
    
    This creates a simple control task where actions directly influence
    the observation state.
    
    Args:
        obs_dim: Observation dimension (default: 4)
        act_dim: Action dimension (default: 2)
        dt: Time step (default: 0.02)
        noise_std: Process noise (default: 0.01)
        max_steps: Maximum episode steps (default: 1000)
        ground_truth_fn: Optional function to compute "ideal" actions
    """
    
    def __init__(
        self,
        obs_dim: int = 4,
        act_dim: int = 2,
        dt: float = 0.02,
        noise_std: float = 0.01,
        max_steps: int = 1000,
        ground_truth_fn: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dt = dt
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.ground_truth_fn = ground_truth_fn
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(act_dim,), dtype=np.float32
        )
        
        self.state = None
        self.step_count = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Initialize state randomly
        self.state = np.random.randn(self.obs_dim).astype(np.float32) * 0.5
        self.step_count = 0
        
        return self.state.copy(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)
        
        # Simple dynamics: state evolves based on action
        # Map action to state change (first act_dim dimensions of state)
        state_change = np.zeros(self.obs_dim, dtype=np.float32)
        state_change[:min(self.act_dim, self.obs_dim)] = action[:min(self.act_dim, self.obs_dim)]
        
        self.state = self.state + state_change * self.dt
        self.state += np.random.randn(self.obs_dim).astype(np.float32) * self.noise_std
        
        self.step_count += 1
        
        # Compute reward based on ground truth if available
        reward = 0.0
        info = {}
        
        if self.ground_truth_fn is not None:
            gt_action = self.ground_truth_fn(self.state)
            action_error = np.linalg.norm(action - gt_action)
            reward = -action_error
            info['gt_action'] = gt_action
            info['action_error'] = action_error
        
        done = False
        truncated = self.step_count >= self.max_steps
        
        return self.state.copy(), reward, done, truncated, info
    
    def render(self):
        pass
    
    def close(self):
        pass


class ToyModularEnvironment(gym.Env):
    """
    Toy environment mimicking the modular robot structure.
    
    Observation: Dict with 'module0', 'module1', ..., 'module{N-1}' keys
    Action: (num_modules,) array
    
    Dynamics per module:
        module[i][t+1] = module[i][t] + action[i] * dt + noise
    
    Args:
        num_modules: Number of modules (default: 5)
        obs_per_module: Observations per module (default: 8)
        dt: Time step (default: 0.02)
        noise_std: Process noise (default: 0.01)
        max_steps: Maximum episode steps (default: 1000)
        ground_truth_fn: Optional function to compute "ideal" actions
        flat_observation: If True, return flat array. If False, return dict (default: False)
    """
    
    def __init__(
        self,
        num_modules: int = 5,
        obs_per_module: int = 8,
        dt: float = 0.02,
        noise_std: float = 0.01,
        max_steps: int = 1000,
        ground_truth_fn: Optional[Callable] = None,
        flat_observation: bool = False,
    ):
        super().__init__()
        
        self.num_modules = num_modules
        self.obs_per_module = obs_per_module
        self.dt = dt
        self.noise_std = noise_std
        self.max_steps = max_steps
        self.ground_truth_fn = ground_truth_fn
        self.flat_observation = flat_observation
        
        total_obs_dim = num_modules * obs_per_module
        
        # Define spaces
        if flat_observation:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32
            )
        else:
            self.observation_space = spaces.Dict({
                f'module{i}': spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_per_module,), dtype=np.float32
                )
                for i in range(num_modules)
            })
        
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(num_modules,), dtype=np.float32
        )
        
        self.module_states = None
        self.step_count = 0
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Any, Dict]:
        super().reset(seed=seed)
        
        # Initialize module states randomly
        self.module_states = {
            f'module{i}': np.random.randn(self.obs_per_module).astype(np.float32) * 0.5
            for i in range(self.num_modules)
        }
        self.step_count = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Return observation in the configured format."""
        if self.flat_observation:
            return np.concatenate([
                self.module_states[f'module{i}'] 
                for i in range(self.num_modules)
            ])
        else:
            return {k: v.copy() for k, v in self.module_states.items()}
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32)
        
        # Apply action to each module
        for i in range(self.num_modules):
            module_key = f'module{i}'
            # Action affects the first dimension of each module's state
            state_change = np.zeros(self.obs_per_module, dtype=np.float32)
            state_change[0] = action[i]
            
            self.module_states[module_key] = (
                self.module_states[module_key] + 
                state_change * self.dt +
                np.random.randn(self.obs_per_module).astype(np.float32) * self.noise_std
            )
        
        self.step_count += 1
        
        # Compute reward based on ground truth if available
        reward = 0.0
        info = {}
        
        if self.ground_truth_fn is not None:
            gt_action = self.ground_truth_fn(self.module_states)
            action_error = np.linalg.norm(action - gt_action)
            reward = -action_error
            info['gt_action'] = gt_action
            info['action_error'] = action_error
        
        done = False
        truncated = self.step_count >= self.max_steps
        
        return self._get_observation(), reward, done, truncated, info
    
    def render(self):
        pass
    
    def close(self):
        pass


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing ToyEnvironment")
    print("=" * 60)
    
    env = ToyEnvironment(obs_dim=4, act_dim=2)
    obs, _ = env.reset(seed=42)
    print(f"Initial obs shape: {obs.shape}")
    print(f"Initial obs: {obs}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step+1}: action={action}, obs={obs[:2]}...")
    
    print("\n" + "=" * 60)
    print("Testing ToyModularEnvironment (dict observation)")
    print("=" * 60)
    
    env = ToyModularEnvironment(num_modules=5, obs_per_module=8, flat_observation=False)
    obs, _ = env.reset(seed=42)
    print(f"Observation keys: {obs.keys()}")
    print(f"module0 shape: {obs['module0'].shape}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step+1}: action={action[:2]}..., module0={obs['module0'][:2]}...")
    
    print("\n" + "=" * 60)
    print("Testing ToyModularEnvironment (flat observation)")
    print("=" * 60)
    
    env = ToyModularEnvironment(num_modules=5, obs_per_module=8, flat_observation=True)
    obs, _ = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Expected: {5 * 8} = 40")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step+1}: action={action[:2]}..., obs={obs[:4]}...")

