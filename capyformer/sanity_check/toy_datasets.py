"""
Toy Datasets for Sanity Checking Transformer Policy Training.

These datasets have simple, deterministic relationships between observations
and actions that should be easy for a transformer to learn. If the model
fails to learn these, there's likely a bug in the training pipeline.

Datasets:
1. ToyLinearDataset: action = W @ obs + b (simple linear relationship)
2. ToyModularDataset: Similar to robot module data (5 modules × 8 dims → 5 actions)
3. ToyPeriodicDataset: action = sin(obs) (tests nonlinear learning)

Copyright 2025 Chen Yu
"""

import numpy as np
import random
from typing import Optional, Dict, Any, List
from capyformer.data import TrajectoryDataset


class ToyLinearDataset(TrajectoryDataset):
    """
    Toy dataset with a simple linear relationship: action = W @ obs + b
    
    This is the simplest possible test - if the model can't learn this,
    there's definitely something wrong with the training pipeline.
    
    Dataset config options:
        num_trajectories: Number of trajectories to generate (default: 1000)
        traj_len: Length of each trajectory (default: 100)
        obs_dim: Observation dimension (default: 4)
        act_dim: Action dimension (default: 2)
        noise_std: Noise added to actions (default: 0.01)
    """
    
    def _setup_dataset(self, dataset_config):
        num_trajectories = dataset_config.get('num_trajectories', 1000)
        traj_len = dataset_config.get('traj_len', 100)
        obs_dim = dataset_config.get('obs_dim', 4)
        act_dim = dataset_config.get('act_dim', 2)
        noise_std = dataset_config.get('noise_std', 0.01)
        
        # Create a fixed linear mapping W and bias b
        # This is the "ground truth" that the model should learn
        np.random.seed(42)  # Fixed seed for reproducibility
        self.W = np.random.randn(act_dim, obs_dim).astype(np.float32) * 0.5
        self.b = np.random.randn(act_dim).astype(np.float32) * 0.1
        
        print(f"[ToyLinearDataset] Ground truth relationship: action = W @ obs + b")
        print(f"  W shape: {self.W.shape}, b shape: {self.b.shape}")
        print(f"  Generating {num_trajectories} trajectories...")
        
        self.trajectories = []
        
        for _ in range(num_trajectories):
            # Generate random observations (smooth random walk)
            obs = np.zeros((traj_len, obs_dim), dtype=np.float32)
            obs[0] = np.random.randn(obs_dim).astype(np.float32)
            
            for t in range(1, traj_len):
                # Random walk with momentum
                obs[t] = obs[t-1] + 0.1 * np.random.randn(obs_dim).astype(np.float32)
            
            # Compute actions using the linear relationship
            actions = (obs @ self.W.T + self.b).astype(np.float32)
            
            # Add small noise
            if noise_std > 0:
                actions += np.random.randn(*actions.shape).astype(np.float32) * noise_std
            
            self.trajectories.append({
                'observation': obs,
                'actions': actions,
            })
        
        # Set up dataset properties for TrajectoryDataset
        self.input_keys = ['observation']
        self.target_key = 'actions'
        self.pad_value = 0.0
        self.val_split = 0.1
        
        print(f"  Created {len(self.trajectories)} trajectories")
        print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}")
    
    def get_ground_truth_fn(self):
        """Return the ground truth function for evaluation."""
        W, b = self.W.copy(), self.b.copy()
        def ground_truth(obs):
            return obs @ W.T + b
        return ground_truth


class ToyModularDataset(TrajectoryDataset):
    """
    Toy dataset mimicking the modular robot structure (5 modules × 8 dims → 5 actions).
    
    Each module has 8 observation dimensions, and there's one action per module.
    The relationship is: action[i] = sum(module[i] * weights) + bias[i]
    
    This matches the structure in transformer_policy_multiple_rollouts.py
    
    Dataset config options:
        num_trajectories: Number of trajectories (default: 1000)
        traj_len: Trajectory length (default: 100)
        num_modules: Number of modules (default: 5)
        obs_per_module: Observations per module (default: 8)
        noise_std: Noise added to actions (default: 0.01)
    """
    
    def _setup_dataset(self, dataset_config):
        num_trajectories = dataset_config.get('num_trajectories', 1000)
        traj_len = dataset_config.get('traj_len', 100)
        num_modules = dataset_config.get('num_modules', 5)
        obs_per_module = dataset_config.get('obs_per_module', 8)
        noise_std = dataset_config.get('noise_std', 0.01)
        
        # Create fixed weights for each module
        np.random.seed(42)
        self.num_modules = num_modules
        self.obs_per_module = obs_per_module
        
        # Each module's action = weighted sum of its observations
        # W[i] has shape (obs_per_module,) for module i
        self.W = [np.random.randn(obs_per_module).astype(np.float32) * 0.3 
                  for _ in range(num_modules)]
        self.b = np.random.randn(num_modules).astype(np.float32) * 0.1
        
        print(f"[ToyModularDataset] Ground truth: action[i] = W[i] @ module[i] + b[i]")
        print(f"  {num_modules} modules × {obs_per_module} dims → {num_modules} actions")
        print(f"  Generating {num_trajectories} trajectories...")
        
        self.trajectories = []
        
        for _ in range(num_trajectories):
            traj_data = {}
            
            # Generate observations for each module
            for m in range(num_modules):
                module_obs = np.zeros((traj_len, obs_per_module), dtype=np.float32)
                module_obs[0] = np.random.randn(obs_per_module).astype(np.float32)
                
                for t in range(1, traj_len):
                    module_obs[t] = module_obs[t-1] + 0.1 * np.random.randn(obs_per_module).astype(np.float32)
                
                traj_data[f'module{m}'] = module_obs
            
            # Compute actions
            actions = np.zeros((traj_len, num_modules), dtype=np.float32)
            for m in range(num_modules):
                actions[:, m] = (traj_data[f'module{m}'] @ self.W[m] + self.b[m]).astype(np.float32)
            
            # Add noise
            if noise_std > 0:
                actions += np.random.randn(*actions.shape).astype(np.float32) * noise_std
            
            traj_data['actions'] = actions
            self.trajectories.append(traj_data)
        
        # Set up dataset properties
        self.input_keys = [f'module{i}' for i in range(num_modules)]
        self.target_key = 'actions'
        self.pad_value = 0.0
        self.val_split = 0.1
        
        print(f"  Created {len(self.trajectories)} trajectories")
    
    def get_ground_truth_fn(self):
        """Return the ground truth function for evaluation."""
        W = [w.copy() for w in self.W]
        b = self.b.copy()
        num_modules = self.num_modules
        
        def ground_truth(module_obs_dict):
            """
            Args:
                module_obs_dict: Dict with 'module0', 'module1', ... keys
            Returns:
                actions: (num_modules,) array
            """
            actions = np.zeros(num_modules, dtype=np.float32)
            for m in range(num_modules):
                module_key = f'module{m}'
                if module_key in module_obs_dict:
                    actions[m] = module_obs_dict[module_key] @ W[m] + b[m]
            return actions
        
        return ground_truth


class ToyPeriodicDataset(TrajectoryDataset):
    """
    Toy dataset with a nonlinear periodic relationship: action = sin(obs).
    
    Tests whether the model can learn nonlinear functions.
    
    Dataset config options:
        num_trajectories: Number of trajectories (default: 1000)
        traj_len: Trajectory length (default: 100)
        obs_dim: Observation dimension (default: 4)
        act_dim: Action dimension (default: 2)
        noise_std: Noise added to actions (default: 0.01)
    """
    
    def _setup_dataset(self, dataset_config):
        num_trajectories = dataset_config.get('num_trajectories', 1000)
        traj_len = dataset_config.get('traj_len', 100)
        obs_dim = dataset_config.get('obs_dim', 4)
        act_dim = dataset_config.get('act_dim', 2)
        noise_std = dataset_config.get('noise_std', 0.01)
        
        # Fixed projection matrix to map obs_dim → act_dim before sin
        np.random.seed(42)
        self.proj = np.random.randn(act_dim, obs_dim).astype(np.float32) * 0.5
        
        print(f"[ToyPeriodicDataset] Ground truth: action = sin(proj @ obs)")
        print(f"  Generating {num_trajectories} trajectories...")
        
        self.trajectories = []
        
        for _ in range(num_trajectories):
            # Generate observations
            obs = np.zeros((traj_len, obs_dim), dtype=np.float32)
            obs[0] = np.random.randn(obs_dim).astype(np.float32)
            
            for t in range(1, traj_len):
                obs[t] = obs[t-1] + 0.1 * np.random.randn(obs_dim).astype(np.float32)
            
            # Compute actions using sin
            actions = np.sin(obs @ self.proj.T).astype(np.float32)
            
            if noise_std > 0:
                actions += np.random.randn(*actions.shape).astype(np.float32) * noise_std
            
            self.trajectories.append({
                'observation': obs,
                'actions': actions,
            })
        
        self.input_keys = ['observation']
        self.target_key = 'actions'
        self.pad_value = 0.0
        self.val_split = 0.1
        
        print(f"  Created {len(self.trajectories)} trajectories")
    
    def get_ground_truth_fn(self):
        """Return the ground truth function."""
        proj = self.proj.copy()
        def ground_truth(obs):
            return np.sin(obs @ proj.T)
        return ground_truth


# ============================================================================
# Test code
# ============================================================================

if __name__ == "__main__":
    # Test ToyLinearDataset
    print("\n" + "=" * 60)
    print("Testing ToyLinearDataset")
    print("=" * 60)
    
    config = {'num_trajectories': 100, 'traj_len': 50}
    dataset = ToyLinearDataset(config, context_len=20)
    
    print(f"\nDataset info:")
    print(f"  Input token names: {dataset.input_token_names}")
    print(f"  Input token dims: {dataset.input_token_dims}")
    print(f"  Target dim: {dataset.target_dim}")
    print(f"  Num trajectories: {len(dataset)}")
    
    # Verify ground truth
    gt_fn = dataset.get_ground_truth_fn()
    traj = dataset.trajectories[0]
    obs = traj['observation'][0]
    action = traj['actions'][0]
    gt_action = gt_fn(obs)
    print(f"\nGround truth verification:")
    print(f"  Predicted action: {gt_action}")
    print(f"  Actual action: {action}")
    print(f"  Error: {np.linalg.norm(gt_action - action):.6f}")
    
    # Test ToyModularDataset
    print("\n" + "=" * 60)
    print("Testing ToyModularDataset")
    print("=" * 60)
    
    config = {'num_trajectories': 100, 'traj_len': 50}
    dataset = ToyModularDataset(config, context_len=20)
    
    print(f"\nDataset info:")
    print(f"  Input token names: {dataset.input_token_names}")
    print(f"  Input token dims: {dataset.input_token_dims}")
    print(f"  Target dim: {dataset.target_dim}")
    
    # Verify ground truth
    gt_fn = dataset.get_ground_truth_fn()
    traj = dataset.trajectories[0]
    module_obs = {f'module{i}': traj[f'module{i}'][0] for i in range(5)}
    action = traj['actions'][0]
    gt_action = gt_fn(module_obs)
    print(f"\nGround truth verification:")
    print(f"  Predicted action: {gt_action}")
    print(f"  Actual action: {action}")
    print(f"  Error: {np.linalg.norm(gt_action - action):.6f}")

