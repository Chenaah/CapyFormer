"""
Bearing Estimation Dataset for CapyFormer.

This module provides dataset classes for training transformers to predict
bearing (direction to target) from locomotion observations and ranging
information.

Example usage:
    from capyformer.data_utils.bearing_dataset import BearingDataset
    
    dataset = BearingDataset(
        {"data_path": "bearing_data.npz", "val_split": 0.1},
        context_len=20,
    )
    
    trainer = HFTrainer(dataset, model_name="google/gemma-3-270m")
    trainer.learn(n_epochs=100)

Copyright 2026 Chen Yu <chenyu@u.northwestern.edu>
"""

import numpy as np
from typing import Optional, List, Dict, Any

from capyformer.data import TrajectoryDataset


class BearingDataset(TrajectoryDataset):
    """
    Dataset for bearing estimation from locomotion observations and ranging info.
    
    Data format (NPZ file):
        - distance_history: (N, history_len) distance measurements
        - delta_history: (N, history_len) distance change measurements
        - bearing: (N,) target bearing angles in radians
        - locomotion_obs_history: (N, history_len, obs_dim) policy observations
        - num_episodes: number of episodes in data
    
    The transformer processes sequences of (locomotion_obs, ranging) to predict
    bearing represented as [sin(θ), cos(θ)].
    
    Trajectory Format Options:
        1. "per_timestep" (default): Each position in the history is a separate
           timestep in the trajectory. This allows the transformer to attend
           across the full history.
           
        2. "flat": History is flattened into a single feature vector per sample.
           Multiple consecutive samples form a trajectory.
    """
    
    def __init__(self, dataset_config: Dict[str, Any], context_len: int):
        """
        Args:
            dataset_config: Dict containing:
                - data_path: Path to .npz file
                - val_split: Validation split ratio (default: 0.1)
                - trajectory_format: "per_timestep" or "flat" (default: "per_timestep")
                - max_trajectories: Max number of trajectories (default: None = all)
                - episode_length: Hardcoded episode length (default: auto-detect from data)
            context_len: Context window length for transformer
        """
        super().__init__(dataset_config, context_len)
    
    def _setup_dataset(self, dataset_config: Dict[str, Any]):
        """Load and preprocess the bearing estimation data."""
        data_path = dataset_config['data_path']
        trajectory_format = dataset_config.get('trajectory_format', 'per_timestep')
        max_trajectories = dataset_config.get('max_trajectories', None)
        
        print(f"Loading bearing data from: {data_path}")
        loaded = np.load(data_path)
        
        # Extract data arrays
        distance_history = loaded['distance_history']  # (N, history_len)
        delta_history = loaded['delta_history']  # (N, history_len)
        bearings = loaded['bearing']  # (N,)
        locomotion_obs_history = loaded['locomotion_obs_history']  # (N, history_len, obs_dim)
        
        n_samples, history_len = distance_history.shape
        obs_dim = locomotion_obs_history.shape[-1]
        
        print(f"  Loaded {n_samples} samples, history_len={history_len}, obs_dim={obs_dim}")
        
        # Get episode boundaries from data or config
        episode_length = dataset_config.get('episode_length', None)
        if episode_length is None:
            # Auto-detect from data
            if 'num_episodes' in loaded.files:
                num_episodes = int(loaded['num_episodes'])
                episode_length = n_samples // num_episodes
                print(f"  Auto-detected episode_length={episode_length} from {num_episodes} episodes")
            else:
                # Default fallback
                episode_length = 1000
                print(f"  Using default episode_length={episode_length}")
        else:
            print(f"  Using configured episode_length={episode_length}")
        
        # Create episode boundaries based on fixed episode length
        episode_boundaries = self._get_episode_boundaries(n_samples, episode_length)
        n_episodes = len(episode_boundaries)
        print(f"  Created {n_episodes} episodes")
        
        # Convert bearings to sin/cos
        bearing_sin_cos = np.stack([np.sin(bearings), np.cos(bearings)], axis=-1)
        
        # Create trajectories based on format
        if trajectory_format == 'per_timestep':
            self._create_per_timestep_trajectories(
                distance_history, delta_history, locomotion_obs_history,
                bearing_sin_cos, episode_boundaries, max_trajectories
            )
        else:
            self._create_flat_trajectories(
                distance_history, delta_history, locomotion_obs_history,
                bearing_sin_cos, episode_boundaries, max_trajectories
            )
        
        self.pad_value = 0.0
        self.val_split = dataset_config.get('val_split', 0.1)
    
    def _get_episode_boundaries(self, n_samples: int, episode_length: int) -> List[int]:
        """
        Get episode boundaries based on fixed episode length.
        
        Args:
            n_samples: Total number of samples
            episode_length: Number of samples per episode
            
        Returns:
            List of indices where each episode starts
        """
        boundaries = list(range(0, n_samples, episode_length))
        return boundaries
    
    def _create_per_timestep_trajectories(
        self,
        distance_history: np.ndarray,
        delta_history: np.ndarray,
        locomotion_obs_history: np.ndarray,
        bearing_sin_cos: np.ndarray,
        episode_boundaries: List[int],
        max_trajectories: Optional[int],
    ):
        """
        Create trajectories where each history position is a separate timestep.
        
        For a sample with history_len=30:
        - locomotion_obs: (30, obs_dim) - one obs per timestep
        - ranging: (30, 2) - distance and delta per timestep
        - bearing: (30, 2) - replicated target (same for all timesteps)
        
        This allows the transformer to learn temporal patterns across history.
        """
        self.trajectories = []
        history_len = distance_history.shape[1]
        n_samples = len(bearing_sin_cos)
        
        # Add end boundary
        episode_boundaries = episode_boundaries + [n_samples]
        
        for ep_idx in range(len(episode_boundaries) - 1):
            start = episode_boundaries[ep_idx]
            end = episode_boundaries[ep_idx + 1]
            
            if end - start < 2:  # Skip very short episodes
                continue
            
            # Each sample in the episode becomes a trajectory
            for i in range(start, end):
                traj = {
                    # Locomotion obs for each timestep in history
                    'locomotion_obs': locomotion_obs_history[i].astype(np.float32),  # (history_len, obs_dim)
                    
                    # Ranging info: distance and delta for each timestep
                    'ranging': np.stack([
                        distance_history[i],
                        delta_history[i],
                    ], axis=-1).astype(np.float32),  # (history_len, 2)
                    
                    # Target: replicate bearing for all timesteps (predict at last)
                    'bearing': np.tile(
                        bearing_sin_cos[i], (history_len, 1)
                    ).astype(np.float32),  # (history_len, 2)
                }
                self.trajectories.append(traj)
                
                if max_trajectories and len(self.trajectories) >= max_trajectories:
                    break
            
            if max_trajectories and len(self.trajectories) >= max_trajectories:
                break
        
        print(f"  Created {len(self.trajectories)} per-timestep trajectories")
        
        self.input_keys = ['locomotion_obs', 'ranging']
        self.target_key = 'bearing'
    
    def _create_flat_trajectories(
        self,
        distance_history: np.ndarray,
        delta_history: np.ndarray,
        locomotion_obs_history: np.ndarray,
        bearing_sin_cos: np.ndarray,
        episode_boundaries: List[int],
        max_trajectories: Optional[int],
    ):
        """
        Create trajectories with flattened history as features.
        
        Multiple consecutive samples (across time) form a trajectory.
        Each sample has flattened history as input.
        """
        self.trajectories = []
        history_len = distance_history.shape[1]
        obs_dim = locomotion_obs_history.shape[-1]
        n_samples = len(bearing_sin_cos)
        
        # Flatten histories
        loco_flat = locomotion_obs_history.reshape(-1, history_len * obs_dim)
        ranging_flat = np.stack([distance_history, delta_history], axis=-1).reshape(-1, history_len * 2)
        
        # Add end boundary
        episode_boundaries = episode_boundaries + [n_samples]
        trajectory_length = 50  # Number of consecutive samples per trajectory
        stride = 25
        
        for ep_idx in range(len(episode_boundaries) - 1):
            start = episode_boundaries[ep_idx]
            end = episode_boundaries[ep_idx + 1]
            
            # Create trajectories from consecutive samples within episode
            for traj_start in range(start, end - trajectory_length + 1, stride):
                traj_end = traj_start + trajectory_length
                
                traj = {
                    'locomotion_history': loco_flat[traj_start:traj_end].astype(np.float32),
                    'ranging_history': ranging_flat[traj_start:traj_end].astype(np.float32),
                    'bearing': bearing_sin_cos[traj_start:traj_end].astype(np.float32),
                }
                self.trajectories.append(traj)
                
                if max_trajectories and len(self.trajectories) >= max_trajectories:
                    break
            
            if max_trajectories and len(self.trajectories) >= max_trajectories:
                break
        
        print(f"  Created {len(self.trajectories)} flat trajectories (length {trajectory_length})")
        
        self.input_keys = ['locomotion_history', 'ranging_history']
        self.target_key = 'bearing'


class BearingDatasetV2(TrajectoryDataset):
    """
    Alternative bearing dataset that creates sequential trajectories from 
    consecutive timesteps within episodes.
    
    Instead of using the pre-computed history per-sample, this dataset treats 
    consecutive samples as consecutive timesteps and lets the transformer 
    handle the history through its context window.
    
    Input tokens per timestep:
        - locomotion_obs: (obs_dim,) current observation
        - ranging: (2,) current distance and delta
    
    Target per timestep:
        - bearing: (2,) sin/cos of bearing angle
    """
    
    def __init__(self, dataset_config: Dict[str, Any], context_len: int):
        super().__init__(dataset_config, context_len)
    
    def _setup_dataset(self, dataset_config: Dict[str, Any]):
        """Load data and create sequential trajectories."""
        data_path = dataset_config['data_path']
        trajectory_length = dataset_config.get('trajectory_length', 100)
        stride = dataset_config.get('stride', 50)
        max_trajectories = dataset_config.get('max_trajectories', None)
        
        print(f"Loading bearing data from: {data_path}")
        loaded = np.load(data_path)
        
        distance_history = loaded['distance_history']
        delta_history = loaded['delta_history']
        bearings = loaded['bearing']
        locomotion_obs_history = loaded['locomotion_obs_history']
        
        n_samples, history_len = distance_history.shape
        obs_dim = locomotion_obs_history.shape[-1]
        
        print(f"  Loaded {n_samples} samples, history_len={history_len}, obs_dim={obs_dim}")
        
        # Get episode length from data or config
        episode_length = dataset_config.get('episode_length', None)
        if episode_length is None:
            if 'num_episodes' in loaded.files:
                num_episodes = int(loaded['num_episodes'])
                episode_length = n_samples // num_episodes
                print(f"  Auto-detected episode_length={episode_length} from {num_episodes} episodes")
            else:
                episode_length = 1000
                print(f"  Using default episode_length={episode_length}")
        else:
            print(f"  Using configured episode_length={episode_length}")
        
        # Use most recent values at each timestep
        locomotion_obs = locomotion_obs_history[:, -1, :]  # (N, obs_dim)
        distance = distance_history[:, -1]  # (N,)
        delta = delta_history[:, -1]  # (N,)
        
        bearing_sin_cos = np.stack([np.sin(bearings), np.cos(bearings)], axis=-1)
        
        # Get episode boundaries
        episode_boundaries = self._get_episode_boundaries(n_samples, episode_length)
        episode_boundaries = episode_boundaries + [n_samples]
        
        print(f"  Created {len(episode_boundaries) - 1} episodes")
        
        # Create trajectories within each episode
        self.trajectories = []
        
        for ep_idx in range(len(episode_boundaries) - 1):
            ep_start = episode_boundaries[ep_idx]
            ep_end = episode_boundaries[ep_idx + 1]
            
            for traj_start in range(ep_start, ep_end - trajectory_length + 1, stride):
                traj_end = traj_start + trajectory_length
                
                traj = {
                    'locomotion_obs': locomotion_obs[traj_start:traj_end].astype(np.float32),
                    'ranging': np.stack([
                        distance[traj_start:traj_end],
                        delta[traj_start:traj_end],
                    ], axis=-1).astype(np.float32),
                    'bearing': bearing_sin_cos[traj_start:traj_end].astype(np.float32),
                }
                self.trajectories.append(traj)
                
                if max_trajectories and len(self.trajectories) >= max_trajectories:
                    break
            
            if max_trajectories and len(self.trajectories) >= max_trajectories:
                break
        
        print(f"  Created {len(self.trajectories)} trajectories (length {trajectory_length})")
        
        self.input_keys = ['locomotion_obs', 'ranging']
        self.target_key = 'bearing'
        self.pad_value = 0.0
        self.val_split = dataset_config.get('val_split', 0.1)
    
    def _get_episode_boundaries(self, n_samples: int, episode_length: int) -> List[int]:
        """Get episode boundaries based on fixed episode length."""
        return list(range(0, n_samples, episode_length))
