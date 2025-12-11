import copy
from typing import Tuple
from collections import defaultdict
import glob
import multiprocessing
import os
import pdb
import random
import time
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
# from d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
from tqdm import tqdm, trange


# from convert_position_frame import convert_positions_to_imu_frame, normalize_trajectory_positions
# from load_trajectories import load_trajectories

def save_rollout(rollout, save_dir, file_name="rolloutN.npz"):
    # Convert (T,B,D) to (T*B, D)
    # The observation returned for the i-th environment when done[i] is true will in fact be the first observation of the next episode
    print("Processing rollout...")
    batch_size = np.shape(rollout["observations"])[1]
    rollout_reshaped = defaultdict(list)
    for b in trange(batch_size):

        # for the b-th environment
        obs = np.array(rollout["observations"])[:,b,:].tolist()
        act = np.array(rollout["actions"])[:,b,:].tolist()
        rew = np.array(rollout["rewards"])[:,b].tolist()
        done = np.array(rollout["dones"])[:,b].tolist()
        am = np.array(rollout["active_modules"])[:,b].tolist()

        # Set the first done to True (for the previous episode) and keep the last one as False
        if b != 0:
            done[0] = True
        # Remove False values from the end of the list
        while done and not done[-1]:  # Check if the list is not empty and the last element is False
            done.pop()
            obs.pop()
            act.pop()
            rew.pop()
            am.pop()
        # also delete the last step
        if done and b != batch_size-1:
            done.pop()
            obs.pop()
            act.pop()
            rew.pop()
            am.pop()

        rollout_reshaped["observations"].extend(obs)
        rollout_reshaped["actions"].extend(act)
        rollout_reshaped["rewards"].extend(rew)
        rollout_reshaped["dones"].extend(done)
        rollout_reshaped["active_modules"].extend(am)

    # np.savez_compressed(generate_unique_filename(os.path.join(save_dir, file_name)), **rollout_reshaped)
    np.savez_compressed(os.path.join(save_dir, file_name), **rollout_reshaped)



class TrajectoryDataset(Dataset):
    """
    Base class for trajectory datasets.
    
    Trajectories can be stored in two formats:
    
    1. New flat format (recommended):
       Each trajectory is a dict with arbitrary keys, each mapping to a (T, dim) array.
       Use `target_key` in dataset_config to specify which key is the prediction target.
       All other keys are treated as input observations.
       
       Example:
       ```
       trajectory = {
           'position': np.array of shape (T, 2),    # input
           'velocity': np.array of shape (T, 2),    # input  
           'acceleration': np.array of shape (T, 2) # target (to be predicted)
       }
       dataset_config = {'target_key': 'acceleration'}
       ```
    
    2. Legacy format (for backward compatibility):
       Each trajectory has 'observations' (dict or array) and 'actions' keys.
       'observations' contains input data, 'actions' is the prediction target.
       
       Example:
       ```
       trajectory = {
           'observations': {'position': ..., 'velocity': ...},
           'actions': np.array of shape (T, act_dim)
       }
       ```
    
    Properties:
        input_token_names: list of input token names (keys)
        input_token_dims: list of dimensions for each input token
        target_dim: dimension of the target (prediction output)
        
    For backward compatibility, these aliases are also available:
        state_token_names -> input_token_names
        state_token_dims -> input_token_dims
        act_dim -> target_dim
    """
    context_len: int
    input_token_dims: list
    input_token_names: list
    target_dim: int
    target_key: str

    def __init__(self, dataset_config, context_len):
        self.context_len = context_len
        
        # Get target key from config (defaults to 'actions' for legacy format)
        self.target_key = dataset_config.get('target_key', None)
        
        # Subclass should populate trajectories
        self._setup_dataset(dataset_config)
        
        # Detect and convert format if needed
        self._detect_and_normalize_format()
        
        # Parent class infers dataset properties from trajectories
        self._infer_dataset_properties()
        
        # Parent class handles normalization and validation
        self._compute_normalization_stats()
        self._normalize_trajectories()
        
        # Optional: split into train/validation sets
        self.val_split = dataset_config.get('val_split', None)
        if self.val_split is not None:
            self._split_train_val()
        else:
            self.val_trajectories = self.trajectories
        
        self._validate_dataset()
    
    # Backward compatibility properties
    @property
    def state_token_names(self):
        return self.input_token_names
    
    @state_token_names.setter
    def state_token_names(self, value):
        self.input_token_names = value
    
    @property
    def state_token_dims(self):
        return self.input_token_dims
    
    @state_token_dims.setter
    def state_token_dims(self, value):
        self.input_token_dims = value
    
    @property
    def act_dim(self):
        return self.target_dim
    
    @act_dim.setter
    def act_dim(self, value):
        self.target_dim = value

    def _setup_dataset(self, dataset_config):
        """
        Load the dataset from the given path.
        Subclasses should implement this method and populate:
        - self.trajectories: list of dicts
        
        Trajectory format options:
        
        1. New flat format (recommended):
           Each trajectory is a dict with arbitrary keys, each mapping to (T, dim) arrays.
           Set 'target_key' in dataset_config to specify which key is the prediction target.
           
           Example:
           ```
           self.trajectories.append({
               'position': position_array,      # shape (T, 2) - input
               'velocity': velocity_array,      # shape (T, 2) - input
               'acceleration': accel_array,     # shape (T, 2) - target
           })
           # In dataset_config: {'target_key': 'acceleration'}
           ```
        
        2. Legacy format (for backward compatibility):
           Each trajectory has 'observations' (dict) and 'actions' keys.
           
           Example:
           ```
           self.trajectories.append({
               'observations': {'position': ..., 'velocity': ...},
               'actions': actions_array,
           })
           ```
        
        The parent class will automatically infer:
        - self.input_token_dims: list of dimensions for each input token
        - self.input_token_names: list of names for each input token
        - self.target_dim: dimension of target (prediction output)
        
        Note: Subclasses can optionally set these properties manually if needed.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
    def _detect_and_normalize_format(self):
        """
        Detect whether trajectories use flat format or legacy format.
        For legacy format, convert internally to flat format for uniform processing.
        """
        if len(self.trajectories) == 0:
            raise ValueError("No trajectories found in dataset.")
        
        first_traj = self.trajectories[0]
        
        # Check if using legacy format (has 'observations' and 'actions' keys)
        if 'observations' in first_traj and 'actions' in first_traj:
            self._is_legacy_format = True
            self.target_key = 'actions'
            
            # Check if observations is a dict or array
            if isinstance(first_traj['observations'], dict):
                # Dict observations - set input keys from observation keys
                self._input_keys = list(first_traj['observations'].keys())
            else:
                # Array observations - will handle in _infer_dataset_properties
                self._input_keys = None
        else:
            # New flat format
            self._is_legacy_format = False
            
            if self.target_key is None:
                raise ValueError(
                    "For flat trajectory format, 'target_key' must be specified in dataset_config. "
                    "This indicates which key in the trajectory dict should be predicted."
                )
            
            if self.target_key not in first_traj:
                raise ValueError(
                    f"target_key '{self.target_key}' not found in trajectory. "
                    f"Available keys: {list(first_traj.keys())}"
                )
            
            # All keys except target_key are inputs (unless explicitly set via input_keys)
            if hasattr(self, 'input_keys') and self.input_keys is not None:
                self._input_keys = self.input_keys
            else:
                self._input_keys = [k for k in first_traj.keys() if k != self.target_key]
            
            if len(self._input_keys) == 0:
                raise ValueError(
                    f"No input keys found. Trajectory only contains target_key '{self.target_key}'."
                )
    
    def _infer_dataset_properties(self):
        """
        Infer dataset properties from trajectories.
        Only infers properties that haven't been manually set by subclass.
        """
        if len(self.trajectories) == 0:
            raise ValueError("No trajectories found in dataset. Cannot infer properties.")
        
        first_traj = self.trajectories[0]
        
        if self._is_legacy_format:
            # Legacy format handling
            # Infer target_dim if not already set
            if not hasattr(self, 'target_dim') or self.target_dim is None:
                self.target_dim = first_traj['actions'].shape[-1]
            
            # Check if observations is a dictionary
            is_dict_obs = isinstance(first_traj['observations'], dict)
            
            if is_dict_obs:
                # Dictionary observations
                if not hasattr(self, 'input_token_names') or self.input_token_names is None:
                    self.input_token_names = list(first_traj['observations'].keys())
                
                if not hasattr(self, 'input_token_dims') or self.input_token_dims is None:
                    self.input_token_dims = [
                        first_traj['observations'][name].shape[-1] 
                        for name in self.input_token_names
                    ]
            else:
                # Legacy tensor format
                if not hasattr(self, 'input_token_dims') or self.input_token_dims is None:
                    obs_shape = first_traj['observations'].shape
                    if len(obs_shape) == 2:
                        self.input_token_dims = [obs_shape[-1]]
                    elif len(obs_shape) == 3:
                        num_tokens = obs_shape[1]
                        token_dim = obs_shape[2]
                        self.input_token_dims = [token_dim] * num_tokens
                    else:
                        raise ValueError(f"Unexpected observation shape: {obs_shape}")
                
                if not hasattr(self, 'input_token_names') or self.input_token_names is None:
                    self.input_token_names = None
        else:
            # New flat format
            # Infer target_dim
            if not hasattr(self, 'target_dim') or self.target_dim is None:
                self.target_dim = first_traj[self.target_key].shape[-1]
            
            # Infer input token names and dims
            if not hasattr(self, 'input_token_names') or self.input_token_names is None:
                self.input_token_names = self._input_keys
            
            if not hasattr(self, 'input_token_dims') or self.input_token_dims is None:
                self.input_token_dims = [
                    first_traj[name].shape[-1] 
                    for name in self.input_token_names
                ]
    
    def _compute_normalization_stats(self):
        """
        Compute normalization statistics for the dataset.
        Handles both flat format and legacy formats.
        """
        if len(self.trajectories) == 0:
            raise ValueError("No trajectories found in dataset")
        
        first_traj = self.trajectories[0]
        
        if self._is_legacy_format:
            # Legacy format: check if observations is dict or array
            is_dict_obs = isinstance(first_traj['observations'], dict)
            
            if is_dict_obs:
                # Dictionary observations: compute stats for each token
                states_dict = {name: [] for name in self.input_token_names}
                for traj in self.trajectories:
                    for name in self.input_token_names:
                        states_dict[name].append(traj['observations'][name])
                
                self.input_mean = {}
                self.input_std = {}
                for name in self.input_token_names:
                    states_concat = np.concatenate(states_dict[name], axis=0)
                    self.input_mean[name] = np.mean(states_concat, axis=0)
                    self.input_std[name] = np.std(states_concat, axis=0) + 1e-6
            else:
                # Legacy tensor observations
                states = []
                for traj in self.trajectories:
                    states.append(traj['observations'])
                
                states = np.concatenate(states, axis=0)
                self.input_mean = np.mean(states, axis=0)
                self.input_std = np.std(states, axis=0) + 1e-6
        else:
            # New flat format: compute stats for each input token
            input_data = {name: [] for name in self.input_token_names}
            for traj in self.trajectories:
                for name in self.input_token_names:
                    input_data[name].append(traj[name])
            
            self.input_mean = {}
            self.input_std = {}
            for name in self.input_token_names:
                data_concat = np.concatenate(input_data[name], axis=0)
                self.input_mean[name] = np.mean(data_concat, axis=0)
                self.input_std[name] = np.std(data_concat, axis=0) + 1e-6
        
        # Backward compatibility: alias state_mean/state_std to input_mean/input_std
        self.state_mean = self.input_mean
        self.state_std = self.input_std
        
        print(f"Input mean: {self.input_mean}")
        print(f"Input std: {self.input_std}")
    
    def _normalize_trajectories(self):
        """
        Normalize all trajectories using computed statistics.
        Only input tokens are normalized, not target.
        """
        first_traj = self.trajectories[0]
        
        if self._is_legacy_format:
            is_dict_obs = isinstance(first_traj['observations'], dict)
            
            if is_dict_obs:
                # Dictionary observations: normalize each token
                for traj in self.trajectories:
                    for name in self.input_token_names:
                        traj['observations'][name] = (
                            traj['observations'][name] - self.input_mean[name]
                        ) / self.input_std[name]
            else:
                # Legacy tensor observations
                for traj in self.trajectories:
                    traj['observations'] = (
                        traj['observations'] - self.input_mean
                    ) / self.input_std
        else:
            # New flat format: normalize each input token
            for traj in self.trajectories:
                for name in self.input_token_names:
                    traj[name] = (traj[name] - self.input_mean[name]) / self.input_std[name]
    
    def _split_train_val(self):
        """
        Split trajectories into training and validation sets.
        
        The split can be specified as:
        - float (0.0-1.0): fraction of trajectories for validation
        - int: absolute number of trajectories for validation
        
        Trajectories are shuffled before splitting to ensure randomness.
        Results are stored in self.trajectories (train) and self.val_trajectories (val).
        """
        if self.val_split is None or self.val_split == 0:
            self.val_trajectories = []
            return
        
        total_trajs = len(self.trajectories)
        
        # Determine number of validation trajectories
        if isinstance(self.val_split, float):
            if not (0.0 < self.val_split < 1.0):
                raise ValueError(f"val_split as float must be between 0 and 1, got {self.val_split}")
            n_val = int(total_trajs * self.val_split)
        elif isinstance(self.val_split, int):
            if not (0 < self.val_split < total_trajs):
                raise ValueError(f"val_split as int must be between 0 and {total_trajs}, got {self.val_split}")
            n_val = self.val_split
        else:
            raise TypeError(f"val_split must be float or int, got {type(self.val_split)}")
        
        # Ensure at least one trajectory in each set
        if n_val == 0:
            print(f"Warning: val_split resulted in 0 validation trajectories. Using 1 instead.")
            n_val = 1
        if n_val >= total_trajs:
            print(f"Warning: val_split resulted in {n_val} validation trajectories (>= total). Using {total_trajs-1} instead.")
            n_val = total_trajs - 1
        
        # Shuffle trajectories for random split
        import random
        random.seed(42)  # For reproducibility
        indices = list(range(total_trajs))
        random.shuffle(indices)
        
        # Split indices
        val_indices = set(indices[:n_val])
        train_indices = set(indices[n_val:])
        
        # Split trajectories
        self.val_trajectories = [self.trajectories[i] for i in sorted(val_indices)]
        train_trajectories = [self.trajectories[i] for i in sorted(train_indices)]
        self.trajectories = train_trajectories
        
        print(f"Split dataset: {len(self.trajectories)} training trajectories, {len(self.val_trajectories)} validation trajectories")
    
    def _validate_dataset(self):
        """
        Validate the dataset structure and check for common issues.
        """
        if len(self.trajectories) == 0:
            raise ValueError("No trajectories in dataset")
        
        first_traj = self.trajectories[0]
        
        if self._is_legacy_format:
            # Legacy format validation
            is_dict_obs = isinstance(first_traj['observations'], dict)
            
            if is_dict_obs:
                # Validate dictionary observations
                if not hasattr(self, 'input_token_names') or self.input_token_names is None:
                    raise ValueError("input_token_names must be set for dictionary format")
                
                if len(self.input_token_names) != len(self.input_token_dims):
                    raise ValueError(f"Length of input_token_names ({len(self.input_token_names)}) must match "
                                   f"input_token_dims ({len(self.input_token_dims)})")
                
                # Check all trajectories have the same observation keys
                expected_keys = set(self.input_token_names)
                for i, traj in enumerate(self.trajectories):
                    actual_keys = set(traj['observations'].keys())
                    if actual_keys != expected_keys:
                        raise ValueError(f"Trajectory {i} observations have keys {actual_keys}, expected {expected_keys}")
        else:
            # New flat format validation
            if not hasattr(self, 'input_token_names') or self.input_token_names is None:
                raise ValueError("input_token_names must be set")
            
            if len(self.input_token_names) != len(self.input_token_dims):
                raise ValueError(f"Length of input_token_names ({len(self.input_token_names)}) must match "
                               f"input_token_dims ({len(self.input_token_dims)})")
            
            # Check all trajectories have the expected keys (expected can be a subset of actual)
            expected_keys = set(self.input_token_names + [self.target_key])
            for i, traj in enumerate(self.trajectories):
                actual_keys = set(traj.keys())
                if not expected_keys.issubset(actual_keys):
                    raise ValueError(f"Trajectory {i} has keys {actual_keys}, expected at least {expected_keys}")
        
        print(f"Dataset validation passed: {len(self.trajectories)} trajectories loaded")
        print(f"  Input token names: {self.input_token_names}")
        print(f"  Input token dims: {self.input_token_dims}")
        print(f"  Target key: {self.target_key}")
        print(f"  Target dim: {self.target_dim}")



    def get_state_stats(self):
        """
        Get normalization statistics for input tokens.
        Validates that no NaN values are present.
        
        Returns:
            tuple: (input_mean, input_std) - dict or array depending on format
        """
        # Check for NaN values
        if isinstance(self.input_mean, dict):
            # Dictionary format: check each token
            for key in self.input_mean.keys():
                assert not np.any(np.isnan(self.input_mean[key])), f"Input mean for '{key}' contains NaN values"
                assert not np.any(np.isnan(self.input_std[key])), f"Input std for '{key}' contains NaN values"
        else:
            # Legacy format
            assert not np.any(np.isnan(self.input_mean)), "Input mean contains NaN values"
            assert not np.any(np.isnan(self.input_std)), "Input std contains NaN values"
        
        # Return using legacy names for backward compatibility
        return self.state_mean, self.state_std

    def get_val_dataset(self):
        """
        Create and return a validation dataset using the validation trajectories.
        Returns None if no validation split was configured.
        
        The validation dataset shares the same normalization stats as the training set.
        """
        if not hasattr(self, 'val_trajectories') or len(self.val_trajectories) == 0:
            return None
        
        # Create a shallow copy of self to use as validation dataset
        val_dataset = copy.copy(self)
        val_dataset.trajectories = self.val_trajectories
        
        return val_dataset

    def __len__(self):
        return len(self.trajectories)

    def get_traj_len(self, traj):
        """
        Get trajectory length from trajectory dict.
        
        Args:
            traj: A trajectory dictionary from self.trajectories
            
        Returns:
            int: Length of the trajectory (number of timesteps)
        """
        if self._is_legacy_format:
            if isinstance(traj['observations'], dict):
                first_key = list(traj['observations'].keys())[0]
                return traj['observations'][first_key].shape[0]
            else:
                return traj['observations'].shape[0]
        else:
            # Flat format: use first input key
            first_key = self.input_token_names[0]
            return traj[first_key].shape[0]
    
    def get_inputs(self, traj):
        """
        Get input data from trajectory as dict.
        
        Args:
            traj: A trajectory dictionary from self.trajectories
            
        Returns:
            dict: Dictionary mapping input token names to their data arrays.
                  For legacy tensor format, returns {'observations': tensor}.
        """
        if self._is_legacy_format:
            if isinstance(traj['observations'], dict):
                return traj['observations']
            else:
                # Legacy tensor format - wrap in dict with single key
                return {'observations': traj['observations']}
        else:
            # Flat format: return dict of input keys
            return {name: traj[name] for name in self.input_token_names}
    
    def get_target(self, traj):
        """
        Get target data from trajectory.
        
        Args:
            traj: A trajectory dictionary from self.trajectories
            
        Returns:
            np.ndarray: Target data array of shape (T, target_dim)
        """
        if self._is_legacy_format:
            return traj['actions']
        else:
            return traj[self.target_key]
    
    # Keep private aliases for internal use
    _get_traj_len = get_traj_len
    _get_inputs = get_inputs
    _get_target = get_target

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        dtype = torch.float32
        
        traj_len = self.get_traj_len(traj)
        inputs = self.get_inputs(traj)
        target = self.get_target(traj)
        
        # Check if inputs is a dictionary
        is_dict_inputs = isinstance(inputs, dict) and len(inputs) > 0
        # Special case: legacy tensor format wrapped in dict
        is_single_tensor = is_dict_inputs and 'observations' in inputs and len(inputs) == 1 and not isinstance(list(inputs.values())[0], dict)

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            if is_dict_inputs and not is_single_tensor:
                # Dictionary format: slice each input token separately
                states = {}
                for key, value in inputs.items():
                    states[key] = torch.from_numpy(value[si : si + self.context_len]).to(dtype)
            else:
                # Legacy array format (possibly wrapped in dict)
                obs_data = inputs['observations'] if is_single_tensor else inputs
                states = torch.from_numpy(obs_data[si : si + self.context_len]).to(dtype)
            
            actions = torch.from_numpy(target[si : si + self.context_len]).to(dtype)
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            if is_dict_inputs and not is_single_tensor:
                # Dictionary format: pad each input token separately
                states = {}
                for key, value in inputs.items():
                    state_tensor = torch.from_numpy(value).to(dtype)
                    padded_state = torch.cat([state_tensor,
                                            torch.zeros(([padding_len] + list(state_tensor.shape[1:])),
                                            dtype=dtype)],
                                           dim=0)
                    states[key] = padded_state
            else:
                # Legacy array format (possibly wrapped in dict)
                obs_data = inputs['observations'] if is_single_tensor else inputs
                states = torch.from_numpy(obs_data).to(dtype)
                states = torch.cat([states,
                                    torch.zeros(([padding_len] + list(states.shape[1:])),
                                    dtype=dtype)],
                                   dim=0)

            actions = torch.from_numpy(target).to(dtype)
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=dtype)],
                               dim=0)

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        # B, T = 1, self.context_len
        return  timesteps, self._reshape_states(states), actions, traj_mask



    def _reshape_states(self, states):
        """
        Reshape the states to match the expected input shape.
        """
        return states
    

class ToyDataset(TrajectoryDataset):
    """
    Toy dataset demonstrating the legacy format with 'observations' dict and 'actions'.
    This is kept for backward compatibility.
    """

    def _setup_dataset(self, dataset_config):
        # Create trajectories - parent class will infer properties automatically
        self.trajectories = []
        num_trajectories = dataset_config.get('num_trajectories', 100)
        
        for _ in range(num_trajectories):
            traj_len = random.randint(10, 50)
            # Store states as a dictionary
            position = 1 + 0.1*np.random.randn(traj_len, 2)
            velocity = 2 + 0.1*np.random.randn(traj_len, 2)
            actions = np.repeat((position[:,0] + velocity[:,0]).reshape(-1,1), repeats=2, axis=1) 
            self.trajectories.append({
                'observations': {
                    'position': position,
                    'velocity': velocity
                },
                'actions': actions,
            })


class ToyDatasetFlat(TrajectoryDataset):
    """
    Toy dataset demonstrating the NEW flat format.
    
    In flat format, all data streams are at the same level in the trajectory dict.
    Use 'target_key' in dataset_config to specify which key is the prediction target.
    
    Example usage:
        data_cfg = {
            'target_key': 'combined',  # This is what we want to predict
            'num_trajectories': 100
        }
        dataset = ToyDatasetFlat(data_cfg, context_len=10)
    """

    def _setup_dataset(self, dataset_config):
        self.trajectories = []
        num_trajectories = dataset_config.get('num_trajectories', 100)
        
        for _ in range(num_trajectories):
            traj_len = random.randint(10, 50)
            
            # All data streams are at the same level (flat structure)
            position = 1 + 0.1*np.random.randn(traj_len, 2)
            velocity = 2 + 0.1*np.random.randn(traj_len, 2)
            
            # Target: combined value to predict
            combined = np.repeat((position[:,0] + velocity[:,0]).reshape(-1,1), repeats=2, axis=1)
            
            # Flat format: all keys at same level, target_key specifies prediction target
            self.trajectories.append({
                'position': position,     # input
                'velocity': velocity,     # input
                'combined': combined,     # target (specified via target_key in config)
            })


class ToyDatasetPositionEstimator(TrajectoryDataset):
    """
    Toy dataset for testing position estimator using the NEW flat format.
    
    Data streams:
    - velocity: 2D velocity vector (input)
    - noise: 2D random vector (input)  
    - position: 2D position (target - to be predicted)
    
    Positions are obtained by integrating velocity vectors.
    
    Example usage:
        data_cfg = {
            'target_key': 'position',  # Predict position from velocity
            'num_trajectories': 10000
        }
        dataset = ToyDatasetPositionEstimator(data_cfg, context_len=10)
    """

    def _setup_dataset(self, dataset_config):
        self.trajectories = []
        num_trajectories = dataset_config.get('num_trajectories', 10000)
        
        for _ in range(num_trajectories):
            traj_len = random.randint(10, 50)
            
            # velocity is a 2D velocity vector (with some noise)
            velocity = 0.5 + 0.2*np.random.randn(traj_len, 2)
            
            # noise is a 2D random vector
            noise = np.random.randn(traj_len, 2)
            
            # positions obtained by integrating velocity
            # Starting from (0, 0) for each trajectory
            positions = np.zeros((traj_len, 2))
            positions[0] = np.array([0.0, 0.0])
            for t in range(1, traj_len):
                # Integrate velocity to get position (simple Euler integration)
                positions[t] = positions[t-1] + velocity[t-1, :]
            
            # Flat format: target_key='position' should be set in dataset_config
            self.trajectories.append({
                'velocity': velocity,   # input
                'noise': noise,         # input
                'position': positions,  # target
            })


class ToyDatasetVelocityEstimator(TrajectoryDataset):
    """
    Toy dataset for testing velocity estimator using the NEW flat format.
    
    Data streams:
    - position: 2D position vector (input)
    - noise: 2D random vector (input)
    - velocity: 2D velocity (target - to be predicted)
    
    Example usage:
        data_cfg = {
            'target_key': 'velocity',  # Predict velocity from position
            'num_trajectories': 10000
        }
        dataset = ToyDatasetVelocityEstimator(data_cfg, context_len=10)
    """

    def _setup_dataset(self, dataset_config):
        self.trajectories = []
        num_trajectories = dataset_config.get('num_trajectories', 10000)
        
        for _ in range(num_trajectories):
            traj_len = random.randint(10, 50)
            
            # velocity is a 2D velocity vector (with some noise)
            velocity = 0.2 + 0.5*np.random.randn(traj_len, 2)
            
            # noise is a 2D random vector
            noise = np.random.randn(traj_len, 2)
            
            # positions obtained by integrating velocity
            # Starting from (0, 0) for each trajectory
            positions = np.zeros((traj_len, 2))
            positions[0] = np.array([0.0, 0.0])
            for t in range(1, traj_len):
                # Integrate velocity to get position (simple Euler integration)
                positions[t] = positions[t-1] + velocity[t-1, :]
            
            # Flat format: target_key='velocity' should be set in dataset_config
            self.trajectories.append({
                'position': positions,  # input
                'noise': noise,         # input
                'velocity': velocity,   # target
            })




class ModuleTrajectoryDataset(TrajectoryDataset):

    def _reshape_states(self, states):
        return states.reshape(self.context_len, self.max_num_modules, 8)


    def _setup_dataset(self, dataset_config):
        dataset_path = dataset_config['dataset_path']
        print("dataset path: " + str(dataset_path))


        self.state_token_dims = [8]*5 # 5 modules, each module has 8 state dimensions
        self.act_dim = 5

        max_state_dim = 40
        max_act_dim = 5

        self.max_num_modules = 5

        # load dataset
        assert type(dataset_path) == list
        # One embodiment per item in the list
        obs_dict = {}
        act_dict = {}
        timeouts_dict = {}
        # One embodiment per file
        for file in dataset_path:
            if file.endswith('.npz'):
                dataset_npz = np.load(file)
            else:
                raise NotImplementedError("Only npz files are supported")

            obs = dataset_npz["observations"]
            act = dataset_npz["actions"]
            timeouts=dataset_npz["dones"]

            # data_token = act.shape[1]
            if act.shape[1] == 1:
                data_token = (1)
            elif act.shape[1] == 3:
                # TODO: Other cases
                data_token = (1,1,1,0,0)
            elif act.shape[1] == 5:
                data_token = (1,1,1,1,1)
            elif act.shape[1] == 4:
                # 4D data should be paired with infomation of how the 4D data is going to be stiched together with 1D data
                if "amputatedA" in file:
                    data_token = (1,0,1,1,1)
                elif "amputatedB" in file:
                    data_token = (1,1,0,1,1)
                elif "amputatedC" in file:
                    data_token = (1,1,1,0,1)
                elif "amputatedD" in file:
                    data_token = (1,1,1,1,0)
                else:
                    print("Find general case 4-module data")
                    data_token = (1,1,1,1,0)

            if data_token in obs_dict:
                obs_dict[data_token] = np.concatenate((obs_dict[data_token], obs), axis=0)
                act_dict[data_token] = np.concatenate((act_dict[data_token], act), axis=0)
                timeouts_dict[data_token] = np.concatenate((timeouts_dict[data_token], timeouts), axis=0)
            else:
                obs_dict[data_token] = obs
                act_dict[data_token] = act
                timeouts_dict[data_token] = timeouts
            

        self.trajectories = []

        ############ Qudruped robot ############
        token = (1,1,1,1,1)
        if token in obs_dict:
            obs = obs_dict[token]
            act = act_dict[token]
            timeouts = timeouts_dict[token]

            # Each item in the list is a trajectory
            state_data = []
            action_data = []

            state_data_p = []
            action_data_p = []
            for i in trange(len(obs)):
                if timeouts[i]:
                    state_data.append(state_data_p)
                    action_data.append(action_data_p)
                    state_data_p = []
                    action_data_p = []
                # When the ith is done, the ith obs is the first obs of the next trajectory
                state_data_p.append(obs[i])
                action_data_p.append(act[i])

            for i in trange(len(state_data)):
                self.trajectories.append({
                    'observations': np.array(state_data[i]),
                    'actions': np.array(action_data[i]),
                })
        else:
            print("!!!!!!!!!!!!!   No quadruped robot trajectories found!")


        ############ Single Modules * 5 ############
        if (1) in obs_dict:
            print("Stitching together single module trajectories")
            obs1 = obs_dict[(1)]
            act1 = act_dict[(1)]
            timeouts1 = timeouts_dict[(1)]


            obs1s = [obs1]
            act1s = [act1]
            timeouts1s = [timeouts1]

            # Data augmentation
            true_indices = [i for i, value in enumerate(timeouts1) if value == True]
            n_new = 5-1
            interval = (len(true_indices) - 20) / (n_new - 1)
            split_indexs = [true_indices[int(10 + i * interval)] for i in range(n_new)]

            for split_index in split_indexs:
                print(f"Splitting at index {split_index}")
                split_index = random.choice(true_indices[:-1])
                timeout1_sub1 = timeouts1[1:split_index + 1]
                timeout1_sub2 = timeouts1[split_index + 1:]
                obs1_sub1 = obs1[:split_index + 1]
                act1_sub1 = act1[:split_index + 1]
                obs1_sub2 = obs1[split_index + 1:-1]
                act1_sub2 = act1[split_index + 1:-1]
                obs1_new = np.concatenate((obs1_sub2, obs1_sub1, obs1[-1:-1]), axis=0)
                act1_new = np.concatenate((act1_sub2, act1_sub1, act1[-1:-1]), axis=0)
                timeout1_new = np.concatenate(([False], timeout1_sub2, timeout1_sub1), axis=0)
                obs1s.append(obs1_new)
                act1s.append(act1_new)
                timeouts1s.append(timeout1_new)

            # Each item in the list is a trajectory
            state_data = []
            action_data = []

            state_data_p = []
            action_data_p = []
            for i in trange(min([len(obs1) for obs1 in obs1s])):
                if np.any([timeouts1s[j][i] for j in range(5)]):
                    state_data.append(state_data_p)
                    action_data.append(action_data_p)
                    state_data_p = []
                    action_data_p = []
                # When the ith is done, the ith obs is the first obs of the next trajectory
                state_data_p.append(np.concatenate([obs1s[j][i] for j in range(5)], axis=0))
                action_data_p.append(np.concatenate([act1s[j][i] for j in range(5)], axis=0))

            for i in trange(len(state_data)):
                self.trajectories.append({
                    'observations': np.array(state_data[i]),
                    'actions': np.array(action_data[i]),
                })



        ############ AmputatedA/B/C + Single Modules * 1 ############

        # if (1) in obs_dict:
        #     obs1 = obs_dict[(1)]
        #     act1 = act_dict[(1)]
        #     timeouts1 = timeouts_dict[(1)]

        tokens = [(1,0,1,1,1), (1,1,0,1,1), (1,1,1,0,1), (1,1,1,1,0)]
        # (1,1,1,0,1)
        for token in tokens:
            print(f"Stitching {token} and (1)")
            if token in obs_dict and (1) in obs_dict:
                print("Stitching together amputatedC and single module trajectories")
                obs4 = obs_dict[token]
                act4 = act_dict[token]
                timeouts4 = timeouts_dict[token]

                while len(obs1) < len(obs4):
                    print("Data augmentation")
                    obs1 = np.concatenate((obs1, obs1), axis=0)
                    act1 = np.concatenate((act1, act1), axis=0)
                    timeouts1 = np.concatenate((timeouts1, timeouts1), axis=0)

                state_data = []
                action_data = []

                state_data_p = []
                action_data_p = []
                for i in trange(min(len(obs1), len(obs4))):
                    if timeouts1[i] or timeouts4[i]:
                        state_data.append(state_data_p)
                        action_data.append(action_data_p)
                        state_data_p = []
                        action_data_p = []
                    # When the ith is done, the ith obs is the first obs of the next trajectory
                    module_state_dim = obs1.shape[1]

                    cut = token.index(0)
                    sticthed_obs = np.concatenate((obs4[i][:module_state_dim*cut], obs1[i], obs4[i][-module_state_dim*(4-cut):]))
                    assert sticthed_obs.shape[0] == max_state_dim, f"sticthed_obs.shape[0] = {sticthed_obs.shape[0]}"
                    assert len(sticthed_obs.shape) == 1, f"len(sticthed_obs.shape) = {len(sticthed_obs.shape)}"
                    state_data_p.append(sticthed_obs)
                    sticthed_act = np.concatenate((act4[i][:cut], act1[i], act4[i][-(4-cut):]))
                    assert sticthed_act.shape[0] == max_act_dim, f"sticthed_act.shape[0] = {sticthed_act.shape[0]}"
                    assert len(sticthed_act.shape) == 1, f"len(sticthed_act.shape) = {len(sticthed_act.shape)}"
                    action_data_p.append(sticthed_act)
                    # state_data_p.append(np.concatenate((obs4[i], obs1[i]), axis=0))
                    # action_data_p.append(np.concatenate((act4[i], act1[i]), axis=0))

                for i in trange(len(state_data)):
                    self.trajectories.append({
                        'observations': np.array(state_data[i]),
                        'actions': np.array(action_data[i]),
                    })


        ############ Tripod + Single Modules * 2 ############
        token = (1,1,1,0,0)
        if token in obs_dict and (1) in obs_dict:
            print("Stitching together single module and tripod robot trajectories")
            # Tripod robot + single modules
            obs3 = obs_dict[token]
            act3 = act_dict[token]
            timeouts3 = timeouts_dict[token]

            while len(obs1) < len(obs3):
                print("Data augmentation")
                obs1 = np.concatenate((obs1, obs1), axis=0)
                act1 = np.concatenate((act1, act1), axis=0)
                timeouts1 = np.concatenate((timeouts1, timeouts1), axis=0)

            # obs1 = obs_dict[1]
            # act1 = act_dict[1]
            # timeouts1 = timeouts_dict[1]

            # Data augmentation
            true_indices = [i for i, value in enumerate(timeouts1) if value == True]
            split_index = random.choice(true_indices[:-1])
            timeout1_sub1 = timeouts1[1:split_index + 1]
            timeout1_sub2 = timeouts1[split_index + 1:]
            obs1_sub1 = obs1[:split_index + 1]
            act1_sub1 = act1[:split_index + 1]
            obs1_sub2 = obs1[split_index + 1:-1]
            act1_sub2 = act1[split_index + 1:-1]
            obs1b = np.concatenate((obs1_sub2, obs1_sub1, obs1[-1:-1]), axis=0)
            act1b = np.concatenate((act1_sub2, act1_sub1, act1[-1:-1]), axis=0)
            timeout1b = np.concatenate(([False], timeout1_sub2, timeout1_sub1), axis=0)

            # Each item in the list is a trajectory
            state_data = []
            action_data = []

            state_data_p = []
            action_data_p = []
            for i in trange(min(len(obs1), len(obs3), len(obs1b))):
                if timeouts1[i] or timeouts3[i] or timeout1b[i]:
                    state_data.append(state_data_p)
                    action_data.append(action_data_p)
                    state_data_p = []
                    action_data_p = []
                # When the ith is done, the ith obs is the first obs of the next trajectory
                state_data_p.append(np.concatenate((obs3[i], obs1[i], obs1b[i]), axis=0))
                action_data_p.append(np.concatenate((act3[i], act1[i], act1b[i]), axis=0))

            for i in trange(len(state_data)):
                self.trajectories.append({
                    'observations': np.array(state_data[i]),
                    'actions': np.array(action_data[i]),
                })


        print(f"Loaded {len(self.trajectories)} trajectories")
            
        
        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std





class QuickDistill(TrajectoryDataset):

    '''Can only work with old policy trainer twist_controller.
    (This is for a better compatibility. For example, with magnetic docking.)
    This is a quick distill dataset for testing purposes.
    TODO: move this class to twist_controller
    '''

    


    def _reshape_states(self, states):
        return states.reshape(self.context_len, self.max_num_modules, 8)



    def _record(self, load_runs):


        from capybararl.common.env_util import make_vec_env
        from capybararl.common.vec_env import DummyVecEnv, VecEnv
        from capybararl import CrossQ


        from collections import defaultdict
        import gymnasium as gym
        from twist_controller.envs.env_sim import ZeroSim
        import wandb

        from tqdm import trange, tqdm

        from twist_controller.utils.files import get_cfg_path, load_cfg
        from twist_controller.scripts.train_sbx import load_model

        self.max_num_modules = None


        for load_run in load_runs:

            load_run_dir = os.path.dirname(load_run)

            file_name = load_run_dir.split("/")[-1] + "_rollout.npz"

            conf_name = os.path.join(load_run_dir, "running_config.yaml")
            if not os.path.exists(conf_name):
                raise FileNotFoundError(f"Configuration file {conf_name} not found. Please check the path.")
            else:
                print(f"Loading configuration from {conf_name}")


            conf = load_cfg(conf_name, alg="sbx")


            assert conf.trainer.mode == "train", "QuickDistill can only work in train mode with old policy trainer twist_controller."
            record_obs_type = "sensed_proprioception_lite"
            record_num_envs = 10
            record_steps = 1000000
            normalize_default_pos = True
            save_dir = self.log_dir

            if self.max_num_modules is None:
                self.max_num_modules = conf.agent.num_act
            else:
                assert self.max_num_modules == conf.agent.num_act, f"self.max_num_modules = {self.max_num_modules}, conf.agent.num_act = {conf.agent.num_act}. Please check the configuration."
            self.act_dim = self.max_num_modules


            num_envs = record_num_envs
            batch_steps = int(record_steps // num_envs)
            def make_env():
                return gym.wrappers.TimeLimit(
                    ZeroSim(conf), max_episode_steps=1000
                )
            model = load_model(load_run, make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv), CrossQ)
            vec_env = model.get_env()
            obs = vec_env.reset()
            constructed_obs = [vec_env.envs[i].unwrapped.brain._construct_obs(record_obs_type) for i in range(len(vec_env.envs))] if record_obs_type is not None else obs

            rollout = defaultdict(list)

            t0 = time.time()
            from rich.progress import Progress
            progress = Progress()
            progress.start()
            task = progress.add_task("[red]Recording...", total=batch_steps)
            while True:
                action, _states = model.predict(obs, deterministic=True)
                rollout["observations"].append(constructed_obs)
                # np.shape(rollout["observations"])
                act_recorded = action if not normalize_default_pos else action+np.array(conf.agent.default_dof_pos)
                rollout["actions"].append(act_recorded)
                obs, reward, done, info = vec_env.step(action)
                active_modules =  [ifo["active_modules"] for ifo in info]
                rollout["active_modules"].append(active_modules)
                constructed_obs = [vec_env.envs[i].unwrapped.brain._construct_obs(record_obs_type) for i in range(len(vec_env.envs))] if record_obs_type is not None else obs
                # print(done)
                rollout["rewards"].append(reward)
                rollout["dones"].append(done)
                # if done[0]:
                    # step_count = 0
                    # obs = vec_env.reset()
                print(f"{len(rollout['observations'])} / {batch_steps}")
                progress.update(task, advance=1)

                if time.time() - t0 > 60*30:
                    print(f"Recording trajectories... ({len(rollout['observations'])} steps)")
                    # np.savez_compressed(os.path.join(save_dir, f"rollout.npz"), **rollout)
                    save_rollout(rollout, save_dir, file_name)
                    t0 = time.time()
                if len(rollout['observations']) >= batch_steps:
                    print(f"Finished recording trajectories ({len(rollout['observations'])} steps). Saving to {save_dir}...")
                    save_rollout(rollout, save_dir, file_name)
                    print("Done!")
                    break

            progress.stop()


    def _setup_dataset(self, dataset_config):
        self.obs_dim_per_module = 8


        self.log_dir = dataset_config['log_dir']
        if not ("use_existing_rollouts" in dataset_config and dataset_config["use_existing_rollouts"]):
            load_runs = dataset_config['load_runs']
            # Legacy config only TODO: converter from old config to new config
            print("Record rollout from the following runs:")
            for load_run in load_runs:
                print(f" - {load_run}")

            self._record(load_runs)

        rollout_file_names = glob.glob(os.path.join(self.log_dir, "*.npz"))
        print(f"Found {len(rollout_file_names)} rollout files in {self.log_dir}")

        # self.state_token_dims = [self.obs_dim_per_module]*self.max_num_modules # 5 modules, each module has 8 state dimensions
        

        # max_state_dim = self.obs_dim_per_module*self.max_num_modules
        # max_act_dim = self.act_dim


        # load dataset
        obs_dict = {}
        act_dict = {}
        timeouts_dict = {}
        # One embodiment per file
        for file in rollout_file_names:
            if file.endswith('.npz'):
                dataset_npz = np.load(file)
            else:
                raise NotImplementedError("Only npz files are supported")

            obs = dataset_npz["observations"]
            act = dataset_npz["actions"]
            timeouts=dataset_npz["dones"]

            data_token = "therobot"

            if data_token in obs_dict:
                obs_dict[data_token] = np.concatenate((obs_dict[data_token], obs), axis=0)
                act_dict[data_token] = np.concatenate((act_dict[data_token], act), axis=0)
                timeouts_dict[data_token] = np.concatenate((timeouts_dict[data_token], timeouts), axis=0)
            else:
                obs_dict[data_token] = obs
                act_dict[data_token] = act
                timeouts_dict[data_token] = timeouts
            

        self.trajectories = []

        token = "therobot"
        if token in obs_dict:
            obs = obs_dict[token]
            act = act_dict[token]
            timeouts = timeouts_dict[token]

            # Each item in the list is a trajectory
            state_data = []
            action_data = []

            state_data_p = []
            action_data_p = []
            for i in trange(len(obs)):
                if timeouts[i]:
                    state_data.append(state_data_p)
                    action_data.append(action_data_p)
                    state_data_p = []
                    action_data_p = []
                # When the ith is done, the ith obs is the first obs of the next trajectory
                state_data_p.append(obs[i])
                action_data_p.append(act[i])

            for i in trange(len(state_data)):
                self.trajectories.append({
                    'observations': np.array(state_data[i]),
                    'actions': np.array(action_data[i]),
                })
        else:
            print("Something wrong!")


        self.max_num_modules = np.array(state_data[i]).shape[-1] // self.obs_dim_per_module
        self.state_token_dims = [self.obs_dim_per_module]*self.max_num_modules
        self.act_dim = self.max_num_modules
        print(f"Loaded {len(self.trajectories)} trajectories")
            
        
        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        print(f"State mean: {self.state_mean}, State std: {self.state_std}")
        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std





if __name__ == "__main__":
    from capyformer.trainer import Trainer

    # Example using NEW flat format with ToyDatasetVelocityEstimator
    # The 'target_key' specifies which data stream to predict
    context_len = 10
    data_cfg = {
        "target_key": "velocity",  # This is what we want to predict
        "num_trajectories": 1000,
    }
    traj_dataset = ToyDatasetVelocityEstimator(data_cfg, context_len)
    
    print("\n=== Dataset Info ===")
    print(f"Input token names: {traj_dataset.input_token_names}")
    print(f"Input token dims: {traj_dataset.input_token_dims}")
    print(f"Target key: {traj_dataset.target_key}")
    print(f"Target dim: {traj_dataset.target_dim}")
    print(f"Number of trajectories: {len(traj_dataset)}")
    
    dt = Trainer(
        traj_dataset,
        log_dir="./debug",
        use_action_tanh=False,
        shared_state_embedding=False,
        n_blocks=3,
        h_dim=256,
        n_heads=1,
        batch_size=32,
        action_is_velocity=True
    )
    dt.learn(
        n_epochs=10000,
    )


    # # Example using ToyDatasetPositionEstimator with flat format
    # context_len = 10
    # data_cfg = {
    #     "target_key": "position",  # Predict position from velocity
    #     "num_trajectories": 1000,
    # }
    # traj_dataset = ToyDatasetPositionEstimator(data_cfg, context_len)
    # 
    # dt = Trainer(
    #     traj_dataset,
    #     log_dir="./debug",
    #     use_action_tanh=False,
    #     shared_state_embedding=False,
    #     n_blocks=3,
    #     h_dim=256,
    #     n_heads=1,
    #     batch_size=32,
    #     action_is_velocity=False
    # )
    # dt.learn(
    #     n_epochs=10000,
    # )
    #     action_is_velocity=False
    # )
    # dt.learn(
    #     n_epochs=10000,
    # )
    