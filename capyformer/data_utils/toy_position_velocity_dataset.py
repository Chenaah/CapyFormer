"""
Toy dataset for testing position-to-velocity estimation.
Similar to RollbotPositionEstimator but with synthetic data.

The dataset generates fake position trajectories and calculates velocities from them.
- Observations: position (3D coordinates)
- Actions: velocity (calculated from position differences)
"""

import numpy as np
import random
from capyformer.data import TrajectoryDataset
from capyformer.trainer import Trainer


class ToyPositionVelocityDataset(TrajectoryDataset):
    """
    Toy dataset that generates synthetic position trajectories.
    
    Observations: 3D positions
    Actions: 3D velocities (calculated from position differences)
    """

    def _setup_dataset(self, dataset_config):
        """
        Generate synthetic trajectories with positions as observations
        and velocities as actions.
        """
        num_trajectories = dataset_config.get('num_trajectories', 1000) if dataset_config else 1000
        min_traj_len = dataset_config.get('min_traj_len', 50) if dataset_config else 50
        max_traj_len = dataset_config.get('max_traj_len', 200) if dataset_config else 200
        dt = dataset_config.get('dt', 0.02) if dataset_config else 0.02  # 50Hz control frequency
        
        print(f"Generating {num_trajectories} synthetic trajectories...")
        
        self.trajectories = []
        
        for i in range(num_trajectories):
            traj_len = random.randint(min_traj_len, max_traj_len)
            
            # Generate synthetic position trajectories
            # Using smooth random walks with some structure
            positions = self._generate_smooth_trajectory(traj_len)
            
            # Calculate velocities from position differences
            velocities = np.zeros_like(positions)
            velocities[0] = np.array([0.0, 0.0, 0.0])  # First velocity is zero
            for t in range(1, traj_len):
                velocities[t] = (positions[t] - positions[t-1]) / dt
            
            self.trajectories.append({
                'observations': {
                    'position': positions,  # 3D position
                },
                'actions': velocities,  # 3D velocity
            })
        
        print(f"Generated {len(self.trajectories)} trajectories")
    
    def _generate_smooth_trajectory(self, traj_len):
        """
        Generate a smooth 3D trajectory using a random walk with momentum.
        
        Args:
            traj_len: Length of the trajectory
            
        Returns:
            positions: (traj_len, 3) array of 3D positions
        """
        positions = np.zeros((traj_len, 3))
        
        # Start at origin with some random offset
        positions[0] = np.random.randn(3) * 0.1
        
        # Generate velocity with momentum (smooth changes)
        velocity = np.random.randn(3) * 0.5
        
        for t in range(1, traj_len):
            # Add random acceleration (smooth velocity changes)
            acceleration = np.random.randn(3) * 0.1
            velocity += acceleration
            
            # Add some damping to prevent runaway velocities
            velocity *= 0.98
            
            # Update position
            positions[t] = positions[t-1] + velocity * 0.02  # dt = 0.02
        
        return positions


class ToyPositionVelocityDataset2D(TrajectoryDataset):
    """
    Simplified 2D version of the toy dataset.
    
    Observations: 2D positions
    Actions: 2D velocities (calculated from position differences)
    """

    def _setup_dataset(self, dataset_config):
        """
        Generate synthetic 2D trajectories with positions as observations
        and velocities as actions.
        """
        num_trajectories = dataset_config.get('num_trajectories', 1000) if dataset_config else 1000
        min_traj_len = dataset_config.get('min_traj_len', 50) if dataset_config else 50
        max_traj_len = dataset_config.get('max_traj_len', 200) if dataset_config else 200
        dt = dataset_config.get('dt', 0.02) if dataset_config else 0.02  # 50Hz control frequency
        
        print(f"Generating {num_trajectories} synthetic 2D trajectories...")
        
        self.trajectories = []
        
        for i in range(num_trajectories):
            traj_len = random.randint(min_traj_len, max_traj_len)
            
            # Generate synthetic 2D position trajectories
            # Create circular or spiral patterns
            positions = self._generate_2d_trajectory(traj_len)
            
            # Calculate velocities from position differences
            velocities = np.zeros_like(positions)
            velocities[0] = np.array([0.0, 0.0])  # First velocity is zero
            for t in range(1, traj_len):
                velocities[t] = (positions[t] - positions[t-1]) / dt
            
            self.trajectories.append({
                'observations': {
                    'position': positions,  # 2D position
                },
                'actions': velocities,  # 2D velocity
            })
        
        print(f"Generated {len(self.trajectories)} trajectories")
    
    def _generate_2d_trajectory(self, traj_len):
        """
        Generate a smooth 2D trajectory (circles, spirals, or random walks).
        
        Args:
            traj_len: Length of the trajectory
            
        Returns:
            positions: (traj_len, 2) array of 2D positions
        """
        positions = np.zeros((traj_len, 2))
        
        # Randomly choose trajectory type
        traj_type = random.choice(['circle', 'spiral', 'random_walk'])
        
        if traj_type == 'circle':
            # Circular trajectory
            radius = random.uniform(1.0, 5.0)
            angular_velocity = random.uniform(0.1, 0.5)
            center = np.random.randn(2) * 2.0
            
            for t in range(traj_len):
                angle = angular_velocity * t * 0.02  # dt = 0.02
                positions[t] = center + radius * np.array([np.cos(angle), np.sin(angle)])
        
        elif traj_type == 'spiral':
            # Spiral trajectory
            angular_velocity = random.uniform(0.1, 0.5)
            radius_growth = random.uniform(0.01, 0.05)
            center = np.random.randn(2) * 2.0
            
            for t in range(traj_len):
                angle = angular_velocity * t * 0.02
                radius = 1.0 + radius_growth * t
                positions[t] = center + radius * np.array([np.cos(angle), np.sin(angle)])
        
        else:  # random_walk
            # Random walk with momentum
            positions[0] = np.random.randn(2) * 0.1
            velocity = np.random.randn(2) * 0.5
            
            for t in range(1, traj_len):
                acceleration = np.random.randn(2) * 0.1
                velocity += acceleration
                velocity *= 0.98  # Damping
                positions[t] = positions[t-1] + velocity * 0.02
        
        return positions


if __name__ == "__main__":
    # Example 1: 3D Position-Velocity Dataset
    # print("=" * 60)
    # print("Testing 3D Position-Velocity Dataset")
    # print("=" * 60)
    
    context_len = 100
    dataset_config = {
        'num_trajectories': 500,
        'min_traj_len': 50,
        'max_traj_len': 150,
        'dt': 0.02,
    }
    
    # traj_dataset_3d = ToyPositionVelocityDataset(dataset_config, context_len)
    
    # print(f"\nDataset properties:")
    # print(f"  State token names: {traj_dataset_3d.state_token_names}")
    # print(f"  State token dims: {traj_dataset_3d.state_token_dims}")
    # print(f"  Action dim: {traj_dataset_3d.act_dim}")
    # print(f"  Number of trajectories: {len(traj_dataset_3d.trajectories)}")
    
    # # Train a model on the 3D dataset
    # dt_3d = Trainer(
    #     traj_dataset_3d,
    #     log_dir="./debug/toy_3d",
    #     use_action_tanh=False,
    #     shared_state_embedding=False,
    #     n_blocks=3,
    #     h_dim=128,
    #     n_heads=2,
    #     batch_size=32,
    #     validation_freq=50,
    #     action_is_velocity=True
    # )
    # dt_3d.learn(n_epochs=1000)
    
    # print("\n" + "=" * 60)
    # print("Testing 2D Position-Velocity Dataset")
    # print("=" * 60)
    
    # Example 2: 2D Position-Velocity Dataset
    traj_dataset_2d = ToyPositionVelocityDataset2D(dataset_config, context_len)
    
    print(f"\nDataset properties:")
    print(f"  State token names: {traj_dataset_2d.state_token_names}")
    print(f"  State token dims: {traj_dataset_2d.state_token_dims}")
    print(f"  Action dim: {traj_dataset_2d.act_dim}")
    print(f"  Number of trajectories: {len(traj_dataset_2d.trajectories)}")
    
    # Train a model on the 2D dataset
    dt_2d = Trainer(
        traj_dataset_2d,
        log_dir="./debug",
        use_action_tanh=False,
        shared_state_embedding=False,
        n_blocks=3,
        h_dim=128,
        n_heads=1,
        batch_size=256,
        validation_freq=50,
        action_is_velocity=True
    )
    dt_2d.learn(n_epochs=10000)
