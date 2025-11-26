"""
Script to load trajectory data from CSV files in the cleaned folder.

Each trajectory is represented as a dictionary with the following keys:
- timestamp: array of timestamps (column 0)
- position: array of positions (columns 1, 2, 3) - x, y, z coordinates
- optitrack_quaternion: array of global quaternions (columns 4, 5, 6, 7) - x, y, z, w
- imu_quaternion: array of local quaternions (columns 8, 9, 10, 11) - x, y, z, w
- acceleration: array of accelerations (columns 12, 13, 14) - ax, ay, az
- angular_velocity: array of angular velocities (columns 15, 16, 17) - wx, wy, wz
- motor_angle: array of motor angles (column 18)
- motor_position: array of motor positions (column 19)
- filename: name of the source CSV file
"""

import os
import csv
import numpy as np
from pathlib import Path


def load_trajectories(cleaned_folder="examples/cleaned", verbose=True, max_length=None):
    """
    Load CSV files from the cleaned folder and organize them into a trajectory list.
    
    Args:
        cleaned_folder: Path to the folder containing cleaned CSV files
        verbose: Whether to print loading progress (default: True)
        max_length: Maximum length of each trajectory. If a trajectory exceeds this length,
                   it will be split into multiple trajectories (default: None, no splitting)
        
    Returns:
        List of dictionaries, each representing a trajectory with the following keys:
        - timestamp: array of timestamps (column 0)
        - position: array of positions (columns 1, 2, 3)
        - optitrack_quaternion: array of global quaternions (columns 4, 5, 6, 7) - format: x, y, z, w
        - imu_quaternion: array of local quaternions (columns 8, 9, 10, 11) - format: x, y, z, w
        - acceleration: array of accelerations (columns 12, 13, 14)
        - angular_velocity: array of angular velocities (columns 15, 16, 17)
        - motor_angle: array of motor angles (column 18)
        - motor_position: array of motor positions (column 19)
        - filename: name of the source CSV file (with _part_X suffix if split)
    """
    trajectory_list = []
    
    # Get all CSV files in the cleaned folder
    cleaned_path = Path(cleaned_folder)
    csv_files = sorted(cleaned_path.glob("*.csv"))
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files in {cleaned_folder}")
    
    for csv_file in csv_files:
        if verbose:
            print(f"Loading {csv_file.name}...")
        
        # Read the CSV file
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            data = [row for row in reader]
        
        # Convert to numpy array for easier slicing
        data_array = np.array(data, dtype=float)
        
        # Split trajectory if max_length is specified
        if max_length is not None and len(data_array) > max_length:
            num_parts = int(np.ceil(len(data_array) / max_length))
            if verbose:
                print(f"  Splitting into {num_parts} parts (length: {len(data_array)} -> max {max_length})")
            
            for part_idx in range(num_parts):
                start_idx = part_idx * max_length
                end_idx = min((part_idx + 1) * max_length, len(data_array))
                
                # Create filename with part suffix
                base_name = csv_file.stem  # filename without extension
                part_filename = f"{base_name}_part_{part_idx}.csv"
                
                # Create trajectory dictionary for this part
                trajectory = {
                    'timestamp': data_array[start_idx:end_idx, 0],
                    'position': data_array[start_idx:end_idx, 1:4],
                    'optitrack_quaternion': data_array[start_idx:end_idx, 4:8],
                    'imu_quaternion': data_array[start_idx:end_idx, 8:12],
                    'acceleration': data_array[start_idx:end_idx, 12:15],
                    'angular_velocity': data_array[start_idx:end_idx, 15:18],
                    'motor_angle': data_array[start_idx:end_idx, 18],
                    'motor_position': data_array[start_idx:end_idx, 19],
                    'filename': part_filename
                }
                
                trajectory_list.append(trajectory)
        else:
            # Create trajectory dictionary without splitting
            trajectory = {
                'timestamp': data_array[:, 0],
                'position': data_array[:, 1:4],
                'optitrack_quaternion': data_array[:, 4:8],
                'imu_quaternion': data_array[:, 8:12],
                'acceleration': data_array[:, 12:15],
                'angular_velocity': data_array[:, 15:18],
                'motor_angle': data_array[:, 18],
                'motor_position': data_array[:, 19],
                'filename': csv_file.name
            }
            
            trajectory_list.append(trajectory)
    
    if verbose:
        print(f"\nLoaded {len(trajectory_list)} trajectories")
    return trajectory_list


if __name__ == "__main__":
    # Load trajectories
    trajectories = load_trajectories()
    
    # Print summary information
    print("\n" + "="*60)
    print("Trajectory Summary")
    print("="*60)
    for i, traj in enumerate(trajectories[:5]):  # Show first 5
        print(f"\nTrajectory {i}: {traj['filename']}")
        print(f"  - Number of timesteps: {len(traj['timestamp'])}")
        print(f"  - Time range: {traj['timestamp'][0]:.0f} - {traj['timestamp'][-1]:.0f}")
        print(f"  - Position shape: {traj['position'].shape}")
        print(f"  - Global quaternion shape: {traj['optitrack_quaternion'].shape}")
        print(f"  - Local quaternion shape: {traj['imu_quaternion'].shape}")
        print(f"  - Acceleration shape: {traj['acceleration'].shape}")
        print(f"  - Angular velocity shape: {traj['angular_velocity'].shape}")
        print(f"  - Motor angle shape: {traj['motor_angle'].shape}")
        print(f"  - Motor position shape: {traj['motor_position'].shape}")
    
    if len(trajectories) > 5:
        print(f"\n... and {len(trajectories) - 5} more trajectories")
