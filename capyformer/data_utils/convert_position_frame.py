"""
Convert position information from Optitrack's global frame to IMU's global frame.

For each trajectory, we calculate the yaw offset between Optitrack and IMU quaternions,
then use this offset to rotate the x/y positions while keeping z unchanged.
"""

import numpy as np
from capyformer.data_utils.load_trajectories import load_trajectories
import argparse
import matplotlib.pyplot as plt


def quaternion_to_yaw(q):
    """
    Extract yaw angle (rotation around z-axis) from quaternion.
    Quaternion format: [x, y, z, w]
    
    Returns yaw angle in radians.
    """
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # Yaw (rotation around z-axis)
    # Formula: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return yaw


def rotate_xy_position(positions, yaw_offset):
    """
    Rotate x/y positions by a yaw angle offset, keeping z unchanged.
    
    Args:
        positions: (N, 3) array of [x, y, z] positions
        yaw_offset: scalar yaw angle in radians
        
    Returns:
        rotated_positions: (N, 3) array of rotated [x, y, z] positions
    """
    rotated_positions = positions.copy()
    
    # Rotation matrix for yaw (rotation around z-axis)
    cos_yaw = np.cos(yaw_offset)
    sin_yaw = np.sin(yaw_offset)
    
    # Apply rotation to x/y coordinates
    x_orig = positions[:, 0]
    y_orig = positions[:, 1]
    
    rotated_positions[:, 0] = cos_yaw * x_orig - sin_yaw * y_orig
    rotated_positions[:, 1] = sin_yaw * x_orig + cos_yaw * y_orig
    # z remains unchanged
    
    return rotated_positions


def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions.
    Quaternion format: [x, y, z, w]
    """
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.stack([x, y, z, w], axis=-1)


def quaternion_conjugate(q):
    """
    Compute the conjugate of a quaternion.
    Quaternion format: [x, y, z, w]
    """
    result = q.copy()
    result[..., :3] = -result[..., :3]  # Negate x, y, z components
    return result


def align_quaternions(q_seq):
    """
    Flip signs of quaternions to ensure they are in the same hemisphere as the first one.
    This avoids issues where q and -q represent the same rotation but mess up mean/std.
    """
    if len(q_seq) == 0:
        return q_seq
    
    aligned = q_seq.copy()
    reference = aligned[0]
    
    # Compute dot products with reference
    # q_seq is (N, 4)
    dots = np.sum(aligned * reference, axis=1)
    
    # Flip signs where dot product is negative
    mask = dots < 0
    aligned[mask] = -aligned[mask]
    
    return aligned


def offset_trajectory_to_origin(positions, offset_z=False):
    """
    Offset trajectory positions so that the first timestep starts at origin.
    
    Args:
        positions: (N, 3) array of [x, y, z] positions
        offset_z: If True, also offset z to 0; if False, keep original z values
        
    Returns:
        offset_positions: (N, 3) array with first timestep at origin
        initial_offset: (3,) array of the offset that was applied
    """
    offset_positions = positions.copy()
    
    # Get the initial position
    initial_position = positions[0].copy()
    
    # Offset x and y to (0, 0)
    offset_positions[:, 0] -= initial_position[0]
    offset_positions[:, 1] -= initial_position[1]
    
    # Optionally offset z
    if offset_z:
        offset_positions[:, 2] -= initial_position[2]
        initial_offset = initial_position
    else:
        initial_offset = np.array([initial_position[0], initial_position[1], 0.0])
    
    return offset_positions, initial_offset


def convert_positions_to_imu_frame(trajectories, use_median=True, verbose=True):
    """
    Convert Optitrack positions to IMU global frame for each trajectory.
    
    Uses quaternion difference to compute the yaw offset, which is more stable
    than extracting yaw from individual quaternions.
    
    Args:
        trajectories: List of trajectory dictionaries
        use_median: If True, use median of yaw offsets; if False, use mean
        verbose: Whether to print conversion details
        
    Returns:
        List of trajectory dictionaries with added 'position_imu_frame' key
    """
    converted_trajectories = []
    
    if verbose:
        print(f"{'Trajectory':<30} | {'Yaw Offset':<12} | {'Std Dev':<10} | {'Method':<8}")
        print("-" * 70)
    
    for traj in trajectories:
        optitrack_quat = traj['optitrack_quaternion']
        imu_quat = traj['imu_quaternion']
        position = traj['position']
        
        # Compute quaternion difference: q_diff = optitrack * conj(imu)
        # This gives us the rotation from IMU frame to Optitrack frame
        q_diff = quaternion_multiply(optitrack_quat, quaternion_conjugate(imu_quat))
        
        # Align quaternions to avoid sign flips
        q_diff = align_quaternions(q_diff)
        
        # Extract yaw from each difference quaternion
        yaw_offsets = quaternion_to_yaw(q_diff)
        
        # Calculate median or mean yaw offset for this trajectory
        if use_median:
            yaw_offset = np.median(yaw_offsets)
            method = "Median"
        else:
            yaw_offset = np.mean(yaw_offsets)
            method = "Mean"
        
        # Calculate standard deviation
        std_dev = np.std(yaw_offsets)
        
        # Convert positions to IMU frame (negative because we want IMU <- Optitrack)
        position_imu_frame = rotate_xy_position(position, -yaw_offset)
        
        # Create new trajectory dict with converted positions
        converted_traj = traj.copy()
        converted_traj['position_imu_frame'] = position_imu_frame
        converted_traj['yaw_offset'] = yaw_offset
        converted_traj['yaw_offset_std'] = std_dev
        converted_trajectories.append(converted_traj)
        
        if verbose:
            print(f"{traj['filename']:<30} | {np.rad2deg(yaw_offset):9.2f}° | {np.rad2deg(std_dev):7.2f}° | {method}")
    
    if verbose:
        print("-" * 70)
        print(f"Converted {len(converted_trajectories)} trajectories")
    
    return converted_trajectories


def calculate_velocity_from_position(trajectories, position_key='position_imu_frame', dt=0.02, verbose=True):
    """
    Calculate velocity from position data using finite differences.
    
    Velocity is computed as: v[i] = (pos[i+1] - pos[i-1]) / (2 * dt)
    For endpoints, uses forward/backward differences.
    
    Args:
        trajectories: List of trajectory dictionaries
        position_key: Key for position data to use ('position' or 'position_imu_frame')
        dt: Time step in seconds (default: 0.02 for 50Hz control frequency)
        verbose: Whether to print calculation details
        
    Returns:
        List of trajectory dictionaries with added 'velocity_calculated' key
    """
    trajectories_with_velocity = []
    
    if verbose:
        print(f"\nCalculating velocity from '{position_key}' (dt={dt}s, {1/dt:.0f}Hz)...")
        print(f"{'Trajectory':<30} | {'Mean Speed':<12} | {'Max Speed':<12} | {'Min Speed':<12}")
        print("-" * 80)
    
    for traj in trajectories:
        traj_with_vel = traj.copy()
        
        # Check if position key exists
        if position_key not in traj:
            if verbose:
                print(f"Warning: '{position_key}' not found in {traj['filename']}, skipping...")
            trajectories_with_velocity.append(traj_with_vel)
            continue
        
        positions = traj[position_key]
        
        # Calculate velocity using finite differences
        velocities = np.zeros_like(positions)
        
        # Central differences for interior points
        for i in range(1, len(positions)):
            velocities[i] = (positions[i] - positions[i-1]) / (dt)
        
        # Calculate speed (magnitude of velocity)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Add to trajectory
        traj_with_vel['velocity'] = velocities
        traj_with_vel['speed'] = speeds
        trajectories_with_velocity.append(traj_with_vel)
        
        if verbose:
            mean_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            min_speed = np.min(speeds)
            print(f"{traj['filename']:<30} | {mean_speed:9.4f} m/s | {max_speed:9.4f} m/s | {min_speed:9.4f} m/s")
    
    if verbose:
        print("-" * 80)
        print(f"Calculated velocity for {len(trajectories_with_velocity)} trajectories")
    
    return trajectories_with_velocity


def normalize_trajectory_positions(trajectories, offset_to_origin=True, offset_z=False, verbose=True):
    """
    Normalize trajectory positions by offsetting to origin.
    
    Args:
        trajectories: List of trajectory dictionaries
        offset_to_origin: If True, offset each trajectory so first timestep is at (0, 0)
        offset_z: If True, also offset z coordinate; if False, keep original z
        verbose: Whether to print normalization details
        
    Returns:
        List of trajectory dictionaries with normalized positions
    """
    normalized_trajectories = []
    
    if verbose and offset_to_origin:
        print(f"\nNormalizing positions to origin (offset_z={offset_z})...")
        print(f"{'Trajectory':<30} | {'Initial X':<10} | {'Initial Y':<10} | {'Initial Z':<10}")
        print("-" * 70)
    
    for traj in trajectories:
        normalized_traj = traj.copy()
        
        # Check if we should offset the converted IMU frame positions or original positions
        if 'position_imu_frame' in traj:
            positions = traj['position_imu_frame']
            key = 'position_imu_frame'
        else:
            positions = traj['position']
            key = 'position'
        
        if offset_to_origin:
            offset_positions, initial_offset = offset_trajectory_to_origin(positions, offset_z=offset_z)
            normalized_traj[key] = offset_positions
            normalized_traj['initial_offset'] = initial_offset
            
            if verbose:
                print(f"{traj['filename']:<30} | {initial_offset[0]:9.3f}m | {initial_offset[1]:9.3f}m | {initial_offset[2]:9.3f}m")
        
        normalized_trajectories.append(normalized_traj)
    
    if verbose and offset_to_origin:
        print("-" * 70)
        print(f"Normalized {len(normalized_trajectories)} trajectories")
    
    return normalized_trajectories


def visualize_conversion(traj, save_path=None):
    """
    Visualize the position conversion for a single trajectory.
    Shows original Optitrack positions and converted IMU-frame positions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original Optitrack positions
    ax1 = axes[0]
    ax1.plot(traj['position'][:, 0], traj['position'][:, 1], 'b-', alpha=0.6, label='Optitrack Frame')
    ax1.scatter(traj['position'][0, 0], traj['position'][0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(traj['position'][-1, 0], traj['position'][-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Optitrack Global Frame')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Converted IMU-frame positions
    ax2 = axes[1]
    ax2.plot(traj['position_imu_frame'][:, 0], traj['position_imu_frame'][:, 1], 'r-', alpha=0.6, label='IMU Frame')
    ax2.scatter(traj['position_imu_frame'][0, 0], traj['position_imu_frame'][0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax2.scatter(traj['position_imu_frame'][-1, 0], traj['position_imu_frame'][-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('IMU Global Frame (Converted)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Overlay both for comparison
    ax3 = axes[2]
    ax3.plot(traj['position'][:, 0], traj['position'][:, 1], 'b-', alpha=0.4, label='Optitrack Frame', linewidth=2)
    ax3.plot(traj['position_imu_frame'][:, 0], traj['position_imu_frame'][:, 1], 'r-', alpha=0.4, label='IMU Frame', linewidth=2)
    ax3.scatter(traj['position'][0, 0], traj['position'][0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_title(f"Overlay (Yaw Offset: {np.rad2deg(traj['yaw_offset']):.2f}°)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.suptitle(f"Position Frame Conversion: {traj['filename']}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def analyze_yaw_offsets(trajectories):
    """
    Analyze the distribution of yaw offsets across all trajectories.
    """
    yaw_offsets = []
    yaw_stds = []
    
    for traj in trajectories:
        optitrack_quat = traj['optitrack_quaternion']
        imu_quat = traj['imu_quaternion']
        
        # Extract yaw angles
        optitrack_yaw = quaternion_to_yaw(optitrack_quat)
        imu_yaw = quaternion_to_yaw(imu_quat)
        
        # Calculate yaw offset for each timestep
        yaw_diff = optitrack_yaw - imu_yaw
        yaw_diff = np.arctan2(np.sin(yaw_diff), np.cos(yaw_diff))
        
        yaw_offsets.append(np.median(yaw_diff))
        yaw_stds.append(np.std(yaw_diff))
    
    yaw_offsets = np.array(yaw_offsets)
    yaw_stds = np.array(yaw_stds)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram of yaw offsets
    ax1 = axes[0, 0]
    ax1.hist(np.rad2deg(yaw_offsets), bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(np.rad2deg(np.median(yaw_offsets)), color='red', linestyle='--', linewidth=2, label=f'Median: {np.rad2deg(np.median(yaw_offsets)):.2f}°')
    ax1.axvline(np.rad2deg(np.mean(yaw_offsets)), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.rad2deg(np.mean(yaw_offsets)):.2f}°')
    ax1.set_xlabel('Yaw Offset (degrees)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Yaw Offsets Across Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of standard deviations
    ax2 = axes[0, 1]
    ax2.hist(np.rad2deg(yaw_stds), bins=30, edgecolor='black', alpha=0.7, color='orange')
    ax2.axvline(np.rad2deg(np.median(yaw_stds)), color='red', linestyle='--', linewidth=2, label=f'Median: {np.rad2deg(np.median(yaw_stds)):.2f}°')
    ax2.set_xlabel('Yaw Offset Std Dev (degrees)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Yaw Offset Variability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Scatter plot: offset vs std dev
    ax3 = axes[1, 0]
    scatter = ax3.scatter(np.rad2deg(yaw_offsets), np.rad2deg(yaw_stds), alpha=0.6, c=np.arange(len(yaw_offsets)), cmap='viridis')
    ax3.set_xlabel('Yaw Offset (degrees)')
    ax3.set_ylabel('Yaw Offset Std Dev (degrees)')
    ax3.set_title('Yaw Offset vs Variability')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Trajectory Index')
    
    # Time series of yaw offsets
    ax4 = axes[1, 1]
    ax4.plot(np.rad2deg(yaw_offsets), 'o-', alpha=0.6)
    ax4.axhline(np.rad2deg(np.median(yaw_offsets)), color='red', linestyle='--', linewidth=2, label='Median')
    ax4.axhline(np.rad2deg(np.mean(yaw_offsets)), color='blue', linestyle='--', linewidth=2, label='Mean')
    ax4.set_xlabel('Trajectory Index')
    ax4.set_ylabel('Yaw Offset (degrees)')
    ax4.set_title('Yaw Offset Across Trajectories')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Yaw Offset Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Yaw Offset Statistics")
    print("="*60)
    print(f"Mean offset: {np.rad2deg(np.mean(yaw_offsets)):.2f}°")
    print(f"Median offset: {np.rad2deg(np.median(yaw_offsets)):.2f}°")
    print(f"Std dev of offsets: {np.rad2deg(np.std(yaw_offsets)):.2f}°")
    print(f"Min offset: {np.rad2deg(np.min(yaw_offsets)):.2f}°")
    print(f"Max offset: {np.rad2deg(np.max(yaw_offsets)):.2f}°")
    print(f"\nMean variability (std dev within trajectory): {np.rad2deg(np.mean(yaw_stds)):.2f}°")
    print(f"Median variability: {np.rad2deg(np.median(yaw_stds)):.2f}°")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Convert positions from Optitrack to IMU frame')
    parser.add_argument('--folder', type=str, default='examples/cleaned',
                        help='Folder containing cleaned trajectory CSV files')
    parser.add_argument('--method', type=str, choices=['mean', 'median'], default='median',
                        help='Method to calculate yaw offset (mean or median)')
    parser.add_argument('--normalize', action='store_true',
                        help='Offset positions so first timestep is at (0, 0)')
    parser.add_argument('--offset-z', action='store_true',
                        help='Also offset z coordinate to 0 (only with --normalize)')
    parser.add_argument('--calculate-velocity', action='store_true',
                        help='Calculate velocity from position_imu_frame')
    parser.add_argument('--visualize', type=int, default=None,
                        help='Visualize specific trajectory by index')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze yaw offsets across all trajectories')
    parser.add_argument('--save', type=str, default=None,
                        help='Save visualization to file')
    
    args = parser.parse_args()
    
    # Load trajectories
    print(f"Loading trajectories from {args.folder}...")
    trajectories = load_trajectories(args.folder, verbose=False, max_length=1000)
    print(f"Loaded {len(trajectories)} trajectories\n")
    
    # Convert positions
    use_median = (args.method == 'median')
    converted_trajectories = convert_positions_to_imu_frame(
        trajectories, 
        use_median=use_median, 
        verbose=True
    )
    
    # Normalize positions if requested
    if args.normalize:
        converted_trajectories = normalize_trajectory_positions(
            converted_trajectories,
            offset_to_origin=True,
            offset_z=args.offset_z,
            verbose=True
        )
    
    # Calculate velocity if requested
    if args.calculate_velocity:
        converted_trajectories = calculate_velocity_from_position(
            converted_trajectories,
            position_key='position_imu_frame',
            dt=0.02,  # 50Hz control frequency
            verbose=True
        )
    
    # Analyze yaw offsets if requested
    if args.analyze:
        print("\nAnalyzing yaw offsets...")
        fig = analyze_yaw_offsets(converted_trajectories)
        if args.save:
            analysis_path = args.save.replace('.png', '_analysis.png')
            fig.savefig(analysis_path, dpi=150, bbox_inches='tight')
            print(f"Saved analysis to {analysis_path}")
        plt.show()
    
    # Visualize specific trajectory if requested
    if args.visualize is not None:
        if 0 <= args.visualize < len(converted_trajectories):
            print(f"\nVisualizing trajectory {args.visualize}...")
            save_path = args.save if args.save else None
            visualize_conversion(converted_trajectories[args.visualize], save_path)
        else:
            print(f"Error: Trajectory index {args.visualize} out of range (0-{len(converted_trajectories)-1})")
    
    return converted_trajectories


if __name__ == "__main__":
    trajectories = main()
