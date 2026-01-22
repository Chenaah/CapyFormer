#!/usr/bin/env python3
"""
Debug training script for transformer policy.

Key fixes:
1. Start with simpler model (no flow matching) to verify learning
2. More frequent validation to track progress
3. Per-joint loss analysis
4. Robot rollout validation with video recording
"""

import argparse
import pickle
import numpy as np
import torch
import time
import os

from capyformer import HFActionChunkingTrainer, TrajectoryDataset
from capyformer.trainer import Trainer

DEFAULT_ROLLOUT_PATH = "batch_rollouts.pkl"
DEFAULT_LOG_DIR = "./debug/transformer_debug"


def create_dataset_class(rollout_path):
    """Create dataset class."""
    class QuadrupedDataset(TrajectoryDataset):
        def _setup_dataset(self, dataset_config):
            rollout_data = pickle.load(open(rollout_path, "rb"))
            self.trajectories = rollout_data["trajectories"]
            self.input_keys = [f"module{i}" for i in range(5)]
            self.target_key = "actions"
            print(f"Loaded {len(self.trajectories)} trajectories")
            self.pad_value = 0.0
            self.val_split = 0.1
    
    return QuadrupedDataset


def evaluate_per_joint(trainer, dataset, n_trajs=10):
    """Evaluate correlation per joint."""
    inference = trainer.get_inference()
    
    correlations = []
    all_pred = []
    all_true = []
    
    for traj_idx in range(min(n_trajs, len(dataset.val_trajectories))):
        traj = dataset.val_trajectories[traj_idx]
        inference.reset()
        
        for t in range(min(50, len(traj['actions']))):
            state = {f'module{i}': traj[f'module{i}'][t] for i in range(5)}
            pred = inference.step(state)
            true = traj['actions'][t]
            all_pred.append(pred)
            all_true.append(true)
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    print("\nPer-joint evaluation:")
    for j in range(5):
        corr = np.corrcoef(all_pred[:, j], all_true[:, j])[0, 1]
        mae = np.abs(all_pred[:, j] - all_true[:, j]).mean()
        print(f"  Joint {j}: correlation={corr:.3f}, MAE={mae:.3f}")
        correlations.append(corr)
    
    return correlations


def evaluate_robot_rollout(trainer, epoch, n_steps=500, seed=42):
    """
    Run robot rollout with the trained policy and record video.
    
    Args:
        trainer: The HFActionChunkingTrainer with trained model
        epoch: Current epoch (for video naming)
        n_steps: Number of steps to run
        seed: Random seed for environment
    
    Returns:
        dict with metrics (total_reward, distance_traveled, avg_reward)
    """
    from metamachine.environments.configs.config_registry import ConfigRegistry
    from metamachine.environments.env_sim import MetaMachine
    
    # Get inference policy
    policy = trainer.get_inference()
    policy.reset()
    
    # Create environment with video recording
    cfg = ConfigRegistry.create_from_name("modular_quadruped")
    cfg.control.default_dof_pos = [0, 0, 0, 0, 0]
    cfg.simulation.render = True
    cfg.simulation.render_mode = "mp4"
    cfg.simulation.video_record_interval = 1  # Record every episode
    
    env = MetaMachine(cfg)
    obs, _ = env.reset(seed=seed)
    
    # Get the log directory from MetaMachine
    env_log_dir = getattr(env, '_log_dir', './logs')
    print(f"  Video will be saved to: {env_log_dir}")
    
    # Run rollout
    total_reward = 0.0
    positions = []
    
    for step in range(n_steps):
        t0 = time.time()
        
        # Get action from policy
        state = {f'module{i}': obs[i*8:(i+1)*8] for i in range(5)}
        action = policy.step(state)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Record position
        if hasattr(env, 'state'):
            positions.append(env.state.raw.pos_world[:2].copy())
        
        # Maintain real-time for smoother video
        elapsed = time.time() - t0
        sleep_time = max(0, cfg.control.dt * 0.5 - elapsed)  # Half speed for video
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        if done or truncated:
            print(f"  Episode ended at step {step}")
            break
    
    # Calculate distance traveled
    distance = 0.0
    if len(positions) > 1:
        positions = np.array(positions)
        distance = np.linalg.norm(positions[-1] - positions[0])
    
    # Close environment (this should save the video)
    env.close()
    
    # Rename video file to include epoch
    try:
        import glob
        video_files = glob.glob(os.path.join(env_log_dir, "episode_*.mp4"))
        if video_files:
            latest_video = max(video_files, key=os.path.getmtime)
            new_name = os.path.join(env_log_dir, f"rollout_epoch_{epoch:04d}.mp4")
            os.rename(latest_video, new_name)
            print(f"  Video saved: {new_name}")
    except Exception as e:
        print(f"  Warning: Could not rename video: {e}")
    
    metrics = {
        'total_reward': total_reward,
        'avg_reward': total_reward / n_steps,
        'distance': distance,
        'steps': step + 1,
        'log_dir': env_log_dir,
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-path", default=DEFAULT_ROLLOUT_PATH)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-flow-matching", action="store_true", 
                        help="Disable flow matching (simpler, for debugging)")
    parser.add_argument("--action-horizon", type=int, default=1,
                        help="Action horizon (1 = no chunking, for debugging)")
    parser.add_argument("--context-len", type=int, default=20,
                        help="Context length (shorter = simpler)")
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR,
                        help="Directory for logs and videos")
    parser.add_argument("--robot-validation", action="store_true",
                        help="Run robot rollout validation with video recording")
    parser.add_argument("--robot-val-steps", type=int, default=500,
                        help="Number of steps for robot validation")
    parser.add_argument("--use-simple-trainer", action="store_true",
                        help="Use the simple Trainer instead of HFActionChunkingTrainer")
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    print("=" * 70)
    print("DEBUG TRAINING FOR TRANSFORMER POLICY")
    print("=" * 70)
    print(f"Settings:")
    print(f"  Trainer: {'Trainer (simple)' if args.use_simple_trainer else 'HFActionChunkingTrainer'}")
    print(f"  Flow matching: {not args.no_flow_matching}")
    print(f"  Action horizon: {args.action_horizon}")
    print(f"  Context length: {args.context_len}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Epochs: {args.n_epochs}")
    print(f"  Log directory: {args.log_dir}")
    print(f"  Robot validation: {args.robot_validation}")
    print("=" * 70)
    
    # Create dataset
    DatasetClass = create_dataset_class(args.rollout_path)
    dataset = DatasetClass({"val_split": 0.1}, context_len=args.context_len)
    
    # Create trainer based on selection
    if args.use_simple_trainer:
        # Use the simpler Trainer from trainer.py
        trainer = Trainer(
            dataset,
            log_dir=args.log_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            validation_freq=10,  # More frequent validation
            action_is_velocity=True,
        )
    else:
        # Use HFActionChunkingTrainer with debug-friendly settings
        trainer = HFActionChunkingTrainer(
            dataset,
            model_name="google/gemma-3-270m",
            log_dir=args.log_dir,
            
            # Simpler settings for debugging
            action_horizon=args.action_horizon,
            execute_horizon=1,
            
            # No flow matching for cleaner debugging
            use_flow_matching=not args.no_flow_matching,
            flow_matching_steps=10 if not args.no_flow_matching else 0,
            
            use_lora=False,
            freeze_backbone=False,
            
            batch_size=args.batch_size,
            learning_rate=args.lr,
            validation_freq=10,  # More frequent validation
            action_is_velocity=True,
        )
    
    # Train in phases and check per-joint progress
    epochs_trained = 0
    phase_epochs = [500, 1000, 2000, args.n_epochs]
    
    for epoch_target in phase_epochs:
        if epoch_target > args.n_epochs:
            break
            
        epochs_to_train = epoch_target - epochs_trained
        if epochs_to_train <= 0:
            continue
            
        print(f"\n[Training] Training {epochs_to_train} epochs (total: {epoch_target})...")
        trainer.learn(n_epochs=epochs_to_train)
        epochs_trained = epoch_target
        
        print(f"\n[Evaluation] After {epoch_target} epochs:")
        correlations = evaluate_per_joint(trainer, dataset)
        
        # Check if any joint has very low correlation
        if min(correlations) < 0.2:
            bad_joints = [j for j, c in enumerate(correlations) if c < 0.2]
            print(f"\n⚠️ WARNING: Joints {bad_joints} have low correlation!")
            print("   Consider: more epochs, different learning rate, or checking data")
        
        # Robot rollout validation with video
        if args.robot_validation:
            print(f"\n[Robot Validation] Running rollout and recording video...")
            try:
                metrics = evaluate_robot_rollout(
                    trainer, 
                    epoch=epoch_target, 
                    n_steps=args.robot_val_steps
                )
                print(f"  Total reward: {metrics['total_reward']:.2f}")
                print(f"  Avg reward/step: {metrics['avg_reward']:.4f}")
                print(f"  Distance traveled: {metrics['distance']:.3f}m")
                print(f"  Steps completed: {metrics['steps']}")
            except Exception as e:
                print(f"  Robot validation failed: {e}")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    # Save model
    model_path = os.path.join(args.log_dir, "final_model")
    trainer.save(model_path)
    print(f"Saved to {model_path}.pt")
    
    # Final robot validation
    if args.robot_validation:
        print("\n[Final Robot Validation]")
        try:
            metrics = evaluate_robot_rollout(
                trainer,
                epoch=args.n_epochs,
                n_steps=args.robot_val_steps * 2  # Longer final rollout
            )
            print(f"  Final total reward: {metrics['total_reward']:.2f}")
            print(f"  Final distance: {metrics['distance']:.3f}m")
        except Exception as e:
            print(f"  Final robot validation failed: {e}")


if __name__ == "__main__":
    main()

