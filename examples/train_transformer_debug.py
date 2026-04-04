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
import glob as glob_module
import pickle
import numpy as np
import torch
import time
import os

from capyformer import HFActionChunkingTrainer, TrajectoryDataset
from capyformer.trainer import Trainer

DEFAULT_ROLLOUT_PATH = "batch_rollouts.pkl"
DEFAULT_LOG_DIR = "./debug/transformer_debug"


def create_dataset_class(rollout_paths, normalize_actions=True):
    """Create dataset class.
    
    Args:
        rollout_paths: List of paths to rollout data files.
        normalize_actions: If True, normalize actions to zero-mean unit-variance
            across the combined dataset. This is critical when combining data from
            robots with different optimized poses (different default_dof_pos), since
            their absolute action ranges differ wildly even though the relative
            motion patterns are similar.
    """
    if isinstance(rollout_paths, str):
        rollout_paths = [rollout_paths]
        
    class QuadrupedDataset(TrajectoryDataset):
        def _setup_dataset(self, dataset_config):
            self.trajectories = []
            for path in rollout_paths:
                print(f"Loading data from: {path}")
                rollout_data = pickle.load(open(path, "rb"))
                self.trajectories.extend(rollout_data["trajectories"])
            
            self.input_keys = [f"module{i}" for i in range(5)]
            self.target_key = "actions"
            print(f"Total loaded {len(self.trajectories)} trajectories from {len(rollout_paths)} files")
            self.pad_value = 0.0
            self.val_split = 0.1
            
            # Normalize actions across the combined dataset
            if normalize_actions:
                all_actions = np.concatenate(
                    [traj["actions"] for traj in self.trajectories], axis=0
                )
                self.action_mean = np.mean(all_actions, axis=0)
                self.action_std = np.std(all_actions, axis=0) + 1e-6
                
                print(f"  Action normalization enabled:")
                print(f"    action_mean = {self.action_mean}")
                print(f"    action_std  = {self.action_std}")
                
                for traj in self.trajectories:
                    traj["actions"] = (traj["actions"] - self.action_mean) / self.action_std
                
                print(f"    Normalized action range: [{all_actions.min():.3f}, {all_actions.max():.3f}] "
                      f"-> [{((all_actions - self.action_mean) / self.action_std).min():.3f}, "
                      f"{((all_actions - self.action_mean) / self.action_std).max():.3f}]")
            else:
                self.action_mean = None
                self.action_std = None
    
    return QuadrupedDataset


def create_isaac_sim_dataset_class(rollout_paths, normalize_actions=True):
    """Create dataset class for Isaac Sim expert data.
    
    Isaac Sim data has flat observations (T, obs_dim), actions (T, act_dim),
    and commands (T, cmd_dim). We treat 'observations' and 'commands' as
    two separate input tokens for the transformer.
    
    Args:
        rollout_paths: List of paths to rollout data files.
        normalize_actions: If True, normalize actions to zero-mean unit-variance.
    """
    if isinstance(rollout_paths, str):
        rollout_paths = [rollout_paths]
        
    class IsaacSimDataset(TrajectoryDataset):
        def _setup_dataset(self, dataset_config):
            self.trajectories = []
            for path in rollout_paths:
                print(f"Loading Isaac Sim data from: {path}")
                rollout_data = pickle.load(open(path, "rb"))
                raw_trajs = rollout_data["trajectories"]
                
                for traj in raw_trajs:
                    # Keep only the keys we need as input tokens + target
                    # NOTE: We rename "observations" -> "obs" to avoid triggering
                    # TrajectoryDataset's legacy format detection, which treats
                    # the combo of "observations" + "actions" keys as legacy format
                    # and expects observations to be a dict of sub-observations.
                    clean_traj = {}
                    
                    # Input tokens
                    if "observations" in traj:
                        clean_traj["obs"] = np.array(traj["observations"], dtype=np.float32)
                    if "commands" in traj:
                        clean_traj["commands"] = np.array(traj["commands"], dtype=np.float32)
                    
                    # Target
                    clean_traj["actions"] = np.array(traj["actions"], dtype=np.float32)
                    
                    self.trajectories.append(clean_traj)
            
            self.input_keys = ["obs", "commands"]
            self.target_key = "actions"
            self.pad_value = 0.0
            self.val_split = 0.1
            
            print(f"Total loaded {len(self.trajectories)} trajectories from {len(rollout_paths)} files")
            if self.trajectories:
                t0 = self.trajectories[0]
                for k, v in t0.items():
                    if isinstance(v, np.ndarray):
                        print(f"  {k}: shape={v.shape}")
            
            # Normalize actions across the combined dataset
            if normalize_actions:
                all_actions = np.concatenate(
                    [traj["actions"] for traj in self.trajectories], axis=0
                )
                self.action_mean = np.mean(all_actions, axis=0)
                self.action_std = np.std(all_actions, axis=0) + 1e-6
                
                print(f"  Action normalization enabled:")
                print(f"    action_mean = {self.action_mean}")
                print(f"    action_std  = {self.action_std}")
                
                for traj in self.trajectories:
                    traj["actions"] = (traj["actions"] - self.action_mean) / self.action_std
            else:
                self.action_mean = None
                self.action_std = None
    
    return IsaacSimDataset


def evaluate_per_joint(trainer, dataset, n_trajs=10):
    """Evaluate correlation per joint."""
    inference = trainer.get_inference()
    
    correlations = []
    all_pred = []
    all_true = []
    
    input_names = dataset.input_token_names
    n_joints = dataset.target_dim
    
    for traj_idx in range(min(n_trajs, len(dataset.val_trajectories))):
        traj = dataset.val_trajectories[traj_idx]
        inference.reset()
        
        for t in range(min(50, len(traj[dataset.target_key]))):
            state = {name: traj[name][t] for name in input_names if name in traj}
            pred = inference.step(state)
            true = traj[dataset.target_key][t]
            all_pred.append(pred)
            all_true.append(true)
    
    all_pred = np.array(all_pred)
    all_true = np.array(all_true)
    
    print("\nPer-joint evaluation:")
    for j in range(n_joints):
        corr = np.corrcoef(all_pred[:, j], all_true[:, j])[0, 1]
        mae = np.abs(all_pred[:, j] - all_true[:, j]).mean()
        print(f"  Joint {j}: correlation={corr:.3f}, MAE={mae:.3f}")
        correlations.append(corr)
    
    return correlations


def evaluate_robot_rollout(trainer, epoch, n_steps=500, seed=42, 
                           action_mean=None, action_std=None):
    """
    Run robot rollout with the trained policy and record video.
    
    Args:
        trainer: The HFActionChunkingTrainer with trained model
        epoch: Current epoch (for video naming)
        n_steps: Number of steps to run
        seed: Random seed for environment
        action_mean: If provided, denormalize predicted actions: action = pred * std + mean
        action_std: If provided, denormalize predicted actions: action = pred * std + mean
    
    Returns:
        dict with metrics (total_reward, distance_traveled, avg_reward)
    """
    from metamachine.environments.configs.config_registry import ConfigRegistry
    from metamachine.environments.env_sim import MetaMachine
    
    # Get inference policy
    policy = trainer.get_inference()
    policy.reset()
    
    if action_mean is not None:
        print(f"  Action denormalization: mean={action_mean}, std={action_std}")
    
    # Create environment with video recording
    cfg = ConfigRegistry.create_from_name("modular_quadruped")
    cfg.control.default_dof_pos = [0, 0, 0, 0, 0]
    cfg.control.symmetric_limit = 100
    
    # Explicitly set initialization pose to avoid collapsed start (since default_dof_pos is 0)
    if 'initialization' not in cfg:
        from omegaconf import OmegaConf
        cfg['initialization'] = OmegaConf.create()
    cfg.initialization.init_joint_pos = [0, -1., 1., 1., -1.]
    
    cfg.simulation.render = True
    cfg.simulation.render_mode = "mp4"
    cfg.simulation.video_record_interval = 1  # Record every episode
    
    env = MetaMachine(cfg)
    obs, _ = env.reset(seed=seed)
    # Get the log directory from MetaMachine
    env_log_dir = getattr(env, '_log_dir', './logs')
    print(f"  Video will be saved to: {env_log_dir}")
    
    # Get observation structure info for debugging
    obs_info = env.state.get_modular_observation_info()
    print(f"  Observation structure: {obs_info['num_modules']} modules × {obs_info['per_module_obs_size']} + {obs_info['global_obs_size']} global = {obs_info['total_obs_size']} total")
    
    # Run rollout
    total_reward = 0.0
    positions = []
    
    for step in range(n_steps):
        t0 = time.time()
        
        # Get action from policy - use unified API for observation parsing
        # This ensures alignment between training data and inference
        state = env.state.flat_obs_to_dict(obs)
        action = policy.step(state)
        
        # Denormalize actions if normalization stats are provided
        if action_mean is not None and action_std is not None:
            action = action * action_std + action_mean
        
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
    parser.add_argument("--rollout-path", nargs="+", default=[DEFAULT_ROLLOUT_PATH])
    parser.add_argument("--rollout-dir", type=str, default=None,
                        help="Directory containing *_expert_data.pkl files. "
                             "If provided, all matching PKL files will be loaded "
                             "(overrides --rollout-path).")
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--no-flow-matching", action="store_true", 
                        help="Disable flow matching (simpler, for debugging)")
    parser.add_argument("--action-horizon", type=int, default=1,
                        help="Action horizon (1 = no chunking, for debugging)")
    parser.add_argument("--context-len", type=int, default=20,
                        help="Context length (shorter = simpler)")
    parser.add_argument("--log-dir", type=str, default=DEFAULT_LOG_DIR,
                        help="Directory for logs and videos")
    parser.add_argument("--robot-validation", default=True, action="store_true",
                        help="Run robot rollout validation with video recording")
    parser.add_argument("--no-robot-validation", dest="robot_validation", action="store_false",
                        help="Disable robot rollout validation (use when env is not available)")
    parser.add_argument("--robot-val-steps", type=int, default=200,
                        help="Number of steps for robot validation")
    parser.add_argument("--use-simple-trainer", action="store_true",
                        help="Use the simple Trainer instead of HFActionChunkingTrainer")
    parser.add_argument("--use-lora", action="store_true", default=False,
                        help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization")
    parser.add_argument("--resume-checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from (e.g. ./logs/transformer_multi_expert/final_model.pt)")
    parser.add_argument("--normalize-actions", default=True, action="store_true",
                        help="Normalize actions to zero-mean unit-variance (recommended for multi-robot data)")
    parser.add_argument("--no-normalize-actions", dest="normalize_actions", action="store_false",
                        help="Disable action normalization")
    parser.add_argument("--isaac-sim-format", action="store_true", default=False,
                        help="Use Isaac Sim flat observation format (observations + commands as input tokens)")
    parser.add_argument("--eval-config", type=str, default=None,
                        help="Path to env config YAML for periodic eval with video (e.g. path/to/new_five_modules_v11.yaml)")
    parser.add_argument("--eval-steps", type=int, default=200,
                        help="Number of steps for periodic eval rollouts")
    parser.add_argument("--validation-freq", type=int, default=500,
                        help="Run validation + video callback every N epochs")
    args = parser.parse_args()
    
    # If --rollout-dir is provided, glob all PKL files from that directory
    if args.rollout_dir is not None:
        pattern = os.path.join(args.rollout_dir, "*_expert_data.pkl")
        found = sorted(glob_module.glob(pattern))
        if not found:
            raise FileNotFoundError(f"No *_expert_data.pkl files found in {args.rollout_dir}")
        args.rollout_path = found
        print(f"Found {len(found)} PKL files in {args.rollout_dir}:")
        for p in found:
            print(f"  {os.path.basename(p)}")
    
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
    if args.isaac_sim_format:
        print("Using Isaac Sim flat observation format")
        DatasetClass = create_isaac_sim_dataset_class(args.rollout_path, normalize_actions=args.normalize_actions)
    else:
        DatasetClass = create_dataset_class(args.rollout_path, normalize_actions=args.normalize_actions)
    dataset = DatasetClass({"val_split": 0.1}, context_len=args.context_len)
    
    # Store action normalization stats for later use
    action_mean = getattr(dataset, 'action_mean', None)
    action_std = getattr(dataset, 'action_std', None)
    
    
    # Define validation callback
    def robot_validation_callback(trainer_instance, epoch_num):
        """Callback to run robot validation during training."""
        if not args.robot_validation:
            return
            
        print(f"\n[Robot Validation] Running rollout and recording video for epoch {epoch_num}...")
        try:
            metrics = evaluate_robot_rollout(
                trainer_instance, 
                epoch=epoch_num, 
                n_steps=args.robot_val_steps,
                action_mean=action_mean,
                action_std=action_std,
            )
            print(f"  Total reward: {metrics['total_reward']:.2f}")
            print(f"  Avg reward/step: {metrics['avg_reward']:.4f}")
            print(f"  Distance traveled: {metrics['distance']:.3f}m")
            print(f"  Steps completed: {metrics['steps']}")
        except Exception as e:
            print(f"  Robot validation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Define Isaac Sim eval callback (runs MuJoCo eval with video)
    def isaac_sim_eval_callback(trainer_instance, epoch_num):
        """Run Isaac Sim transformer evaluation on MuJoCo with video recording."""
        print(f"\n[Isaac Sim Eval] Running eval with video for epoch {epoch_num}...")
        try:
            import sys
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            from evaluate_isaac_sim_transformer import evaluate as isaac_eval
            import json
            
            # Save action norm to checkpoint so evaluate can load it
            if action_mean is not None:
                checkpoint_path = os.path.join(args.log_dir, "latest_checkpoint.pt")
                if os.path.exists(checkpoint_path):
                    ckpt = torch.load(checkpoint_path, map_location="cpu")
                    ckpt["action_norm"] = {
                        "action_mean": action_mean.tolist(),
                        "action_std": action_std.tolist(),
                    }
                    torch.save(ckpt, checkpoint_path)
            
            # Run evaluation
            isaac_eval(
                checkpoint_path=os.path.join(args.log_dir, "latest_checkpoint.pt"),
                config_path=args.eval_config,
                n_steps=args.eval_steps,
                seed=42,
                forward_speed=0.5,
                turn_rate=0.0,
            )
            print(f"  [Isaac Sim Eval] Epoch {epoch_num} eval complete!")
        except Exception as e:
            print(f"  [Isaac Sim Eval] Failed: {e}")
            import traceback
            traceback.print_exc()

    # Determine which callback to use
    if args.eval_config and args.isaac_sim_format:
        active_callback = isaac_sim_eval_callback
    elif args.robot_validation:
        active_callback = robot_validation_callback
    else:
        active_callback = None

    # Create trainer based on selection
    if args.use_simple_trainer:
        # Use the simpler Trainer from trainer.py
        trainer = Trainer(
            dataset,
            log_dir=args.log_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            validation_freq=args.validation_freq,
            action_is_velocity=True,
            validation_callback=active_callback,
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
            
            use_lora=args.use_lora,
            load_in_4bit=args.load_in_4bit,
            freeze_backbone=False, # Train the whole model (270M parameters)
            
            batch_size=args.batch_size,
            learning_rate=args.lr,
            validation_freq=args.validation_freq,
            action_is_velocity=True,
            validation_callback=active_callback,
        )
    
    # Load checkpoint if resuming
    if args.resume_checkpoint:
        print(f"\n[Resume] Loading checkpoint from: {args.resume_checkpoint}")
        trainer.load(args.resume_checkpoint)
        print(f"[Resume] Model loaded, continuing training on new data...")
    
    # Train in one go, validation callbacks will handle the rest
    print(f"\n[Training] Training for {args.n_epochs} epochs...")
    trainer.learn(n_epochs=args.n_epochs)
    
    print(f"\n[Evaluation] Final per-joint analysis:")
    correlations = evaluate_per_joint(trainer, dataset)
    
    # Check if any joint has very low correlation
    if min(correlations) < 0.2:
        bad_joints = [j for j, c in enumerate(correlations) if c < 0.2]
        print(f"\n⚠️ WARNING: Joints {bad_joints} have low correlation!")
        print("   Consider: more epochs, different learning rate, or checking data")
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)
    
    # Save model
    model_path = os.path.join(args.log_dir, "final_model")
    trainer.save(model_path)
    print(f"Saved to {model_path}.pt")
    
    # Save action normalization stats alongside the checkpoint
    if action_mean is not None:
        import json
        action_norm_path = os.path.join(args.log_dir, "action_norm.json")
        action_norm = {
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
        }
        with open(action_norm_path, "w") as f:
            json.dump(action_norm, f, indent=2)
        print(f"Action normalization stats saved to {action_norm_path}")
        
        # Also patch into the checkpoint so it's self-contained
        checkpoint_path = model_path + ".pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        checkpoint["action_norm"] = action_norm
        torch.save(checkpoint, checkpoint_path)
        print(f"Action normalization stats also saved in checkpoint")
    
    # Final robot validation (extra long)
    if args.robot_validation:
        print("\n[Final Robot Validation]")
        try:
            metrics = evaluate_robot_rollout(
                trainer,
                epoch=args.n_epochs,
                n_steps=args.robot_val_steps * 2,  # Longer final rollout
                action_mean=action_mean,
                action_std=action_std,
            )
            print(f"  Final total reward: {metrics['total_reward']:.2f}")
            print(f"  Final distance: {metrics['distance']:.3f}m")
        except Exception as e:
            print(f"  Final robot validation failed: {e}")


if __name__ == "__main__":
    main()

