"""
Train a transformer policy using CapyFormer and run inference.

Requires: pip install git+https://github.com/Chenaah/CapyFormer.git
"""

import argparse
import pickle
import time
import os
import numpy as np

try:
    from capyformer import HFTrainer, TrajectoryDataset, HFActionChunkingTrainer
except ImportError:
    raise ImportError("Please install capyformer: pip install git+https://github.com/Chenaah/CapyFormer.git")

from metamachine.environments.configs.config_registry import ConfigRegistry
from metamachine.environments.env_sim import MetaMachine

# ============ Default Configuration ============
DEFAULT_ROLLOUT_PATHS = ["batch_rollouts.pkl"]
DEFAULT_MODEL_SAVE_PATH = "./models/my_model11"
DEFAULT_LOG_DIR = "./debug"
DEFAULT_CONTEXT_LEN = 20
DEFAULT_N_EPOCHS = 2000
DEFAULT_BATCH_SIZE = 32
# ===============================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a transformer policy using CapyFormer and run inference."
    )
    parser.add_argument(
        "--rollout-paths", 
        nargs="+", 
        default=DEFAULT_ROLLOUT_PATHS,
        help=f"Paths to rollout pickle files (default: {DEFAULT_ROLLOUT_PATHS})"
    )
    parser.add_argument(
        "--model-save-path", 
        type=str, 
        default=DEFAULT_MODEL_SAVE_PATH,
        help=f"Path to save the trained model (default: {DEFAULT_MODEL_SAVE_PATH})"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default=DEFAULT_LOG_DIR,
        help=f"Directory for logs (default: {DEFAULT_LOG_DIR})"
    )
    parser.add_argument(
        "--context-len", 
        type=int, 
        default=DEFAULT_CONTEXT_LEN,
        help=f"Context length for the transformer (default: {DEFAULT_CONTEXT_LEN})"
    )
    parser.add_argument(
        "--n-epochs", 
        type=int, 
        default=DEFAULT_N_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_N_EPOCHS})"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=None,
        help="Path to checkpoint to load for inference (skips training if provided)"
    )
    parser.add_argument(
        "--robot-validation",
        action="store_true",
        help="Run robot rollout validation with video recording during training"
    )
    parser.add_argument(
        "--robot-val-steps",
        type=int,
        default=500,
        help="Number of steps for robot validation rollout"
    )
    parser.add_argument(
        "--robot-val-interval",
        type=int,
        default=100,
        help="Run robot validation every N epochs"
    )
    parser.add_argument(
        "--no-flow-matching",
        action="store_true",
        help="Disable flow matching (use direct regression)"
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=4,
        help="Action horizon for action chunking"
    )
    return parser.parse_args()


def create_dataset_class(rollout_paths):
    """Factory function to create a dataset class with the given rollout paths."""
    class ThreeModulesController(TrajectoryDataset):
        def _setup_dataset(self, dataset_config):
            self.trajectories = []
            for path in rollout_paths:
                rollout_data = pickle.load(open(path, "rb"))
                self.trajectories.extend(rollout_data["trajectories"])
            self.input_keys = [f"module{i}" for i in range(5)]
            self.target_key = "actions"
            print(f"Loaded {len(self.trajectories)} trajectories from {rollout_paths}")

            self.pad_value = 0.0
            self.val_split = 0.1
    
    return ThreeModulesController


def evaluate_robot_rollout(trainer, epoch, n_steps=500, seed=42):
    """
    Run robot rollout with the trained policy and record video.
    
    Args:
        trainer: The HFActionChunkingTrainer with trained model
        epoch: Current epoch (for video naming)
        n_steps: Number of steps to run
        seed: Random seed for environment
    
    Returns:
        dict with metrics (total_reward, distance_traveled, avg_reward, log_dir)
    """
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
        sleep_time = max(0, cfg.control.dt * 0.5 - elapsed)
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
    
    # Close environment (this saves the video)
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
    
    return {
        'total_reward': total_reward,
        'avg_reward': total_reward / max(step + 1, 1),
        'distance': distance,
        'steps': step + 1,
        'log_dir': env_log_dir,
    }


def train(args):
    """Train the transformer policy with optional robot validation."""
    DatasetClass = create_dataset_class(args.rollout_paths)
    traj_dataset = DatasetClass({"val_split": 0.1}, args.context_len)
    
    use_flow_matching = not args.no_flow_matching
    
    trainer = HFActionChunkingTrainer(
        traj_dataset,
        model_name="google/gemma-3-270m",
        log_dir=args.log_dir,
        
        # 🎯 Action Chunking settings
        action_horizon=args.action_horizon,
        execute_horizon=1,    # Execute 1 action before replanning (MPC)
        
        # LoRA settings
        use_lora=False,
        freeze_backbone=False,  # Train entire model
        
        # 🌊 Flow Matching (like pi0)
        use_flow_matching=use_flow_matching,
        flow_matching_steps=10 if use_flow_matching else 0,
        
        # Training settings
        batch_size=args.batch_size,
        learning_rate=1e-4,
        validation_freq=20,
        action_is_velocity=True,
    )
    
    # Train with periodic robot validation
    if args.robot_validation:
        epochs_trained = 0
        while epochs_trained < args.n_epochs:
            # Train for interval epochs
            epochs_to_train = min(args.robot_val_interval, args.n_epochs - epochs_trained)
            trainer.learn(n_epochs=epochs_to_train)
            epochs_trained += epochs_to_train
            
            # Run robot validation
            print(f"\n[Robot Validation] Epoch {epochs_trained}/{args.n_epochs}")
            try:
                metrics = evaluate_robot_rollout(
                    trainer, 
                    epoch=epochs_trained, 
                    n_steps=args.robot_val_steps
                )
                print(f"  Total reward: {metrics['total_reward']:.2f}")
                print(f"  Avg reward/step: {metrics['avg_reward']:.4f}")
                print(f"  Distance traveled: {metrics['distance']:.3f}m")
            except Exception as e:
                print(f"  Robot validation failed: {e}")
    else:
        trainer.learn(n_epochs=args.n_epochs)
    
    trainer.save(args.model_save_path)
    return trainer


def load_checkpoint(checkpoint_path):
    """Load a trainer from a checkpoint."""
    trainer = HFActionChunkingTrainer.from_checkpoint(checkpoint_path)
    return trainer


def run_inference(trainer):
    """Run inference with the trained policy."""
    policy = trainer.get_inference()

    cfg = ConfigRegistry.create_from_name("modular_quadruped")
    cfg.control.default_dof_pos = [0, 0, 0, 0, 0]
    env = MetaMachine(cfg)
    env.render_mode = "viewer"
    obs, _ = env.reset(seed=123)
    policy.reset()

    for step in range(100000000):
        t0 = time.time()
        assert len(obs) == 40, f"Observation length is {len(obs)}, but expected 40"
        action = policy.step({f'module{i}': obs[i*8:(i+1)*8] for i in range(5)})
        obs, reward, done, truncated, info = env.step(action)
        env.render()

        # Maintain real-time simulation speed
        elapsed = time.time() - t0
        sleep_time = max(0, env.cfg.control.dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

        if done or truncated:
            print(f"Episode ended at step {step}, reward: {reward:.3f}")
            # break

    print("Inference completed!")


if __name__ == "__main__":
    args = parse_args()
    
    if args.checkpoint:
        # Load from checkpoint and run inference only
        trainer = load_checkpoint(args.checkpoint)
    else:
        # Train from scratch
        trainer = train(args)
    
    run_inference(trainer)
