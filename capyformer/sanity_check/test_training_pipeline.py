#!/usr/bin/env python3
"""
Sanity Check Test Script for CapyFormer Training Pipeline.

This script tests the transformer policy training on toy datasets with known
ground truth relationships. If the model fails to learn these simple patterns,
there's likely a bug in the training pipeline.

Tests:
1. Linear relationship: action = W @ obs + b
2. Modular structure: action[i] = W[i] @ module[i] + b[i] (matches robot data format)

Usage:
    # Quick test (recommended for debugging)
    python -m capyformer.sanity_check.test_training_pipeline --quick
    
    # Full test
    python -m capyformer.sanity_check.test_training_pipeline
    
    # Test specific model configuration
    python -m capyformer.sanity_check.test_training_pipeline --model-name google/gemma-3-270m
    
    # Test without flow matching
    python -m capyformer.sanity_check.test_training_pipeline --no-flow-matching

Copyright 2025 Chen Yu
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
SCRIPT_DIR = Path(__file__).parent.resolve()
CAPYFORMER_ROOT = SCRIPT_DIR.parent.parent
if str(CAPYFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(CAPYFORMER_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sanity check for CapyFormer training pipeline"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick test with fewer epochs and trajectories"
    )
    parser.add_argument(
        "--test-linear", action="store_true",
        help="Only test ToyLinearDataset"
    )
    parser.add_argument(
        "--test-modular", action="store_true",
        help="Only test ToyModularDataset (matches robot format)"
    )
    parser.add_argument(
        "--model-name", type=str, default="google/gemma-3-270m",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--no-flow-matching", action="store_true",
        help="Disable flow matching"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=None,
        help="Number of training epochs (default: 500 for quick, 2000 for full)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--context-len", type=int, default=20,
        help="Context length for transformer"
    )
    parser.add_argument(
        "--action-horizon", type=int, default=4,
        help="Action horizon for action chunking"
    )
    parser.add_argument(
        "--num-trajectories", type=int, default=None,
        help="Number of trajectories (default: 100 for quick, 1000 for full)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./sanity_check_logs",
        help="Directory for logs"
    )
    parser.add_argument(
        "--inference-steps", type=int, default=100,
        help="Number of inference steps to test"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def test_linear_dataset(args):
    """Test training on ToyLinearDataset."""
    from capyformer import HFActionChunkingTrainer
    from capyformer.sanity_check import ToyLinearDataset, ToyEnvironment
    
    print("\n" + "=" * 70)
    print("TEST 1: ToyLinearDataset (action = W @ obs + b)")
    print("=" * 70)
    
    # Dataset configuration
    num_trajectories = args.num_trajectories or (100 if args.quick else 1000)
    n_epochs = args.n_epochs or (200 if args.quick else 1000)
    
    config = {
        'num_trajectories': num_trajectories,
        'traj_len': 100,
        'obs_dim': 4,
        'act_dim': 2,
        'noise_std': 0.01,
        'val_split': 0.1,
    }
    
    print(f"\n[1/4] Creating dataset...")
    dataset = ToyLinearDataset(config, context_len=args.context_len)
    
    print(f"\nDataset info:")
    print(f"  Input token names: {dataset.input_token_names}")
    print(f"  Input token dims: {dataset.input_token_dims}")
    print(f"  Target dim: {dataset.target_dim}")
    print(f"  Num trajectories: {len(dataset)}")
    
    # Create trainer
    print(f"\n[2/4] Creating trainer...")
    log_dir = os.path.join(args.log_dir, "linear_test")
    os.makedirs(log_dir, exist_ok=True)
    
    trainer = HFActionChunkingTrainer(
        dataset,
        model_name=args.model_name,
        log_dir=log_dir,
        action_horizon=args.action_horizon,
        execute_horizon=1,
        use_lora=False,
        freeze_backbone=False,
        use_flow_matching=not args.no_flow_matching,
        flow_matching_steps=10,
        batch_size=args.batch_size,
        learning_rate=1e-4,
        validation_freq=20,
        action_is_velocity=True,
    )
    
    # Train
    print(f"\n[3/4] Training for {n_epochs} epochs...")
    trainer.learn(n_epochs=n_epochs)
    
    # Test inference
    print(f"\n[4/4] Testing inference...")
    policy = trainer.get_inference()
    gt_fn = dataset.get_ground_truth_fn()
    
    # Create environment with ground truth
    env = ToyEnvironment(
        obs_dim=config['obs_dim'],
        act_dim=config['act_dim'],
        ground_truth_fn=gt_fn,
    )
    
    obs, _ = env.reset(seed=42)
    policy.reset()
    
    errors = []
    for step in range(args.inference_steps):
        # Get action from policy
        predicted_action = policy.step({'observation': obs})
        
        # Get ground truth action
        gt_action = gt_fn(obs)
        
        # Compute error
        error = np.linalg.norm(predicted_action - gt_action)
        errors.append(error)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(predicted_action)
        
        if args.verbose and step % 20 == 0:
            print(f"  Step {step}: pred={predicted_action}, gt={gt_action}, error={error:.4f}")
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    
    print(f"\n[RESULTS] ToyLinearDataset:")
    print(f"  Mean action error: {mean_error:.4f} ± {std_error:.4f}")
    print(f"  Max action error: {max_error:.4f}")
    
    # Determine pass/fail
    # For a simple linear relationship, error should be < 0.5 after training
    passed = mean_error < 0.5
    
    if passed:
        print(f"  ✅ PASSED - Model learned the linear relationship!")
    else:
        print(f"  ❌ FAILED - Model did not learn the relationship (error too high)")
    
    return passed, mean_error


def test_modular_dataset(args):
    """Test training on ToyModularDataset (matches robot format)."""
    from capyformer import HFActionChunkingTrainer
    from capyformer.sanity_check import ToyModularDataset, ToyModularEnvironment
    
    print("\n" + "=" * 70)
    print("TEST 2: ToyModularDataset (5 modules × 8 dims → 5 actions)")
    print("=" * 70)
    print("This matches the format used in transformer_policy_multiple_rollouts.py")
    
    # Dataset configuration
    num_trajectories = args.num_trajectories or (100 if args.quick else 1000)
    n_epochs = args.n_epochs or (200 if args.quick else 1000)
    
    config = {
        'num_trajectories': num_trajectories,
        'traj_len': 100,
        'num_modules': 5,
        'obs_per_module': 8,
        'noise_std': 0.01,
        'val_split': 0.1,
    }
    
    print(f"\n[1/4] Creating dataset...")
    dataset = ToyModularDataset(config, context_len=args.context_len)
    
    print(f"\nDataset info:")
    print(f"  Input token names: {dataset.input_token_names}")
    print(f"  Input token dims: {dataset.input_token_dims}")
    print(f"  Target dim: {dataset.target_dim}")
    print(f"  Num trajectories: {len(dataset)}")
    
    # Create trainer (same configuration as transformer_policy_multiple_rollouts.py)
    print(f"\n[2/4] Creating trainer...")
    log_dir = os.path.join(args.log_dir, "modular_test")
    os.makedirs(log_dir, exist_ok=True)
    
    trainer = HFActionChunkingTrainer(
        dataset,
        model_name=args.model_name,
        log_dir=log_dir,
        
        # Match transformer_policy_multiple_rollouts.py settings
        action_horizon=args.action_horizon,  # Same as script
        execute_horizon=1,
        
        use_lora=False,
        freeze_backbone=False,
        
        use_flow_matching=not args.no_flow_matching,
        flow_matching_steps=10,
        
        batch_size=args.batch_size,
        learning_rate=1e-4,
        validation_freq=20,
        action_is_velocity=True,
    )
    
    # Train
    print(f"\n[3/4] Training for {n_epochs} epochs...")
    trainer.learn(n_epochs=n_epochs)
    
    # Test inference
    print(f"\n[4/4] Testing inference...")
    policy = trainer.get_inference()
    gt_fn = dataset.get_ground_truth_fn()
    
    # Create environment with ground truth
    env = ToyModularEnvironment(
        num_modules=config['num_modules'],
        obs_per_module=config['obs_per_module'],
        ground_truth_fn=gt_fn,
        flat_observation=False,  # Return dict observation
    )
    
    obs, _ = env.reset(seed=42)
    policy.reset()
    
    errors = []
    for step in range(args.inference_steps):
        # Get action from policy (pass dict observation)
        predicted_action = policy.step(obs)
        
        # Get ground truth action
        gt_action = gt_fn(obs)
        
        # Compute error
        error = np.linalg.norm(predicted_action - gt_action)
        errors.append(error)
        
        # Step environment
        obs, reward, done, truncated, info = env.step(predicted_action)
        
        if args.verbose and step % 20 == 0:
            print(f"  Step {step}: pred={predicted_action[:2]}..., gt={gt_action[:2]}..., error={error:.4f}")
    
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    max_error = np.max(errors)
    
    print(f"\n[RESULTS] ToyModularDataset:")
    print(f"  Mean action error: {mean_error:.4f} ± {std_error:.4f}")
    print(f"  Max action error: {max_error:.4f}")
    
    # Determine pass/fail
    passed = mean_error < 0.5
    
    if passed:
        print(f"  ✅ PASSED - Model learned the modular relationship!")
    else:
        print(f"  ❌ FAILED - Model did not learn the relationship (error too high)")
    
    return passed, mean_error


def test_data_only(args):
    """Test dataset creation without training (for quick debugging)."""
    from capyformer.sanity_check import ToyLinearDataset, ToyModularDataset
    
    print("\n" + "=" * 70)
    print("DATA-ONLY TEST: Verifying dataset structure")
    print("=" * 70)
    
    # Test ToyLinearDataset
    print("\n[ToyLinearDataset]")
    config = {'num_trajectories': 10, 'traj_len': 50}
    dataset = ToyLinearDataset(config, context_len=20)
    
    print(f"  Input names: {dataset.input_token_names}")
    print(f"  Input dims: {dataset.input_token_dims}")
    print(f"  Target dim: {dataset.target_dim}")
    
    # Get a batch
    batch = dataset[0]
    timesteps, states, target, traj_mask, state_mask = batch
    print(f"  Batch shapes:")
    print(f"    timesteps: {timesteps.shape}")
    print(f"    states['observation']: {states['observation'].shape}")
    print(f"    target: {target.shape}")
    print(f"    traj_mask: {traj_mask.shape}")
    
    # Test ToyModularDataset
    print("\n[ToyModularDataset]")
    config = {'num_trajectories': 10, 'traj_len': 50}
    dataset = ToyModularDataset(config, context_len=20)
    
    print(f"  Input names: {dataset.input_token_names}")
    print(f"  Input dims: {dataset.input_token_dims}")
    print(f"  Target dim: {dataset.target_dim}")
    
    # Get a batch
    batch = dataset[0]
    timesteps, states, target, traj_mask, state_mask = batch
    print(f"  Batch shapes:")
    print(f"    timesteps: {timesteps.shape}")
    for name in dataset.input_token_names:
        print(f"    states['{name}']: {states[name].shape}")
    print(f"    target: {target.shape}")
    print(f"    traj_mask: {traj_mask.shape}")
    
    print("\n✅ Dataset structure verification passed!")
    return True, 0.0


def main():
    args = parse_args()
    
    print("=" * 70)
    print("CapyFormer Sanity Check - Training Pipeline Test")
    print("=" * 70)
    print(f"Model: {args.model_name}")
    print(f"Flow Matching: {'Disabled' if args.no_flow_matching else 'Enabled'}")
    print(f"Quick mode: {args.quick}")
    print(f"Context length: {args.context_len}")
    print(f"Action horizon: {args.action_horizon}")
    
    results = {}
    
    # Determine which tests to run
    run_linear = args.test_linear or (not args.test_linear and not args.test_modular)
    run_modular = args.test_modular or (not args.test_linear and not args.test_modular)
    
    try:
        if run_linear:
            passed, error = test_linear_dataset(args)
            results['linear'] = {'passed': passed, 'error': error}
        
        if run_modular:
            passed, error = test_modular_dataset(args)
            results['modular'] = {'passed': passed, 'error': error}
    
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        return
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"  {test_name}: {status} (error: {result['error']:.4f})")
        if not result['passed']:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All sanity checks passed!")
        print("The training pipeline is working correctly on toy data.")
        print("\nIf your robot policy still doesn't work, the issue is likely:")
        print("  1. Data quality/format issues in your rollout data")
        print("  2. More complex relationship in real data requires more training")
        print("  3. Action space scaling or normalization issues")
    else:
        print("\n⚠️ Some sanity checks failed!")
        print("There may be issues with the training pipeline.")
        print("\nDebugging suggestions:")
        print("  1. Check if loss is decreasing during training")
        print("  2. Try disabling flow matching (--no-flow-matching)")
        print("  3. Try a different/smaller model")
        print("  4. Increase training epochs")


if __name__ == "__main__":
    main()

