import argparse
from argparse import Namespace
import os
import pdb
import random
import csv
from datetime import datetime
import glob

import copy
import shutil
import time
import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from capyformer.model import Transformer
from capyformer.data import ModuleTrajectoryDataset, ToyDataset, ToyDatasetPositionEstimator, TrajectoryDataset #, trajectory_collate_fn
import wandb

from tqdm import trange, tqdm

from capyformer.utils import load_checkpoint, save_checkpoint



class Trainer():

    def __init__(self, 
                dataset: TrajectoryDataset,
                log_dir: str,
                use_action_tanh: bool = False,
                shared_state_embedding: bool = True,
                wandb_on: bool = False,
                load_run: str = None,
                batch_size: int = 256,
                device: str = "cuda:0",
                n_blocks: int = 12,
                h_dim: int = 384,
                n_heads: int = 6,
                drop_p: float = 0.1,
                learning_rate: float = 1e-4,
                wt_decay: float = 0.005,
                warmup_steps: int = 10000,
                seed: int = 0,
                validation_freq: int = 100,
                validation_trajectories: int = 10,
                action_is_velocity: bool = True,
                dt: float = 0.02):

        self.traj_dataset = dataset
        self.act_dim = dataset.act_dim
        self.state_token_dims = dataset.state_token_dims
        self.log_dir = log_dir
        self.context_len = dataset.context_len
        self.use_action_tanh = use_action_tanh
        self.shared_state_embedding = shared_state_embedding
        self.wandb_on = wandb_on
        self.load_run = load_run
        self.batch_size = batch_size
        self.device = device
        self.n_blocks = n_blocks
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.drop_p = drop_p
        self.learning_rate = learning_rate
        self.wt_decay = wt_decay
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.validation_freq = validation_freq
        self.validation_trajectories = validation_trajectories
        self.action_is_velocity = action_is_velocity
        self.dt = dt
        
        # Will be set after training
        self.model = None

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def get_inference(self, device: str = None):
        """
        Get an inference wrapper for easy step-by-step prediction.
        
        Must be called after learn() or after loading a trained model.
        
        Args:
            device: Device to run inference on. If None, uses trainer's device.
        
        Returns:
            TransformerInference instance
        
        Example:
            trainer = Trainer(dataset, log_dir="./logs")
            trainer.learn(n_epochs=100)
            
            inference = trainer.get_inference()
            
            for t in range(episode_length):
                state = {'position': pos, 'velocity': vel}
                prediction = inference.step(state)
        """
        if self.model is None:
            raise RuntimeError(
                "No model available. Call learn() first or load a checkpoint."
            )
        
        if device is None:
            device = self.device
        
        return self.model.get_inference(device=device)
    
    def save(self, path: str, save_torchscript: bool = True):
        """
        Save the trained model checkpoint.
        
        Args:
            path: Path to save the checkpoint (without extension).
                  Will create {path}.pt for checkpoint and {path}.jit for TorchScript.
            save_torchscript: Whether to also save a TorchScript version for deployment.
        
        Example:
            trainer.learn(n_epochs=100)
            trainer.save("./models/my_model")
            # Creates: ./models/my_model.pt and ./models/my_model.jit
        """
        if self.model is None:
            raise RuntimeError(
                "No model available. Call learn() first."
            )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        
        # Remove extension if provided
        if path.endswith(".pt") or path.endswith(".jit"):
            path = path.rsplit(".", 1)[0]
        
        checkpoint_path = f"{path}.pt"
        
        # Save full checkpoint (model + metadata)
        # Handle state_mean/state_std which can be dict or tensor
        state_mean = getattr(self.model, 'state_mean', None)
        state_std = getattr(self.model, 'state_std', None)
        
        if isinstance(state_mean, dict):
            state_mean = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in state_mean.items()}
        elif hasattr(state_mean, 'cpu'):
            state_mean = state_mean.cpu()
            
        if isinstance(state_std, dict):
            state_std = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in state_std.items()}
        elif hasattr(state_std, 'cpu'):
            state_std = state_std.cpu()
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "state_token_dims": self.state_token_dims,
                "state_token_names": getattr(self.model, 'state_token_names', None),
                "act_dim": self.act_dim,
                "n_blocks": self.n_blocks,
                "h_dim": self.h_dim,
                "context_len": self.context_len,
                "n_heads": self.n_heads,
                "drop_p": self.drop_p,
                "shared_state_embedding": self.shared_state_embedding,
                "use_action_tanh": self.use_action_tanh,
                "state_mean": state_mean,
                "state_std": state_std,
            },
            "trainer_config": {
                "device": self.device,
                "action_is_velocity": self.action_is_velocity,
                "dt": self.dt,
            }
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save TorchScript version for deployment
        if save_torchscript:
            jit_path = f"{path}.jit"
            try:
                model_cpu = copy.deepcopy(self.model).to('cpu')
                traced_script_module = torch.jit.script(model_cpu)
                traced_script_module.save(jit_path)
                print(f"TorchScript model saved to {jit_path}")
            except Exception as e:
                print(f"Warning: Could not save TorchScript model: {e}")
    
    def load(self, path: str, device: str = None):
        """
        Load a model from checkpoint.
        
        Args:
            path: Path to the checkpoint file (.pt).
            device: Device to load the model to. If None, uses trainer's device.
        
        Returns:
            self (for chaining)
        
        Example:
            trainer = Trainer(dataset, log_dir="./logs")
            trainer.load("./models/my_model.pt")
            inference = trainer.get_inference()
        """
        if device is None:
            device = self.device
        
        if not path.endswith(".pt"):
            path = f"{path}.pt"
        
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No checkpoint found at {path}")
        
        print(f"Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location=device)
        
        config = checkpoint["model_config"]
        
        # Get state stats from checkpoint or dataset
        state_mean = config.get("state_mean")
        state_std = config.get("state_std")
        if state_mean is None or state_std is None:
            state_mean, state_std = self.traj_dataset.get_state_stats()
        
        # Reconstruct the model
        self.model = Transformer(
            state_token_dims=config["state_token_dims"],
            state_token_names=config.get("state_token_names"),
            act_dim=config["act_dim"],
            n_blocks=config["n_blocks"],
            h_dim=config["h_dim"],
            context_len=config["context_len"],
            n_heads=config["n_heads"],
            drop_p=config["drop_p"],
            state_mean=state_mean,
            state_std=state_std,
            shared_state_embedding=config.get("shared_state_embedding", True),
            use_action_tanh=config.get("use_action_tanh", False),
        ).to(device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # Update trainer config if available
        trainer_config = checkpoint.get("trainer_config", {})
        if "action_is_velocity" in trainer_config:
            self.action_is_velocity = trainer_config["action_is_velocity"]
        if "dt" in trainer_config:
            self.dt = trainer_config["dt"]
        
        print(f"Model loaded successfully from {path}")
        return self
    
    @staticmethod
    def load_torchscript(path: str, device: str = "cuda:0"):
        """
        Load a TorchScript model directly (without Trainer).
        
        This is useful for deployment when you don't need the full Trainer.
        
        Args:
            path: Path to the TorchScript file (.jit).
            device: Device to load the model to.
        
        Returns:
            TorchScript model ready for inference
        
        Example:
            model = Trainer.load_torchscript("./models/my_model.jit")
            # Use directly for inference
            with torch.no_grad():
                _, action_pred, _ = model(timesteps, states, actions)
        """
        if not path.endswith(".jit"):
            path = f"{path}.jit"
        
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No TorchScript model found at {path}")
        
        print(f"Loading TorchScript model from {path}")
        model = torch.jit.load(path, map_location=device)
        model.eval()
        print(f"TorchScript model loaded successfully")
        return model

    def validate_rollout(self, model, device):
        """
        Validate the model by performing trajectory rollouts using the inference API.
        
        Uses model.get_inference() to ensure validation matches actual usage pattern.
        The inference wrapper handles:
        - Context window management
        - Missing token masking (NaN values replaced with zeros and masked in attention)
        - State normalization (val_trajectories are stored unnormalized for this purpose)
        
        For each sampled trajectory:
        1. Start from the initial state (raw/unnormalized)
        2. Feed states step-by-step through inference.step() which normalizes them
        3. Collect predictions and compute MSE against ground truth targets
        
        Note: val_trajectories are intentionally kept in raw (unnormalized) form so that
        validation properly tests the full inference pipeline including normalization,
        matching the actual deployment scenario.
        
        Returns:
            Average MSE across all validation trajectories
        """
        model.eval()
        total_mse = 0.0
        num_predictions = 0
        
        # Create inference wrapper - handles context management and NaN masking
        inference = model.get_inference(device=device, context_len=self.context_len)
        
        with torch.no_grad():
            for _ in range(self.validation_trajectories):
                # Sample a random trajectory from the dataset
                traj_idx = random.randint(0, len(self.traj_dataset.val_trajectories) - 1)
                traj = self.traj_dataset.val_trajectories[traj_idx]

                # Use dataset methods to access trajectory data (works for both flat and legacy formats)
                inputs = self.traj_dataset.get_inputs(traj)
                target_gt = self.traj_dataset.get_target(traj)
                traj_len = self.traj_dataset.get_traj_len(traj)
                
                # Check if inputs is a dictionary - model always expects dict format now
                if not isinstance(inputs, dict):
                    raise ValueError(
                        "Legacy tensor format not supported. "
                        "Use dictionary format for observations/inputs."
                    )

                traj_len = min(traj_len, 200)  # Limit length for validation speed
                    
                if traj_len < 2:
                    continue
                
                # Reset inference state for new trajectory
                inference.reset()
                
                # Predict actions step by step using inference wrapper
                predicted_actions = []
                for t in range(traj_len - 1):
                    # Build current state dict for this timestep
                    # Missing tokens (NaN values) are handled by the inference wrapper:
                    # - If a key has all NaN values, pass None to mark it as missing
                    # - If a key has partial NaN, pass the value and let inference handle it
                    current_state = {}
                    for key in inputs.keys():
                        state_value = inputs[key][t]
                        if np.all(np.isnan(state_value)):
                            # Entire token is missing - pass None
                            current_state[key] = None
                        else:
                            # Token present (may have partial NaN from dimension padding)
                            # Replace any NaN with 0 for now (dimension padding case)
                            current_state[key] = np.nan_to_num(state_value, nan=0.0)
                    
                    # inference.step() handles:
                    # - Normalizing states using stored mean/std
                    # - Managing context window
                    # - Masking missing tokens in attention
                    pred_action = inference.step(current_state, return_numpy=False)
                    predicted_actions.append(pred_action.cpu())
                
                # Compute MSE between predicted and actual targets
                if len(predicted_actions) > 0:
                    predicted_actions = torch.stack(predicted_actions)
                    actual_targets = torch.from_numpy(target_gt[:len(predicted_actions)]).float()
                    
                    # Mask out NaN values in targets (from padded target dimensions)
                    # Predictions should not have NaN since inference handles missing inputs
                    valid_mask = ~torch.isnan(actual_targets)
                    
                    if valid_mask.any():
                        pred_masked = predicted_actions[valid_mask]
                        target_masked = actual_targets[valid_mask]
                        mse = F.mse_loss(pred_masked, target_masked, reduction='sum')
                        num_valid = valid_mask.sum().item()
                        
                        total_mse += mse.item()
                        num_predictions += num_valid
                        traj_mse = mse.item() / num_valid if num_valid > 0 else 0.0
                    else:
                        # Skip this trajectory if no valid targets
                        traj_mse = 0.0
                    
                    # Plot 2D trajectories if action dimension >= 2
                    plot_position = True
                    if plot_position and valid_mask.any() and self.act_dim >= 2:
                        # Replace NaN with 0 for plotting (only affects padded dimensions)
                        pred_actions_np = np.nan_to_num(predicted_actions.numpy()[:, :2], nan=0.0)
                        actual_actions_np = np.nan_to_num(actual_targets.numpy()[:, :2], nan=0.0)
                        
                        if self.action_is_velocity:
                            # Actions are velocities - integrate to get positions starting from (0, 0)
                            pred_positions = np.zeros_like(pred_actions_np)
                            actual_positions = np.zeros_like(actual_actions_np)
                            
                            # Cumulative sum to convert velocities to positions
                            pred_positions[0] = [0, 0]  # Start at origin
                            actual_positions[0] = [0, 0]  # Start at origin
                            for i in range(1, len(pred_actions_np)):
                                pred_positions[i] = pred_positions[i-1] + pred_actions_np[i-1] * self.dt
                                actual_positions[i] = actual_positions[i-1] + actual_actions_np[i-1] * self.dt
                            
                            plt.figure(figsize=(10, 8))
                            plt.plot(actual_positions[:, 0], actual_positions[:, 1], 
                                    'b-o', label='Actual Trajectory', linewidth=2, markersize=4, alpha=0.7)
                            plt.plot(pred_positions[:, 0], pred_positions[:, 1], 
                                    'r--s', label='Predicted Trajectory', linewidth=2, markersize=4, alpha=0.7)
                            
                            # Mark start and end points
                            plt.plot(0, 0, 'go', markersize=10, label='Start')
                            plt.plot(actual_positions[-1, 0], actual_positions[-1, 1], 
                                    'bo', markersize=10, label='Actual End')
                            plt.plot(pred_positions[-1, 0], pred_positions[-1, 1], 
                                    'ro', markersize=10, label='Predicted End')
                            
                            plt.xlabel('X Position', fontsize=12)
                            plt.ylabel('Y Position', fontsize=12)
                            plt.title(f'2D Trajectory Comparison (Traj {traj_idx}, MSE: {traj_mse:.6f})', fontsize=14)
                            plt.legend(fontsize=10)
                            plt.grid(True, alpha=0.3)
                            plt.axis('equal')  # Equal aspect ratio for better visualization
                        else:
                            # Actions are positions - plot directly
                            plt.figure(figsize=(10, 8))
                            plt.plot(actual_actions_np[:, 0], actual_actions_np[:, 1], 
                                    'b-o', label='Actual Actions', linewidth=2, markersize=4, alpha=0.7)
                            plt.plot(pred_actions_np[:, 0], pred_actions_np[:, 1], 
                                    'r--s', label='Predicted Actions', linewidth=2, markersize=4, alpha=0.7)
                            
                            # Mark start and end points
                            plt.plot(actual_actions_np[0, 0], actual_actions_np[0, 1], 
                                    'go', markersize=10, label='Start')
                            plt.plot(actual_actions_np[-1, 0], actual_actions_np[-1, 1], 
                                    'ko', markersize=10, label='End')
                            
                            plt.xlabel('Action Dimension 0', fontsize=12)
                            plt.ylabel('Action Dimension 1', fontsize=12)
                            plt.title(f'2D Trajectory Comparison (Traj {traj_idx}, MSE: {traj_mse:.6f})', fontsize=14)
                            plt.legend(fontsize=10)
                            plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        
                        # Save figure
                        plot_path = os.path.join(self.log_dir, f'trajectory_validation_{traj_idx}.png')
                        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                        plt.close()
        
        model.train()
        
        if num_predictions > 0:
            avg_mse = total_mse / num_predictions
            return avg_mse
        else:
            return 0.0

    def learn(
        self,
        n_epochs: int = None,
        save_final: bool = True,
    ):
        """
        Train the model.
        
        Args:
            n_epochs: Number of training epochs. If None, automatically determined.
            save_final: Whether to save checkpoint at the end of training.
        """

        state_dim = np.sum(self.state_token_dims)
        # training and evaluation device
        device = torch.device(self.device)

        # Loggers

        log_csv_path = os.path.join(self.log_dir, "log.csv")
        csv_writer = csv.writer(open(log_csv_path, 'a', 1))
        csv_header = (["loss", "validation_mse"])
        csv_writer.writerow(csv_header)


        print("=" * 60)
        print("start time: " + datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
        print("=" * 60)
        print("device set to: " + str(device))
        print("model save path: " + self.log_dir+".pt (.jit)")
        print("log csv save path: " + log_csv_path)

        print("Loading dataset...")
        
        traj_data_loader = DataLoader(
                                self.traj_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                # collate_fn=trajectory_collate_fn  # Custom collate for state_mask support
                            )

        n_epochs = int(1e6 / len(traj_data_loader)) if n_epochs is None else n_epochs
        num_updates_per_iter = len(traj_data_loader)

        if num_updates_per_iter == 0:
            raise ValueError("The dataset is too small for the given batch size. Please reduce the batch size.")
        
        ## get state stats from dataset
        state_mean, state_std = self.traj_dataset.get_state_stats()
        
        # Get state token names from dataset if available
        state_token_names = getattr(self.traj_dataset, 'state_token_names', None)

        start_epoch = 0

        model = Transformer(
            state_token_dims=self.state_token_dims,
            state_token_names=state_token_names,
            act_dim=self.act_dim,
            n_blocks=self.n_blocks,
            h_dim=self.h_dim,
            context_len=self.context_len,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            state_mean=state_mean,
            state_std=state_std,
            shared_state_embedding=self.shared_state_embedding,
            use_action_tanh=self.use_action_tanh,
        ).to(device)

        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=self.learning_rate,
                            weight_decay=self.wt_decay,
                            betas=(0.9, 0.999)
                        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
                                optimizer,
                                lambda steps: min((steps+1)/self.warmup_steps, 1)
                            )

        if self.load_run is not None:
            # Load checkpoint if specified
            assert self.load_run.endswith(".pt"), "Only .pt run files supported"
            model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, self.load_run, device=device)

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        total_updates = 0

        inner_bar = tqdm(range(num_updates_per_iter), leave = False)

        for epoch in trange(start_epoch, n_epochs):

            log_action_losses = []
            model.train()

            for batch_data in iter(traj_data_loader):
                # Unpack batch data - now includes optional state_mask
                timesteps, states, actions, traj_mask, state_mask = batch_data

                timesteps = timesteps.to(device)    # B x T
                
                # Handle dictionary or tensor states
                if isinstance(states, dict):
                    # Dictionary format: move each state token to device
                    states = {key: value.to(device) for key, value in states.items()}
                else:
                    # Legacy tensor format
                    states = states.to(device)          # B x T x state_dim
                
                # Handle state_mask (for missing tokens)
                # state_mask can be a dict of tensors, or a list/tuple of Nones from collation
                if state_mask is not None:
                    if isinstance(state_mask, dict):
                        # Move each mask tensor to device
                        state_mask = {key: value.to(device) for key, value in state_mask.items()}
                    else:
                        pdb.set_trace()
                    
                actions = actions.to(device)        # B x T x act_dim
                # returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1) # B x T x 1
                traj_mask = traj_mask.to(device)    # B x T
                action_target = torch.clone(actions).detach().to(device)

                state_preds, action_preds, return_preds = model.forward(
                                                                timesteps=timesteps,
                                                                states=states,
                                                                actions=actions,
                                                                state_mask=state_mask
                                                            )
                # only consider non padded elements
                action_preds = action_preds.view(-1, self.act_dim)[traj_mask.view(-1,) > 0]
                action_target = action_target.view(-1, self.act_dim)[traj_mask.view(-1,) > 0]

                # Mask out NaN values in target (from padded target dimensions)
                valid_target_mask = ~torch.isnan(action_target)
                if valid_target_mask.all():
                    # No NaN values - use standard MSE loss
                    action_loss = F.mse_loss(action_preds, action_target, reduction='mean')
                else:
                    # Mask out NaN values from both predictions and targets
                    action_preds_masked = action_preds[valid_target_mask]
                    action_target_masked = action_target[valid_target_mask]
                    action_loss = F.mse_loss(action_preds_masked, action_target_masked, reduction='mean')

                optimizer.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step()

                log_action_losses.append(action_loss.detach().cpu().item())

                loss_value = action_loss.detach().cpu().item()
                if self.wandb_on:
                    wandb.log({"Loss": loss_value})
                # tqdm.write(f"Loss: {loss_value}")

                total_updates += num_updates_per_iter

                inner_bar.update(1)

            
            inner_bar.reset()

            # Perform validation rollout
            validation_mse = 0.0
            if epoch % self.validation_freq == 0:
                tqdm.write(f"Performing validation rollout at epoch {epoch}...")
                validation_mse = self.validate_rollout(model, device)
                tqdm.write(f"Validation MSE: {validation_mse:.6f}")
                if self.wandb_on:
                    wandb.log({"Validation_MSE": validation_mse})
            
            # Log average loss and validation mse for the epoch
            avg_epoch_loss = np.mean(log_action_losses) if log_action_losses else 0.0
            csv_writer.writerow([avg_epoch_loss, validation_mse])

            # DEBUG!!!!
            # if epoch % 100 == 0:
            #     save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(self.log_dir, str(epoch)+"epoch.pt"))
            #     traced_script_module = torch.jit.script(copy.deepcopy(model).to('cpu'))
            #     traced_script_module.save(os.path.join(self.log_dir, str(epoch)+"epoch.jit"))

            # save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(self.log_dir, str(epoch%10)+".pt"))
            # traced_script_module = torch.jit.script(copy.deepcopy(model).to('cpu'))
            # traced_script_module.save(os.path.join(self.log_dir, str(epoch%10)+".jit"))

        # Store the trained model for later use (e.g., get_inference())
        self.model = model

        tqdm.write("=" * 60)
        tqdm.write("finished training!")
        tqdm.write("=" * 60)
        
        # Auto-save final checkpoint
        if save_final:
            final_path = os.path.join(self.log_dir, "final")
            self.save(final_path, save_torchscript=True)

        wandb.finish()
        
        return self

def _save_conf(conf, conf_name, log_dir, notes=None):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not conf_name is None:
        # Copy the configuration file to the log directory
        shutil.copy(get_cfg_path(conf_name), log_dir)
    with open(os.path.join(log_dir, "running_config.yaml"), "w") as file:
        yaml.dump(OmegaConf.to_container(conf, resolve=True), file, default_flow_style=False)
    with open(os.path.join(log_dir, "note.txt"), 'w') as f:
        # Use explicit notes parameter if provided, otherwise fall back to conf.trainer.notes
        if notes is not None:
            f.write(notes)
        else:
            f.write(conf.trainer.notes)




class ModularDecisionTransformer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def learn(
        self,
        n_epochs: int = None,
    ):

        state_dim = np.sum(self.state_token_dims)
        # training and evaluation device
        device = torch.device(self.device)

        # Loggers

        log_csv_path = os.path.join(self.log_dir, "log.csv")
        csv_writer = csv.writer(open(log_csv_path, 'a', 1))
        csv_header = (["loss"])
        csv_writer.writerow(csv_header)


        print("=" * 60)
        print("start time: " + datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
        print("=" * 60)
        print("device set to: " + str(device))
        print("model save path: " + self.log_dir+".pt (.jit)")
        print("log csv save path: " + log_csv_path)

        print("Loading dataset...")
        
        traj_data_loader = DataLoader(
                                self.traj_dataset,
                                batch_size=self.batch_size,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True
                            )

        n_epochs = int(1e6 / len(traj_data_loader)) if n_epochs is None else n_epochs
        num_updates_per_iter = len(traj_data_loader)

        if num_updates_per_iter == 0:
            raise ValueError("The dataset is too small for the given batch size. Please reduce the batch size.")
        
        ## get state stats from dataset
        state_mean, state_std = self.traj_dataset.get_state_stats()
        
        # Get state token names from dataset if available
        state_token_names = getattr(self.traj_dataset, 'state_token_names', None)

        start_epoch = 0

        model = Transformer(
            state_token_dims=self.state_token_dims,
            state_token_names=state_token_names,
            act_dim=self.act_dim,
            n_blocks=self.n_blocks,
            h_dim=self.h_dim,
            context_len=self.context_len,
            n_heads=self.n_heads,
            drop_p=self.drop_p,
            state_mean=state_mean,
            state_std=state_std,
            shared_state_embedding=self.shared_state_embedding,
            use_action_tanh=self.use_action_tanh,
        ).to(device)

        optimizer = torch.optim.AdamW(
                            model.parameters(),
                            lr=self.learning_rate,
                            weight_decay=self.wt_decay,
                            betas=(0.9, 0.999)
                        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
                                optimizer,
                                lambda steps: min((steps+1)/self.warmup_steps, 1)
                            )

        if self.load_run is not None:
            # Load checkpoint if specified
            assert self.load_run.endswith(".pt"), "Only .pt run files supported"
            model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, self.load_run, device=device)

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        total_updates = 0

        inner_bar = tqdm(range(num_updates_per_iter), leave = False)

        num_modules = self.traj_dataset.max_num_modules

        for epoch in trange(start_epoch, n_epochs):

            log_action_losses = []
            model.train()

            for timesteps, states, actions, traj_mask, module_mask in iter(traj_data_loader):

                timesteps = timesteps.to(device)    # B x T
                states = states.to(device)          # B x T x state_dim
                actions = actions.to(device)        # B x T x act_dim
                # returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1) # B x T x 1
                traj_mask = traj_mask.to(device)    # B x T
                module_mask = module_mask.to(device)  # B x T x num_modules
                action_target = torch.clone(actions).detach().to(device)

                state_preds, action_preds, return_preds = model.forward(
                                                                timesteps=timesteps,
                                                                states=states,
                                                                actions=actions
                                                            )
                # only consider non padded elements
                action_preds = action_preds.view(-1, self.act_dim)[traj_mask.view(-1,) > 0]
                action_target = action_target.view(-1, self.act_dim)[traj_mask.view(-1,) > 0]
                module_mask = module_mask.view(-1, num_modules)[traj_mask.view(-1,) > 0]
                action_preds = action_preds * module_mask
                action_target = action_target * module_mask

                # Mask out NaN values in target (from padded target dimensions)
                valid_target_mask = ~torch.isnan(action_target)
                if valid_target_mask.all():
                    # No NaN values - use standard MSE loss
                    action_loss = F.mse_loss(action_preds, action_target, reduction='mean')
                else:
                    # Mask out NaN values from both predictions and targets
                    action_preds_masked = action_preds[valid_target_mask]
                    action_target_masked = action_target[valid_target_mask]
                    action_loss = F.mse_loss(action_preds_masked, action_target_masked, reduction='mean')

                optimizer.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step()

                log_action_losses.append(action_loss.detach().cpu().item())

                loss_value = action_loss.detach().cpu().item()
                if self.wandb_on:
                    wandb.log({"Loss": loss_value})
                tqdm.write(f"Loss: {loss_value}")

                total_updates += num_updates_per_iter

                inner_bar.update(1)

            
            inner_bar.reset()

            # Perform validation rollout
            validation_mse = 0.0
            if epoch % self.validation_freq == 0:
                tqdm.write(f"Performing validation rollout at epoch {epoch}...")
                validation_mse = self.validate_rollout(model, device)
                tqdm.write(f"Validation MSE: {validation_mse:.6f}")
                if self.wandb_on:
                    wandb.log({"Validation_MSE": validation_mse})
            
            # Log average loss and validation mse for the epoch
            avg_epoch_loss = np.mean(log_action_losses) if log_action_losses else 0.0
            csv_writer.writerow([avg_epoch_loss, validation_mse])

            if epoch % 100 == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(self.log_dir, str(epoch)+"epoch.pt"))
                traced_script_module = torch.jit.script(copy.deepcopy(model).to('cpu'))
                traced_script_module.save(os.path.join(self.log_dir, str(epoch)+"epoch.jit"))

            save_checkpoint(model, optimizer, scheduler, epoch, os.path.join(self.log_dir, str(epoch%10)+".pt"))
            traced_script_module = torch.jit.script(copy.deepcopy(model).to('cpu'))
            traced_script_module.save(os.path.join(self.log_dir, str(epoch%10)+".jit"))


        tqdm.write("=" * 60)
        tqdm.write("finished training!")
        tqdm.write("=" * 60)
        

        wandb.finish()




if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('cfg', nargs='+', default=['sim_train_tf5'])
    # args = parser.parse_args()
    # conf_name = args.cfg[0]
    # conf = load_cfg('sim_train_tf5', alg="tf")

    # Call train with explicit parameters
    dataset_path_list = glob.glob("./debug/test_dataset/*.npz")
    context_len=10
    data_cfg = {"dataset_path": dataset_path_list}
    traj_dataset = ToyDatasetPositionEstimator(data_cfg, context_len)
    
    dt = Trainer(
        traj_dataset,
        log_dir="./debug",
        use_action_tanh=False,
        shared_state_embedding=False,
        n_blocks=3,
        h_dim=256,
        n_heads=1,
        batch_size=32
    )
    dt.learn(
        n_epochs=10000,
    )
    
