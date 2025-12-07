"""
RNN Trainer for baseline comparison with Decision Transformer.
"""

import csv
import os
import pdb
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from capyformer.rnn_model import RNNModel
from capyformer.data import TrajectoryDataset


class RNNTrainer:
    """
    Trainer class for RNN baseline model.
    Similar interface to the Transformer Trainer for easy comparison.
    """
    
    def __init__(self, 
                dataset: TrajectoryDataset,
                log_dir: str,
                wandb_on: bool = False,
                batch_size: int = 256,
                device: str = "cuda:0",
                h_dim: int = 256,
                n_layers: int = 2,
                rnn_type: str = 'LSTM',
                drop_p: float = 0.1,
                learning_rate: float = 1e-3,
                wt_decay: float = 0.001,
                warmup_steps: int = 10000,
                seed: int = 0,
                validation_freq: int = 100,
                validation_trajectories: int = 10,
                action_is_velocity: bool = True,
                dt: float = 0.02,
                shared_state_embedding: bool = True):
        
        self.traj_dataset = dataset
        self.act_dim = dataset.act_dim
        self.state_token_dims = dataset.state_token_dims
        self.log_dir = log_dir
        self.context_len = dataset.context_len
        self.wandb_on = wandb_on
        self.batch_size = batch_size
        self.device = device
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.drop_p = drop_p
        self.learning_rate = learning_rate
        self.wt_decay = wt_decay
        self.warmup_steps = warmup_steps
        self.seed = seed
        self.validation_freq = validation_freq
        self.validation_trajectories = validation_trajectories
        self.action_is_velocity = action_is_velocity
        self.dt = dt
        self.shared_state_embedding = shared_state_embedding
        
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
    
    def validate_rollout(self, model, device):
        """
        Validate the model by performing trajectory rollouts.
        Predicts actions step-by-step using only the history of states available.
        This mimics the real deployment scenario where the robot predicts velocity
        at each timestep using only past observations.
        """
        model.eval()
        total_mse = 0.0
        num_predictions = 0
        
        with torch.no_grad():
            # Check if validation trajectories exist
            if not hasattr(self.traj_dataset, 'val_trajectories') or len(self.traj_dataset.val_trajectories) == 0:
                print("Warning: No validation trajectories available")
                return 0.0
            
            # Sample validation trajectories
            num_val = min(self.validation_trajectories, len(self.traj_dataset.val_trajectories))
            
            for _ in range(num_val):
                # Sample a random trajectory from validation set
                traj_idx = random.randint(0, len(self.traj_dataset.val_trajectories) - 1)
                traj = self.traj_dataset.val_trajectories[traj_idx]
                states = traj['observations']
                actual_actions = traj['actions']
                
                traj_len = min(len(actual_actions), self.context_len)  # Limit length for validation speed
                
                if traj_len < 2:
                    continue
                
                # Initialize predicted actions storage
                predicted_actions = []
                
                # RNN hidden state (will be updated incrementally)
                hidden = None
                
                # Step-by-step prediction using incremental inference
                # At each timestep, we only pass the current state and reuse the hidden state
                # from the previous step, making inference O(T) instead of O(T^2)
                for t in range(traj_len):
                    # Get only the current state at timestep t
                    current_state = {}
                    for key, value in states.items():
                        # Only include state at timestep t (shape: 1 x 1 x dim)
                        current_state[key] = torch.FloatTensor(value[t:t+1]).unsqueeze(0).to(device)
                    
                    # Forward pass through RNN with current state and previous hidden
                    # - First step (t=0): hidden=None initializes the hidden state
                    # - Subsequent steps: reuse hidden from previous step for efficiency
                    action_preds, hidden = model(current_state, hidden=hidden)
                    
                    # Get prediction (only one timestep, so index 0)
                    pred_action = action_preds[0, 0].cpu().numpy()
                    predicted_actions.append(pred_action)
                
                # Convert predictions to tensor for MSE computation
                predicted_actions = np.array(predicted_actions)
                actual_actions_np = actual_actions[:traj_len]
                
                pred_actions_tensor = torch.FloatTensor(predicted_actions).to(device)
                actual_actions_tensor = torch.FloatTensor(actual_actions_np).to(device)
                
                # Compute MSE
                mse = F.mse_loss(pred_actions_tensor, actual_actions_tensor)
                total_mse += mse.item()
                num_predictions += 1
                
                # Visualize trajectories if action_is_velocity
                if self.action_is_velocity and self.act_dim >= 2:
                    pred_actions_viz = predicted_actions[:, :2]
                    actual_actions_viz = actual_actions_np[:, :2]
                    
                    # Integrate velocities to get positions
                    pred_positions = np.zeros_like(pred_actions_viz)
                    actual_positions = np.zeros_like(actual_actions_viz)
                    
                    pred_positions[0] = [0, 0]
                    actual_positions[0] = [0, 0]
                    for i in range(1, len(pred_actions_viz)):
                        pred_positions[i] = pred_positions[i-1] + pred_actions_viz[i-1] * self.dt
                        actual_positions[i] = actual_positions[i-1] + actual_actions_viz[i-1] * self.dt
                    
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
                    plt.title(f'RNN 2D Trajectory (Traj {traj_idx}, MSE: {mse.item():.6f})', fontsize=14)
                    plt.legend(fontsize=10)
                    plt.grid(True, alpha=0.3)
                    plt.axis('equal')
                    plt.tight_layout()
                    
                    # Save figure
                    plot_path = os.path.join(self.log_dir, f'rnn_trajectory_validation_{traj_idx}.png')
                    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                    plt.close()
        
        model.train()
        
        if num_predictions > 0:
            avg_mse = total_mse / num_predictions
            return avg_mse
        else:
            return 0.0
    
    def learn(self, n_epochs: int = None):
        """
        Train the RNN model.
        """
        # Training device
        device = torch.device(self.device)
        
        # Setup logging
        log_csv_path = os.path.join(self.log_dir, "rnn_log.csv")
        csv_writer = csv.writer(open(log_csv_path, 'a', 1))
        csv_header = ["epoch", "loss", "validation_mse"]
        csv_writer.writerow(csv_header)
        
        print("=" * 60)
        print("RNN Baseline Training")
        print("=" * 60)
        print("start time: " + datetime.now().strftime("%y-%m-%d-%H-%M-%S"))
        print("=" * 60)
        print(f"RNN Type: {self.rnn_type}")
        print(f"Hidden dim: {self.h_dim}")
        print(f"Num layers: {self.n_layers}")
        print("device set to: " + str(device))
        print("model save path: " + self.log_dir)
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
        
        # Get state stats from dataset
        state_mean, state_std = self.traj_dataset.get_state_stats()
        state_token_names = getattr(self.traj_dataset, 'state_token_names', None)
        
        # Create model
        model = RNNModel(
            state_token_dims=self.state_token_dims,
            state_token_names=state_token_names,
            act_dim=self.act_dim,
            h_dim=self.h_dim,
            n_layers=self.n_layers,
            rnn_type=self.rnn_type,
            drop_p=self.drop_p,
            state_mean=state_mean,
            state_std=state_std,
            shared_state_embedding=self.shared_state_embedding,
        ).to(device)
        
        # Optimizer and scheduler
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
        
        # Set random seeds
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Training loop
        total_updates = 0
        inner_bar = tqdm(range(num_updates_per_iter), leave=False)
        
        for epoch in trange(n_epochs):
            log_action_losses = []
            model.train()
            
            for timesteps, states, actions, traj_mask in iter(traj_data_loader):
                # Note: RNN doesn't use timesteps, but we keep it for compatibility with the dataset
                
                # Move data to device
                if isinstance(states, dict):
                    states = {key: value.to(device) for key, value in states.items()}
                else:
                    states = states.to(device)
                
                actions = actions.to(device)  # B x T x act_dim
                traj_mask = traj_mask.to(device)  # B x T
                action_target = torch.clone(actions).detach().to(device)
                
                # Forward pass
                action_preds, _ = model(states)
                
                # Only consider non-padded elements
                action_preds = action_preds.view(-1, self.act_dim)[traj_mask.view(-1,) > 0]
                action_target = action_target.view(-1, self.act_dim)[traj_mask.view(-1,) > 0]
                
                # Compute loss
                action_loss = F.mse_loss(action_preds, action_target, reduction='mean')
                
                # Backward pass
                optimizer.zero_grad()
                action_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()
                scheduler.step()
                
                log_action_losses.append(action_loss.detach().cpu().item())
                
                if self.wandb_on:
                    import wandb
                    wandb.log({"RNN_Loss": action_loss.detach().cpu().item()})
                
                total_updates += 1
                inner_bar.update(1)
            
            inner_bar.reset()
            
            # Validation
            validation_mse = 0.0
            if epoch % self.validation_freq == 0:
                tqdm.write(f"Performing validation rollout at epoch {epoch}...")
                validation_mse = self.validate_rollout(model, device)
                tqdm.write(f"Validation MSE: {validation_mse:.6f}")
                
                if self.wandb_on:
                    import wandb
                    wandb.log({"RNN_Validation_MSE": validation_mse})
            
            # Log average loss and validation MSE for the epoch
            avg_epoch_loss = np.mean(log_action_losses) if log_action_losses else 0.0
            csv_writer.writerow([epoch, avg_epoch_loss, validation_mse])
            
            # Save checkpoint periodically
            if epoch % 1000 == 0 and epoch > 0:
                checkpoint_path = os.path.join(self.log_dir, f"rnn_checkpoint_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
        
        # Save final model
        final_model_path = os.path.join(self.log_dir, "rnn_final_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'state_token_dims': self.state_token_dims,
                'act_dim': self.act_dim,
                'h_dim': self.h_dim,
                'n_layers': self.n_layers,
                'rnn_type': self.rnn_type,
                'drop_p': self.drop_p,
                'shared_state_embedding': self.shared_state_embedding,
            }
        }, final_model_path)
        
        tqdm.write("=" * 60)
        tqdm.write("Finished training RNN baseline!")
        tqdm.write(f"Final model saved to: {final_model_path}")
        tqdm.write("=" * 60)
        
        if self.wandb_on:
            import wandb
            wandb.finish()
