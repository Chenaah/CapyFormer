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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
import yaml

from capyformer.model import Transformer
from capyformer.data import ModuleTrajectoryDataset, ToyDataset, ToyDatasetPositionEstimator, TrajectoryDataset
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
                action_is_velocity: bool = True):

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

    def validate_rollout(self, model, device):
        """
        Validate the model by performing trajectory rollouts.
        For each sampled trajectory:
        1. Start from the initial state
        2. Predict the first action using only the first state
        3. Use the predicted action and the actual next state to predict the next action
        4. Continue rolling out predictions using actual states but predicted actions
        5. Compute the error between predicted and actual actions
        
        Returns:
            Average MSE across all validation trajectories
        """
        model.eval()
        total_mse = 0.0
        num_predictions = 0
        
        with torch.no_grad():
            for _ in range(self.validation_trajectories):
                # Sample a random trajectory from the dataset
                traj_idx = random.randint(0, len(self.traj_dataset.trajectories) - 1)
                traj = self.traj_dataset.trajectories[traj_idx]
                
                # Check if observations is a dictionary (new format) or array (legacy format)
                is_dict_format = isinstance(traj['observations'], dict)
                
                # Get trajectory length
                if is_dict_format:
                    first_key = list(traj['observations'].keys())[0]
                    traj_len = traj['observations'][first_key].shape[0]
                else:
                    traj_len = traj['observations'].shape[0]
                    
                if traj_len < 2:
                    continue
                
                actions_gt = traj['actions']
                
                actions = torch.zeros((1, traj_len, self.act_dim),
                        dtype=torch.float32, device=device)
                
                # Initialize states based on format
                if is_dict_format:
                    states = {key: torch.zeros((1, traj_len, traj['observations'][key].shape[1]),
                                               dtype=torch.float32, device=device)
                             for key in traj['observations'].keys()}
                else:
                    states = torch.zeros(1, traj_len, *self.state_token_dims,
                                        dtype=torch.float32, device=device)
                
                timesteps = torch.arange(start=0, end=traj_len, step=1)
                timesteps = timesteps.repeat(1, 1).to(device)
                
                # Predict actions step by step
                predicted_actions = []
                for t in range(traj_len - 1):
                    
                    if is_dict_format:
                        # Dictionary format: update each state token
                        for key in traj['observations'].keys():
                            running_state = traj['observations'][key][t]
                            states[key][:,t,:] = torch.tensor(running_state, device=device)
                    else:
                        # Legacy format: update single state tensor
                        running_state = traj['observations'][t]
                        states[:,t,:] = torch.tensor(running_state, device=device)

                    if t < self.context_len:
                        # Use context from start to current position
                        if is_dict_format:
                            states_slice = {key: states[key][:,:self.context_len] for key in states.keys()}
                        else:
                            states_slice = states[:,:self.context_len]
                            
                        _, act_preds, _ = model.forward(timesteps[:,:self.context_len],
                                                states_slice,
                                                actions[:,:self.context_len])

                        # Get prediction at position t
                        pred_action = act_preds[:, t].detach()
                    else:
                        if is_dict_format:
                            states_slice = {key: states[key][:,t-self.context_len+1:t+1] for key in states.keys()}
                        else:
                            states_slice = states[:,t-self.context_len+1:t+1]
                            
                        _, act_preds, _ = model.forward(timesteps[:,t-self.context_len+1:t+1],
                                            states_slice,
                                            actions[:,t-self.context_len+1:t+1])
                        # Get prediction at the last position
                        pred_action = act_preds[:, -1].detach()
                    
                    actions[:, t] = pred_action
                    
                    # Store the predicted action
                    predicted_actions.append(pred_action.squeeze(0).cpu())
                    
                
                # Compute MSE between predicted and actual actions
                if len(predicted_actions) > 0:
                    predicted_actions = torch.stack(predicted_actions)
                    actual_actions = torch.from_numpy(actions_gt[:len(predicted_actions)]).float()
                    
                    mse = F.mse_loss(predicted_actions, actual_actions, reduction='sum')
                    total_mse += mse.item()
                    num_predictions += len(predicted_actions)
                    # print(f"Trajectory MSE: {mse.item():.6f}")
                    
                    # Plot 2D trajectories if action dimension is 2
                    plot_position = True
                    if plot_position:
                        pred_actions_np = predicted_actions.numpy()[:, :2]
                        actual_actions_np = actual_actions.numpy()[:, :2]
                        
                        if self.action_is_velocity:
                            # Actions are velocities - integrate to get positions starting from (0, 0)
                            pred_positions = np.zeros_like(pred_actions_np)
                            actual_positions = np.zeros_like(actual_actions_np)
                            
                            # Cumulative sum to convert velocities to positions
                            pred_positions[0] = [0, 0]  # Start at origin
                            actual_positions[0] = [0, 0]  # Start at origin
                            for i in range(1, len(pred_actions_np)):
                                pred_positions[i] = pred_positions[i-1] + pred_actions_np[i-1]
                                actual_positions[i] = actual_positions[i-1] + actual_actions_np[i-1]
                            
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
                            plt.title(f'2D Trajectory Comparison (Traj {traj_idx}, MSE: {mse.item():.6f})', fontsize=14)
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
                            plt.title(f'2D Trajectory Comparison (Traj {traj_idx}, MSE: {mse.item():.6f})', fontsize=14)
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
    ):

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

        for epoch in trange(start_epoch, n_epochs):

            log_action_losses = []
            model.train()

            for timesteps, states, actions, traj_mask in iter(traj_data_loader):

                timesteps = timesteps.to(device)    # B x T
                
                # Handle dictionary or tensor states
                if isinstance(states, dict):
                    # Dictionary format: move each state token to device
                    states = {key: value.to(device) for key, value in states.items()}
                else:
                    # Legacy tensor format
                    states = states.to(device)          # B x T x state_dim
                    
                actions = actions.to(device)        # B x T x act_dim
                # returns_to_go = returns_to_go.to(device).unsqueeze(dim=-1) # B x T x 1
                traj_mask = traj_mask.to(device)    # B x T
                action_target = torch.clone(actions).detach().to(device)

                state_preds, action_preds, return_preds = model.forward(
                                                                timesteps=timesteps,
                                                                states=states,
                                                                actions=actions
                                                            )
                # only consider non padded elements
                action_preds = action_preds.view(-1, self.act_dim)[traj_mask.view(-1,) > 0]
                action_target = action_target.view(-1, self.act_dim)[traj_mask.view(-1,) > 0]

                action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

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


        tqdm.write("=" * 60)
        tqdm.write("finished training!")
        tqdm.write("=" * 60)
        

        wandb.finish()

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

                action_loss = F.mse_loss(action_preds, action_target, reduction='mean')

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
    
