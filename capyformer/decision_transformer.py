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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
import yaml

from capybararl.decision_transformer.model import Transformer
from capybararl.decision_transformer.data import ModuleTrajectoryDataset, TrajectoryDataset
import wandb

from tqdm import trange, tqdm

from capybararl.decision_transformer.utils import load_checkpoint, save_checkpoint
from twist_controller.utils.files import get_cfg_path, load_cfg



class DecisionTransformer():

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
                seed: int = 0):

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

        start_epoch = 0

        model = Transformer(
            state_token_dims=self.state_token_dims,
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
            model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, load_run, device=device)

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
                tqdm.write(f"Loss: {loss_value}")
                csv_writer.writerow([loss_value])

                total_updates += num_updates_per_iter

                inner_bar.update(1)

            
            inner_bar.reset()

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




class ModularDecisionTransformer(DecisionTransformer):

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

        start_epoch = 0

        model = Transformer(
            state_token_dims=self.state_token_dims,
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
                csv_writer.writerow([loss_value])

                total_updates += num_updates_per_iter

                inner_bar.update(1)

            
            inner_bar.reset()

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
    conf = load_cfg('sim_train_tf5', alg="tf")

    # Call train with explicit parameters
    dataset_path_list = glob.glob("./debug/test_dataset/*.npz")
    context_len=60
    data_cfg = {"dataset_path": dataset_path_list}
    traj_dataset = ModuleTrajectoryDataset(data_cfg, context_len)
    
    dt = DecisionTransformer(
        traj_dataset,
        log_dir="./debug",
        use_action_tanh=False,
        shared_state_embedding=True,
        n_blocks=3,
        h_dim=256,
        n_heads=1,
    )
    dt.learn(
        n_epochs=10000,
    )
    
