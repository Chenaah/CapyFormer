import copy
from typing import Tuple
import ray
from collections import defaultdict
import glob
import multiprocessing
import os
import pdb
import random
import time
import pickle
from regex import F
import torch
import numpy as np
from torch.utils.data import Dataset
# from d4rl_infos import REF_MIN_SCORE, REF_MAX_SCORE, D4RL_DATASET_STATS
import imageio
from tqdm import tqdm, trange

def save_rollout(rollout, save_dir, file_name="rolloutN.npz"):
    # Convert (T,B,D) to (T*B, D)
    # The observation returned for the i-th environment when done[i] is true will in fact be the first observation of the next episode
    print("Processing rollout...")
    batch_size = np.shape(rollout["observations"])[1]
    rollout_reshaped = defaultdict(list)
    for b in trange(batch_size):

        # for the b-th environment
        obs = np.array(rollout["observations"])[:,b,:].tolist()
        act = np.array(rollout["actions"])[:,b,:].tolist()
        rew = np.array(rollout["rewards"])[:,b].tolist()
        done = np.array(rollout["dones"])[:,b].tolist()
        am = np.array(rollout["active_modules"])[:,b].tolist()

        # Set the first done to True (for the previous episode) and keep the last one as False
        if b != 0:
            done[0] = True
        # Remove False values from the end of the list
        while done and not done[-1]:  # Check if the list is not empty and the last element is False
            done.pop()
            obs.pop()
            act.pop()
            rew.pop()
            am.pop()
        # also delete the last step
        if done and b != batch_size-1:
            done.pop()
            obs.pop()
            act.pop()
            rew.pop()
            am.pop()

        rollout_reshaped["observations"].extend(obs)
        rollout_reshaped["actions"].extend(act)
        rollout_reshaped["rewards"].extend(rew)
        rollout_reshaped["dones"].extend(done)
        rollout_reshaped["active_modules"].extend(am)

    # np.savez_compressed(generate_unique_filename(os.path.join(save_dir, file_name)), **rollout_reshaped)
    np.savez_compressed(os.path.join(save_dir, file_name), **rollout_reshaped)



class TrajectoryDataset(Dataset):
    context_len: int
    state_token_dims: list
    act_dim: int

    def __init__(self, dataset_config, context_len):
        self.context_len = context_len

        self._setup_dataset(dataset_config)

    def _setup_dataset(self, dataset_config):
        """
        Load the dataset from the given path.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")



    def get_state_stats(self):
        # if body:
        #     return self.state_mean, self.state_std, self.body_mean, self.body_std
        assert not np.any(np.isnan(self.state_mean)), "State mean contains NaN values"
        assert not np.any(np.isnan(self.state_std)), "State std contains NaN values"
        return self.state_mean, self.state_std


    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]
        dtype = torch.float32

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len]).to(dtype)
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len]).to(dtype)
            # returns_to_go = torch.from_numpy(traj['returns_to_go'][si : si + self.context_len])
            timesteps = torch.arange(start=si, end=si+self.context_len, step=1)

            # all ones since no padding
            traj_mask = torch.ones(self.context_len, dtype=torch.long)

        else:
            padding_len = self.context_len - traj_len

            # padding with zeros
            states = torch.from_numpy(traj['observations']).to(dtype)
            states = torch.cat([states,
                                torch.zeros(([padding_len] + list(states.shape[1:])),
                                dtype=dtype)],
                               dim=0)

            actions = torch.from_numpy(traj['actions']).to(dtype)
            actions = torch.cat([actions,
                                torch.zeros(([padding_len] + list(actions.shape[1:])),
                                dtype=dtype)],
                               dim=0)

            # returns_to_go = torch.from_numpy(traj['returns_to_go'])
            # returns_to_go = torch.cat([returns_to_go,
            #                     torch.zeros(([padding_len] + list(returns_to_go.shape[1:])),
            #                     dtype=returns_to_go.dtype)],
            #                    dim=0)
            # pdb.set_trace()

            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)

        # B, T = 1, self.context_len
        return  timesteps, self._reshape_states(states), actions, traj_mask



    def _reshape_states(self, states):
        """
        Reshape the states to match the expected input shape.
        """
        return states
    

class ToyDataset(TrajectoryDataset):

    def _setup_dataset(self, dataset_config):
        # Create some toy data
        self.state_token_dims = [2, 2]
        self.act_dim = 2

        self.trajectories = []
        for _ in range(100):
            traj_len = random.randint(10, 50)
            states = np.zeros((traj_len, 2, 2))
            states[:,0,:] = 1 + 0.1*np.random.randn(traj_len, 2)
            states[:,1,:] = 2 + 0.1*np.random.randn(traj_len, 2)
            actions = np.repeat((states[:,0,0] + states[:,1,0]).reshape(-1,1), repeats=2, axis=1) 
            self.trajectories.append({
                'observations': states,
                'actions': actions,
            })

        # calculate min len of traj, state mean and variance 
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        print(f"State mean: {self.state_mean}, State std: {self.state_std}")
        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std


class ToyDatasetPositionEstimator(TrajectoryDataset):
    """
    Toy dataset for testing position estimator.
    states[:,0] is a 2D velocity vector
    states[:,1] is a 2D random vector
    actions are the positions, starting from (0,0) for each trajectory,
    obtained by integrating the velocity vectors.
    """

    def _setup_dataset(self, dataset_config):
        # Create toy data for position estimation
        self.state_token_dims = [2, 2]
        self.act_dim = 2

        self.trajectories = []
        for _ in range(100000):
            traj_len = random.randint(10, 50)
            states = np.zeros((traj_len, 2, 2))
            
            # states[:,0] is a 2D velocity vector (with some noise)
            states[:,0,:] = 0.5 + 0.2*np.random.randn(traj_len, 2)
            
            # states[:,1] is a 2D random vector
            states[:,1,:] = np.random.randn(traj_len, 2)
            
            # actions are positions obtained by integrating velocity
            # Starting from (0, 0) for each trajectory
            positions = np.zeros((traj_len, 2))
            positions[0] = np.array([0.0, 0.0])
            for t in range(1, traj_len):
                # Integrate velocity to get position (simple Euler integration)
                positions[t] = positions[t-1] + states[t-1, 0, :]
            
            self.trajectories.append({
                'observations': states,
                'actions': positions,
            })

        # calculate min len of traj, state mean and variance 
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        print(f"State mean: {self.state_mean}, State std: {self.state_std}")
        
        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std




class ModuleTrajectoryDataset(TrajectoryDataset):

    def _reshape_states(self, states):
        return states.reshape(self.context_len, self.max_num_modules, 8)


    def _setup_dataset(self, dataset_config):
        dataset_path = dataset_config['dataset_path']
        print("dataset path: " + str(dataset_path))


        self.state_token_dims = [8]*5 # 5 modules, each module has 8 state dimensions
        self.act_dim = 5

        max_state_dim = 40
        max_act_dim = 5

        self.max_num_modules = 5

        # load dataset
        assert type(dataset_path) == list
        # One embodiment per item in the list
        obs_dict = {}
        act_dict = {}
        timeouts_dict = {}
        # One embodiment per file
        for file in dataset_path:
            if file.endswith('.npz'):
                dataset_npz = np.load(file)
            else:
                raise NotImplementedError("Only npz files are supported")

            obs = dataset_npz["observations"]
            act = dataset_npz["actions"]
            timeouts=dataset_npz["dones"]

            # data_token = act.shape[1]
            if act.shape[1] == 1:
                data_token = (1)
            elif act.shape[1] == 3:
                # TODO: Other cases
                data_token = (1,1,1,0,0)
            elif act.shape[1] == 5:
                data_token = (1,1,1,1,1)
            elif act.shape[1] == 4:
                # 4D data should be paired with infomation of how the 4D data is going to be stiched together with 1D data
                if "amputatedA" in file:
                    data_token = (1,0,1,1,1)
                elif "amputatedB" in file:
                    data_token = (1,1,0,1,1)
                elif "amputatedC" in file:
                    data_token = (1,1,1,0,1)
                elif "amputatedD" in file:
                    data_token = (1,1,1,1,0)
                else:
                    print("Find general case 4-module data")
                    data_token = (1,1,1,1,0)

            if data_token in obs_dict:
                obs_dict[data_token] = np.concatenate((obs_dict[data_token], obs), axis=0)
                act_dict[data_token] = np.concatenate((act_dict[data_token], act), axis=0)
                timeouts_dict[data_token] = np.concatenate((timeouts_dict[data_token], timeouts), axis=0)
            else:
                obs_dict[data_token] = obs
                act_dict[data_token] = act
                timeouts_dict[data_token] = timeouts
            

        self.trajectories = []

        ############ Qudruped robot ############
        token = (1,1,1,1,1)
        if token in obs_dict:
            obs = obs_dict[token]
            act = act_dict[token]
            timeouts = timeouts_dict[token]

            # Each item in the list is a trajectory
            state_data = []
            action_data = []

            state_data_p = []
            action_data_p = []
            for i in trange(len(obs)):
                if timeouts[i]:
                    state_data.append(state_data_p)
                    action_data.append(action_data_p)
                    state_data_p = []
                    action_data_p = []
                # When the ith is done, the ith obs is the first obs of the next trajectory
                state_data_p.append(obs[i])
                action_data_p.append(act[i])

            for i in trange(len(state_data)):
                self.trajectories.append({
                    'observations': np.array(state_data[i]),
                    'actions': np.array(action_data[i]),
                })
        else:
            print("!!!!!!!!!!!!!   No quadruped robot trajectories found!")


        ############ Single Modules * 5 ############
        if (1) in obs_dict:
            print("Stitching together single module trajectories")
            obs1 = obs_dict[(1)]
            act1 = act_dict[(1)]
            timeouts1 = timeouts_dict[(1)]


            obs1s = [obs1]
            act1s = [act1]
            timeouts1s = [timeouts1]

            # Data augmentation
            true_indices = [i for i, value in enumerate(timeouts1) if value == True]
            n_new = 5-1
            interval = (len(true_indices) - 20) / (n_new - 1)
            split_indexs = [true_indices[int(10 + i * interval)] for i in range(n_new)]

            for split_index in split_indexs:
                print(f"Splitting at index {split_index}")
                split_index = random.choice(true_indices[:-1])
                timeout1_sub1 = timeouts1[1:split_index + 1]
                timeout1_sub2 = timeouts1[split_index + 1:]
                obs1_sub1 = obs1[:split_index + 1]
                act1_sub1 = act1[:split_index + 1]
                obs1_sub2 = obs1[split_index + 1:-1]
                act1_sub2 = act1[split_index + 1:-1]
                obs1_new = np.concatenate((obs1_sub2, obs1_sub1, obs1[-1:-1]), axis=0)
                act1_new = np.concatenate((act1_sub2, act1_sub1, act1[-1:-1]), axis=0)
                timeout1_new = np.concatenate(([False], timeout1_sub2, timeout1_sub1), axis=0)
                obs1s.append(obs1_new)
                act1s.append(act1_new)
                timeouts1s.append(timeout1_new)

            # Each item in the list is a trajectory
            state_data = []
            action_data = []

            state_data_p = []
            action_data_p = []
            for i in trange(min([len(obs1) for obs1 in obs1s])):
                if np.any([timeouts1s[j][i] for j in range(5)]):
                    state_data.append(state_data_p)
                    action_data.append(action_data_p)
                    state_data_p = []
                    action_data_p = []
                # When the ith is done, the ith obs is the first obs of the next trajectory
                state_data_p.append(np.concatenate([obs1s[j][i] for j in range(5)], axis=0))
                action_data_p.append(np.concatenate([act1s[j][i] for j in range(5)], axis=0))

            for i in trange(len(state_data)):
                self.trajectories.append({
                    'observations': np.array(state_data[i]),
                    'actions': np.array(action_data[i]),
                })



        ############ AmputatedA/B/C + Single Modules * 1 ############

        # if (1) in obs_dict:
        #     obs1 = obs_dict[(1)]
        #     act1 = act_dict[(1)]
        #     timeouts1 = timeouts_dict[(1)]

        tokens = [(1,0,1,1,1), (1,1,0,1,1), (1,1,1,0,1), (1,1,1,1,0)]
        # (1,1,1,0,1)
        for token in tokens:
            print(f"Stitching {token} and (1)")
            if token in obs_dict and (1) in obs_dict:
                print("Stitching together amputatedC and single module trajectories")
                obs4 = obs_dict[token]
                act4 = act_dict[token]
                timeouts4 = timeouts_dict[token]

                while len(obs1) < len(obs4):
                    print("Data augmentation")
                    obs1 = np.concatenate((obs1, obs1), axis=0)
                    act1 = np.concatenate((act1, act1), axis=0)
                    timeouts1 = np.concatenate((timeouts1, timeouts1), axis=0)

                state_data = []
                action_data = []

                state_data_p = []
                action_data_p = []
                for i in trange(min(len(obs1), len(obs4))):
                    if timeouts1[i] or timeouts4[i]:
                        state_data.append(state_data_p)
                        action_data.append(action_data_p)
                        state_data_p = []
                        action_data_p = []
                    # When the ith is done, the ith obs is the first obs of the next trajectory
                    module_state_dim = obs1.shape[1]

                    cut = token.index(0)
                    sticthed_obs = np.concatenate((obs4[i][:module_state_dim*cut], obs1[i], obs4[i][-module_state_dim*(4-cut):]))
                    assert sticthed_obs.shape[0] == max_state_dim, f"sticthed_obs.shape[0] = {sticthed_obs.shape[0]}"
                    assert len(sticthed_obs.shape) == 1, f"len(sticthed_obs.shape) = {len(sticthed_obs.shape)}"
                    state_data_p.append(sticthed_obs)
                    sticthed_act = np.concatenate((act4[i][:cut], act1[i], act4[i][-(4-cut):]))
                    assert sticthed_act.shape[0] == max_act_dim, f"sticthed_act.shape[0] = {sticthed_act.shape[0]}"
                    assert len(sticthed_act.shape) == 1, f"len(sticthed_act.shape) = {len(sticthed_act.shape)}"
                    action_data_p.append(sticthed_act)
                    # state_data_p.append(np.concatenate((obs4[i], obs1[i]), axis=0))
                    # action_data_p.append(np.concatenate((act4[i], act1[i]), axis=0))

                for i in trange(len(state_data)):
                    self.trajectories.append({
                        'observations': np.array(state_data[i]),
                        'actions': np.array(action_data[i]),
                    })


        ############ Tripod + Single Modules * 2 ############
        token = (1,1,1,0,0)
        if token in obs_dict and (1) in obs_dict:
            print("Stitching together single module and tripod robot trajectories")
            # Tripod robot + single modules
            obs3 = obs_dict[token]
            act3 = act_dict[token]
            timeouts3 = timeouts_dict[token]

            while len(obs1) < len(obs3):
                print("Data augmentation")
                obs1 = np.concatenate((obs1, obs1), axis=0)
                act1 = np.concatenate((act1, act1), axis=0)
                timeouts1 = np.concatenate((timeouts1, timeouts1), axis=0)

            # obs1 = obs_dict[1]
            # act1 = act_dict[1]
            # timeouts1 = timeouts_dict[1]

            # Data augmentation
            true_indices = [i for i, value in enumerate(timeouts1) if value == True]
            split_index = random.choice(true_indices[:-1])
            timeout1_sub1 = timeouts1[1:split_index + 1]
            timeout1_sub2 = timeouts1[split_index + 1:]
            obs1_sub1 = obs1[:split_index + 1]
            act1_sub1 = act1[:split_index + 1]
            obs1_sub2 = obs1[split_index + 1:-1]
            act1_sub2 = act1[split_index + 1:-1]
            obs1b = np.concatenate((obs1_sub2, obs1_sub1, obs1[-1:-1]), axis=0)
            act1b = np.concatenate((act1_sub2, act1_sub1, act1[-1:-1]), axis=0)
            timeout1b = np.concatenate(([False], timeout1_sub2, timeout1_sub1), axis=0)

            # Each item in the list is a trajectory
            state_data = []
            action_data = []

            state_data_p = []
            action_data_p = []
            for i in trange(min(len(obs1), len(obs3), len(obs1b))):
                if timeouts1[i] or timeouts3[i] or timeout1b[i]:
                    state_data.append(state_data_p)
                    action_data.append(action_data_p)
                    state_data_p = []
                    action_data_p = []
                # When the ith is done, the ith obs is the first obs of the next trajectory
                state_data_p.append(np.concatenate((obs3[i], obs1[i], obs1b[i]), axis=0))
                action_data_p.append(np.concatenate((act3[i], act1[i], act1b[i]), axis=0))

            for i in trange(len(state_data)):
                self.trajectories.append({
                    'observations': np.array(state_data[i]),
                    'actions': np.array(action_data[i]),
                })


        print(f"Loaded {len(self.trajectories)} trajectories")
            
        
        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std





class QuickDistill(TrajectoryDataset):

    '''Can only work with old policy trainer twist_controller.
    (This is for a better compatibility. For example, with magnetic docking.)
    This is a quick distill dataset for testing purposes.
    TODO: move this class to twist_controller
    '''

    


    def _reshape_states(self, states):
        return states.reshape(self.context_len, self.max_num_modules, 8)



    def _record(self, load_runs):


        from capybararl.common.env_util import make_vec_env
        from capybararl.common.vec_env import DummyVecEnv, VecEnv
        from capybararl import CrossQ


        from collections import defaultdict
        import gymnasium as gym
        from twist_controller.envs.env_sim import ZeroSim
        import wandb

        from tqdm import trange, tqdm

        from twist_controller.utils.files import get_cfg_path, load_cfg
        from twist_controller.scripts.train_sbx import load_model

        self.max_num_modules = None


        for load_run in load_runs:

            load_run_dir = os.path.dirname(load_run)

            file_name = load_run_dir.split("/")[-1] + "_rollout.npz"

            conf_name = os.path.join(load_run_dir, "running_config.yaml")
            if not os.path.exists(conf_name):
                raise FileNotFoundError(f"Configuration file {conf_name} not found. Please check the path.")
            else:
                print(f"Loading configuration from {conf_name}")


            conf = load_cfg(conf_name, alg="sbx")


            assert conf.trainer.mode == "train", "QuickDistill can only work in train mode with old policy trainer twist_controller."
            record_obs_type = "sensed_proprioception_lite"
            record_num_envs = 10
            record_steps = 1000000
            normalize_default_pos = True
            save_dir = self.log_dir

            if self.max_num_modules is None:
                self.max_num_modules = conf.agent.num_act
            else:
                assert self.max_num_modules == conf.agent.num_act, f"self.max_num_modules = {self.max_num_modules}, conf.agent.num_act = {conf.agent.num_act}. Please check the configuration."
            self.act_dim = self.max_num_modules


            num_envs = record_num_envs
            batch_steps = int(record_steps // num_envs)
            def make_env():
                return gym.wrappers.TimeLimit(
                    ZeroSim(conf), max_episode_steps=1000
                )
            model = load_model(load_run, make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv), CrossQ)
            vec_env = model.get_env()
            obs = vec_env.reset()
            constructed_obs = [vec_env.envs[i].unwrapped.brain._construct_obs(record_obs_type) for i in range(len(vec_env.envs))] if record_obs_type is not None else obs

            rollout = defaultdict(list)

            t0 = time.time()
            from rich.progress import Progress
            progress = Progress()
            progress.start()
            task = progress.add_task("[red]Recording...", total=batch_steps)
            while True:
                action, _states = model.predict(obs, deterministic=True)
                rollout["observations"].append(constructed_obs)
                # np.shape(rollout["observations"])
                act_recorded = action if not normalize_default_pos else action+np.array(conf.agent.default_dof_pos)
                rollout["actions"].append(act_recorded)
                obs, reward, done, info = vec_env.step(action)
                active_modules =  [ifo["active_modules"] for ifo in info]
                rollout["active_modules"].append(active_modules)
                constructed_obs = [vec_env.envs[i].unwrapped.brain._construct_obs(record_obs_type) for i in range(len(vec_env.envs))] if record_obs_type is not None else obs
                # print(done)
                rollout["rewards"].append(reward)
                rollout["dones"].append(done)
                # if done[0]:
                    # step_count = 0
                    # obs = vec_env.reset()
                print(f"{len(rollout['observations'])} / {batch_steps}")
                progress.update(task, advance=1)

                if time.time() - t0 > 60*30:
                    print(f"Recording trajectories... ({len(rollout['observations'])} steps)")
                    # np.savez_compressed(os.path.join(save_dir, f"rollout.npz"), **rollout)
                    save_rollout(rollout, save_dir, file_name)
                    t0 = time.time()
                if len(rollout['observations']) >= batch_steps:
                    print(f"Finished recording trajectories ({len(rollout['observations'])} steps). Saving to {save_dir}...")
                    save_rollout(rollout, save_dir, file_name)
                    print("Done!")
                    break

            progress.stop()


    def _setup_dataset(self, dataset_config):
        self.obs_dim_per_module = 8


        self.log_dir = dataset_config['log_dir']
        if not ("use_existing_rollouts" in dataset_config and dataset_config["use_existing_rollouts"]):
            load_runs = dataset_config['load_runs']
            # Legacy config only TODO: converter from old config to new config
            print("Record rollout from the following runs:")
            for load_run in load_runs:
                print(f" - {load_run}")

            self._record(load_runs)

        rollout_file_names = glob.glob(os.path.join(self.log_dir, "*.npz"))
        print(f"Found {len(rollout_file_names)} rollout files in {self.log_dir}")

        # self.state_token_dims = [self.obs_dim_per_module]*self.max_num_modules # 5 modules, each module has 8 state dimensions
        

        # max_state_dim = self.obs_dim_per_module*self.max_num_modules
        # max_act_dim = self.act_dim


        # load dataset
        obs_dict = {}
        act_dict = {}
        timeouts_dict = {}
        # One embodiment per file
        for file in rollout_file_names:
            if file.endswith('.npz'):
                dataset_npz = np.load(file)
            else:
                raise NotImplementedError("Only npz files are supported")

            obs = dataset_npz["observations"]
            act = dataset_npz["actions"]
            timeouts=dataset_npz["dones"]

            data_token = "therobot"

            if data_token in obs_dict:
                obs_dict[data_token] = np.concatenate((obs_dict[data_token], obs), axis=0)
                act_dict[data_token] = np.concatenate((act_dict[data_token], act), axis=0)
                timeouts_dict[data_token] = np.concatenate((timeouts_dict[data_token], timeouts), axis=0)
            else:
                obs_dict[data_token] = obs
                act_dict[data_token] = act
                timeouts_dict[data_token] = timeouts
            

        self.trajectories = []

        token = "therobot"
        if token in obs_dict:
            obs = obs_dict[token]
            act = act_dict[token]
            timeouts = timeouts_dict[token]

            # Each item in the list is a trajectory
            state_data = []
            action_data = []

            state_data_p = []
            action_data_p = []
            for i in trange(len(obs)):
                if timeouts[i]:
                    state_data.append(state_data_p)
                    action_data.append(action_data_p)
                    state_data_p = []
                    action_data_p = []
                # When the ith is done, the ith obs is the first obs of the next trajectory
                state_data_p.append(obs[i])
                action_data_p.append(act[i])

            for i in trange(len(state_data)):
                self.trajectories.append({
                    'observations': np.array(state_data[i]),
                    'actions': np.array(action_data[i]),
                })
        else:
            print("Something wrong!")


        self.max_num_modules = np.array(state_data[i]).shape[-1] // self.obs_dim_per_module
        self.state_token_dims = [self.obs_dim_per_module]*self.max_num_modules
        self.act_dim = self.max_num_modules
        print(f"Loaded {len(self.trajectories)} trajectories")
            
        
        # calculate min len of traj, state mean and variance
        # and returns_to_go for all traj
        min_len = 10**6
        states = []
        for traj in self.trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        print(f"State mean: {self.state_mean}, State std: {self.state_std}")
        # normalize states
        for traj in self.trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

