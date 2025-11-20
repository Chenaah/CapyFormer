import copy
from typing import Any, Dict, Tuple
import jax
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"


@ray.remote
class RemoteMetaMachine:
    """Ray remote MetaMachine environment wrapper."""
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # os.environ["JAX_PLATFORMS"] = "cpu"
    
    def __init__(self, cfg, env_id: int):
        """Initialize remote environment.
        
        Args:
            cfg: Environment configuration
            env_id: Environment ID for logging/rendering setup
        """
        # Set EGL for headless rendering
        os.environ['MUJOCO_GL'] = 'egl'
        
        # Deep copy config to avoid shared state issues
        self.cfg = copy.deepcopy(cfg)
        self.env_id = env_id
        
        # Configure logging and rendering for first environment only
        # if env_id == 0:
        #     self.cfg.simulation.render_mode = "mp4"  # Render first env for debugging
        #     self.cfg.logging.data_dir = "/home/zmb8634/Lab/metamachine-dev/logs/debug/ray_batchtest"
        #     self.cfg.logging.create_log_dir = True
        # else:
        #     self.cfg.logging.create_log_dir = False
        #     self.cfg.simulation.render_mode = "none"
        
        # Create the environment
        self.env = gym.wrappers.TimeLimit(
            ZeroSim(self.cfg), max_episode_steps=1000
        )
        
    def reset(self) -> np.ndarray:
        """Reset environment and return observation."""
        obs, _ = self.env.reset()
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step environment with given action."""
        obs, reward, done0, done1, info = self.env.step(action)
        done = done0 | done1
        
        # Auto-reset if episode is done
        if done:
            obs, _ = self.env.reset()
            
        return obs, reward, done, info
    
    def get_observation(self) -> np.ndarray:
        """Get current observation without stepping."""
        record_obs_type = "sensed_proprioception_lite"
        return self.env.env.brain._construct_obs(record_obs_type)
    
    # def get_env_info(self) -> Dict[str, Any]:
    #     """Get environment information."""
    #     return {
    #         'num_actions': self.cfg.control.num_actions,
    #         'max_episode_length': self.cfg.task.termination_conditions.max_episode_steps
    #     }


class RayVecMetaMachine():
    """Ray-based vectorized MetaMachine environment for multiprocessing."""
    
    def __init__(self, cfg, num_envs: int = 1, device: str = "cuda:0", 
                 num_cpus_per_env: float = 1.0, num_gpus_per_env: float = 0.0):
        """Initialize Ray-based vectorized environment.
        
        Args:
            cfg: Environment configuration
            num_envs: Number of parallel environments
            device: PyTorch device for tensors
            num_cpus_per_env: CPU resources per environment
            num_gpus_per_env: GPU resources per environment
        """
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, _temp_dir="/m9400/users/zmb8634/tmp/ray")
        
        self.num_envs = num_envs
        # self.device = torch.device(device)
        self.cfg = cfg
        
        # Create remote environment actors with resource allocation
        self.envs = []
        for i in range(num_envs):
            # Configure Ray actor with resource requirements
            RemoteEnvClass = RemoteMetaMachine.options(
                num_cpus=num_cpus_per_env,
                num_gpus=num_gpus_per_env
            )
            env_actor = RemoteEnvClass.remote(cfg, i)
            self.envs.append(env_actor)
        
        # Get environment info from first environment
        # env_info = ray.get(self.envs[0].get_env_info.remote())
        # self.num_actions = env_info['num_actions']
        # self.max_episode_length = env_info['max_episode_length']
        
        # Initialize environments
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset all environments and return observations."""
        # Reset all environments in parallel
        reset_futures = [env.reset.remote() for env in self.envs]
        obs_list = ray.get(reset_futures)
        
        # Stack and convert to torch tensor
        obs_np = np.stack(obs_list, axis=0)
        # obs = torch.tensor(obs_np, device=self.device, dtype=torch.float32)
        return obs_np
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step all environments with given actions."""
        # Step all environments in parallel
        step_futures = [
            self.envs[i].step.remote(actions[i]) 
            for i in range(self.num_envs)
        ]
        results = ray.get(step_futures)
        
        # Unpack results
        obs_list, reward_list, done_list, info_list = zip(*results)
        
        # Convert to torch tensors
        obs_np = np.stack(obs_list, axis=0)
        rewards_np = np.stack(reward_list, axis=0)
        dones_np = np.stack(done_list, axis=0)

        # Aggregate info
        info = {
            "observations": {
                "critic": obs_np  # Assuming critic observations are the same
            },
            "original_info": info_list
        }

        return obs_np, rewards_np, dones_np, info

    def get_observations(self) -> np.ndarray:
        """Get current observations from all environments."""
        # Get observations from all environments in parallel
        obs_futures = [env.get_observation.remote() for env in self.envs]
        obs_list = ray.get(obs_futures)
        
        # Convert to torch tensors
        obs_np = np.stack(obs_list, axis=0)
        
        return obs_np
    
    def close(self):
        """Close all remote environments and shutdown Ray."""
        # Terminate remote actors
        for env in self.envs:
            ray.kill(env)
        
        # Optionally shutdown Ray (commented out to avoid conflicts with other Ray usage)
        # ray.shutdown()




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

        # Set the first done to True (for the previous episode) and keep the last one as False
        if b != 0:
            done[0] = True
        # Remove False values from the end of the list
        while done and not done[-1]:  # Check if the list is not empty and the last element is False
            done.pop()
            obs.pop()
            act.pop()
            rew.pop()
        # also delete the last step
        if done and b != batch_size-1:
            done.pop()
            obs.pop()
            act.pop()
            rew.pop()

        rollout_reshaped["observations"].extend(obs)
        rollout_reshaped["actions"].extend(act)
        rollout_reshaped["rewards"].extend(rew)
        rollout_reshaped["dones"].extend(done)

    # np.savez_compressed(generate_unique_filename(os.path.join(save_dir, file_name)), **rollout_reshaped)
    np.savez_compressed(os.path.join(save_dir, file_name), **rollout_reshaped)

def save_rollout_am(rollout, save_dir, file_name="rolloutN.npz"):
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






class QuickDistillRay(TrajectoryDataset):

    '''Can only work with old policy trainer twist_controller.
    (This is for a better compatibility. For example, with magnetic docking.)
    This is a quick distill dataset for testing purposes.
    TODO: move this class to twist_controller
    '''

    


    def _reshape_states(self, states):
        return states.reshape(self.context_len, self.max_num_modules, 8)



    def _record(self, load_runs, record_steps=1000000):


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
            record_num_envs = 100
            
            normalize_default_pos = True
            save_dir = self.log_dir

            if self.max_num_modules is None:
                self.max_num_modules = conf.agent.num_act
            else:
                assert self.max_num_modules == conf.agent.num_act, f"self.max_num_modules = {self.max_num_modules}, conf.agent.num_act = {conf.agent.num_act}. Please check the configuration."
            self.act_dim = self.max_num_modules


            num_envs = record_num_envs
            batch_steps = int(record_steps // num_envs)
            # def make_env():
            #     return gym.wrappers.TimeLimit(
            #         ZeroSim(conf), max_episode_steps=1000
            #     )
            vec_env = RayVecMetaMachine(
                conf,
                num_envs=num_envs,
                device="cpu",
                num_cpus_per_env=0.1,
                num_gpus_per_env=0.0
            )
            model = load_model(load_run, None, CrossQ)
            obs = vec_env.reset()
            constructed_obs = vec_env.get_observations()

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

                
                constructed_obs = vec_env.get_observations()
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
            jax.clear_backends()



    def _setup_dataset(self, dataset_config):
        self.obs_dim_per_module = 8


        self.log_dir = dataset_config['log_dir']
        record_steps = dataset_config.get("record_steps", 1000000)  
        if not ("use_existing_rollouts" in dataset_config and dataset_config["use_existing_rollouts"]):
            load_runs = dataset_config['load_runs']
            # Legacy config only TODO: converter from old config to new config
            print("Record rollout from the following runs:")
            for load_run in load_runs:
                print(f" - {load_run}")

            self._record(load_runs, record_steps)
            rollout_dir = self.log_dir
        elif "rollout_dir" in dataset_config:
            print("Using existing rollouts.")
            rollout_dir = dataset_config["rollout_dir"]
        else:
            print("Using existing rollouts.")
            rollout_dir = self.log_dir
        rollout_file_names = glob.glob(os.path.join(rollout_dir, "*.npz"))
        print(f"Found {len(rollout_file_names)} rollout files in {rollout_dir}")

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
                am_dict[data_token] = am
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




class QuickDistillMaskedRay(TrajectoryDataset):

    '''Can only work with old policy trainer twist_controller.
    (This is for a better compatibility. For example, with magnetic docking.)
    This is a quick distill dataset for testing purposes.
    TODO: move this class to twist_controller
    '''

    


    def _reshape_states(self, states):
        return states.reshape(self.context_len, self.max_num_modules, 8)



    def _record(self, load_runs, record_steps=1000000):


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
            record_num_envs = 100
            
            normalize_default_pos = True
            save_dir = self.log_dir

            if self.max_num_modules is None:
                self.max_num_modules = conf.agent.num_act
            else:
                assert self.max_num_modules == conf.agent.num_act, f"self.max_num_modules = {self.max_num_modules}, conf.agent.num_act = {conf.agent.num_act}. Please check the configuration."
            self.act_dim = self.max_num_modules


            num_envs = record_num_envs
            batch_steps = int(record_steps // num_envs)
            # def make_env():
            #     return gym.wrappers.TimeLimit(
            #         ZeroSim(conf), max_episode_steps=1000
            #     )
            vec_env = RayVecMetaMachine(
                conf,
                num_envs=num_envs,
                device="cpu",
                num_cpus_per_env=0.1,
                num_gpus_per_env=0.0
            )
            model = load_model(load_run, None, CrossQ)
            obs = vec_env.reset()
            constructed_obs = vec_env.get_observations()

            rollout = defaultdict(list)
            active_modules_mask = np.ones((num_envs, self.max_num_modules))

            t0 = time.time()
            from rich.progress import Progress
            progress = Progress()
            progress.start()
            task = progress.add_task("[red]Recording...", total=batch_steps)
            while True:
                action, _states = model.predict(obs, deterministic=True)
                action = action * active_modules_mask
                rollout["observations"].append(constructed_obs)
                # np.shape(rollout["observations"])
                act_recorded = action if not normalize_default_pos else action+np.array(conf.agent.default_dof_pos)
                rollout["actions"].append(act_recorded)
                obs, reward, done, info = vec_env.step(action)
                active_modules_mask = np.zeros((num_envs, self.max_num_modules))
                for i, ifo in enumerate(info["original_info"]):
                    active_idx = ifo["active_modules"]
                    active_modules_mask[i, active_idx] = 1

                # active_modules =  [ifo["active_modules"] for ifo in info]
                rollout["active_modules"].append(active_modules_mask)

                constructed_obs = vec_env.get_observations()
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
                    save_rollout_am(rollout, save_dir, file_name)
                    t0 = time.time()
                if len(rollout['observations']) >= batch_steps:
                    print(f"Finished recording trajectories ({len(rollout['observations'])} steps). Saving to {save_dir}...")
                    save_rollout_am(rollout, save_dir, file_name)
                    print("Done!")
                    break

            progress.stop()
            jax.clear_backends()



    def _setup_dataset(self, dataset_config):
        self.obs_dim_per_module = 8


        self.log_dir = dataset_config['log_dir']
        record_steps = dataset_config.get("record_steps", 1000000)  
        if not ("use_existing_rollouts" in dataset_config and dataset_config["use_existing_rollouts"]):
            load_runs = dataset_config['load_runs']
            # Legacy config only TODO: converter from old config to new config
            print("Record rollout from the following runs:")
            for load_run in load_runs:
                print(f" - {load_run}")

            self._record(load_runs, record_steps)
            rollout_dir = self.log_dir
        elif "rollout_dir" in dataset_config:
            print("Using existing rollouts.")
            rollout_dir = dataset_config["rollout_dir"]
        else:
            print("Using existing rollouts.")
            rollout_dir = self.log_dir
        if not isinstance(rollout_dir, list):
            rollout_file_names = glob.glob(os.path.join(rollout_dir, "*.npz"))
        else:
            rollout_file_names = []
            for rd in rollout_dir:
                rollout_file_names.extend(glob.glob(os.path.join(rd, "*.npz")))
        print(f"Found {len(rollout_file_names)} rollout files in {rollout_dir}")

        # self.state_token_dims = [self.obs_dim_per_module]*self.max_num_modules # 5 modules, each module has 8 state dimensions
        

        # max_state_dim = self.obs_dim_per_module*self.max_num_modules
        # max_act_dim = self.act_dim


        # load dataset
        obs_dict = {}
        act_dict = {}
        am_dict = {}
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
            am = dataset_npz["active_modules"]

            data_token = "therobot"

            if data_token in obs_dict:
                obs_dict[data_token] = np.concatenate((obs_dict[data_token], obs), axis=0)
                act_dict[data_token] = np.concatenate((act_dict[data_token], act), axis=0)
                am_dict[data_token] = np.concatenate((am_dict[data_token], am), axis=0)
                timeouts_dict[data_token] = np.concatenate((timeouts_dict[data_token], timeouts), axis=0)
            else:
                obs_dict[data_token] = obs
                act_dict[data_token] = act
                am_dict[data_token] = am
                timeouts_dict[data_token] = timeouts
            

        self.trajectories = []

        token = "therobot"
        if token in obs_dict:
            obs = obs_dict[token]
            act = act_dict[token]
            am = am_dict[token]
            timeouts = timeouts_dict[token]

            # Each item in the list is a trajectory
            state_data = []
            action_data = []
            module_data = []

            state_data_p = []
            action_data_p = []
            module_data_p = []
            for i in trange(len(obs)):
                if timeouts[i]:
                    state_data.append(state_data_p)
                    action_data.append(action_data_p)
                    module_data.append(module_data_p)
                    state_data_p = []
                    action_data_p = []
                    module_data_p = []
                # When the ith is done, the ith obs is the first obs of the next trajectory
                state_data_p.append(obs[i])
                action_data_p.append(act[i])
                module_data_p.append(am[i])

            for i in trange(len(state_data)):
                self.trajectories.append({
                    'observations': np.array(state_data[i]),
                    'actions': np.array(action_data[i]),
                    'active_modules': np.array(module_data[i]),
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

    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]
        dtype = torch.float32

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len]).to(dtype)
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len]).to(dtype)
            active_modules = torch.from_numpy(traj['active_modules'][si : si + self.context_len]).to(torch.long)
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
            active_modules = torch.from_numpy(traj['active_modules']).to(torch.long)
            active_modules = torch.cat([active_modules,
                                torch.zeros(([padding_len] + list(active_modules.shape[1:])),
                                dtype=torch.long)],
                               dim=0)
            


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)
        # print("traj_mask shape:", traj_mask.shape)
        # print("active_modules shape:", active_modules.shape)

        # B, T = 1, self.context_len
        return  timesteps, self._reshape_states(states), actions, traj_mask, active_modules



class QuickDistillMaskedRayHybrid(TrajectoryDataset):

    '''Sim+real hybrid dataset for testing purposes.
    '''

    


    def _reshape_states(self, states):
        return states.reshape(self.context_len, self.max_num_modules, 8)



    def _record(self, load_runs, record_steps=1000000):


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
            record_num_envs = 100
            
            normalize_default_pos = True
            save_dir = self.log_dir

            if self.max_num_modules is None:
                self.max_num_modules = conf.agent.num_act
            else:
                assert self.max_num_modules == conf.agent.num_act, f"self.max_num_modules = {self.max_num_modules}, conf.agent.num_act = {conf.agent.num_act}. Please check the configuration."
            self.act_dim = self.max_num_modules


            num_envs = record_num_envs
            batch_steps = int(record_steps // num_envs)
            # def make_env():
            #     return gym.wrappers.TimeLimit(
            #         ZeroSim(conf), max_episode_steps=1000
            #     )
            vec_env = RayVecMetaMachine(
                conf,
                num_envs=num_envs,
                device="cpu",
                num_cpus_per_env=0.1,
                num_gpus_per_env=0.0
            )
            model = load_model(load_run, None, CrossQ)
            obs = vec_env.reset()
            constructed_obs = vec_env.get_observations()

            rollout = defaultdict(list)
            active_modules_mask = np.ones((num_envs, self.max_num_modules))

            t0 = time.time()
            from rich.progress import Progress
            progress = Progress()
            progress.start()
            task = progress.add_task("[red]Recording...", total=batch_steps)
            while True:
                action, _states = model.predict(obs, deterministic=True)
                action = action * active_modules_mask
                rollout["observations"].append(constructed_obs)
                # np.shape(rollout["observations"])
                act_recorded = action if not normalize_default_pos else action+np.array(conf.agent.default_dof_pos)
                rollout["actions"].append(act_recorded)
                obs, reward, done, info = vec_env.step(action)
                active_modules_mask = np.zeros((num_envs, self.max_num_modules))
                for i, ifo in enumerate(info["original_info"]):
                    active_idx = ifo["active_modules"]
                    active_modules_mask[i, active_idx] = 1

                # active_modules =  [ifo["active_modules"] for ifo in info]
                rollout["active_modules"].append(active_modules_mask)

                constructed_obs = vec_env.get_observations()
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
                    save_rollout_am(rollout, save_dir, file_name)
                    t0 = time.time()
                if len(rollout['observations']) >= batch_steps:
                    print(f"Finished recording trajectories ({len(rollout['observations'])} steps). Saving to {save_dir}...")
                    save_rollout_am(rollout, save_dir, file_name)
                    print("Done!")
                    break

            progress.stop()
            jax.clear_backends()



    def _setup_dataset(self, dataset_config):
        self.obs_dim_per_module = 8


        self.log_dir = dataset_config['log_dir']
        record_steps = dataset_config.get("record_steps", 1000000)  
        if not ("use_existing_rollouts" in dataset_config and dataset_config["use_existing_rollouts"]):
            load_runs = dataset_config['load_runs']
            # Legacy config only TODO: converter from old config to new config
            print("Record rollout from the following runs:")
            for load_run in load_runs:
                print(f" - {load_run}")

            self._record(load_runs, record_steps)
            rollout_dir = self.log_dir
        elif "rollout_dir" in dataset_config:
            print("Using existing rollouts.")
            rollout_dir = dataset_config["rollout_dir"]
        else:
            print("Using existing rollouts.")
            rollout_dir = self.log_dir
            
        if not isinstance(rollout_dir, list):
            rollout_file_names = glob.glob(os.path.join(rollout_dir, "*.npz"))
        else:
            rollout_file_names = []
            for rd in rollout_dir:
                rollout_file_names.extend(glob.glob(os.path.join(rd, "*.npz")))
        print(f"Found {len(rollout_file_names)} rollout files in {rollout_dir}")

        # self.state_token_dims = [self.obs_dim_per_module]*self.max_num_modules # 5 modules, each module has 8 state dimensions

        # self.state_token_dims = [self.obs_dim_per_module]*self.max_num_modules # 5 modules, each module has 8 state dimensions
        

        from twist_controller.utils.others import is_list_like, is_number
        # max_state_dim = self.obs_dim_per_module*self.max_num_modules
        # max_act_dim = self.act_dim
        real_traj_file = dataset_config["real_traj_file"]
        if isinstance(real_traj_file, str):
            with open(real_traj_file, 'rb') as file:
                self.real_trajectories = pickle.load(file)
        elif is_list_like(real_traj_file):
            self.real_trajectories = []
            for traj_file in real_traj_file:
                with open(traj_file, 'rb') as file:
                    traj = pickle.load(file)
                    self.real_trajectories.extend(traj)

        for traj in self.real_trajectories:
            traj['observations'] = np.array(traj['observations'])
            traj['actions'] = np.array(traj['actions'])
            traj['active_modules'] = np.array(traj['active_modules'])


        # load dataset
        obs_dict = {}
        act_dict = {}
        am_dict = {}
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
            am = dataset_npz["active_modules"]

            data_token = "therobot"

            if data_token in obs_dict:
                obs_dict[data_token] = np.concatenate((obs_dict[data_token], obs), axis=0)
                act_dict[data_token] = np.concatenate((act_dict[data_token], act), axis=0)
                am_dict[data_token] = np.concatenate((am_dict[data_token], am), axis=0)
                timeouts_dict[data_token] = np.concatenate((timeouts_dict[data_token], timeouts), axis=0)
            else:
                obs_dict[data_token] = obs
                act_dict[data_token] = act
                am_dict[data_token] = am
                timeouts_dict[data_token] = timeouts
            

        self.trajectories = []
        

        token = "therobot"
        if token in obs_dict:
            obs = obs_dict[token]
            act = act_dict[token]
            am = am_dict[token]
            timeouts = timeouts_dict[token]

            # Each item in the list is a trajectory
            state_data = []
            action_data = []
            module_data = []

            state_data_p = []
            action_data_p = []
            module_data_p = []
            for i in trange(len(obs)):
                if timeouts[i]:
                    state_data.append(state_data_p)
                    action_data.append(action_data_p)
                    module_data.append(module_data_p)
                    state_data_p = []
                    action_data_p = []
                    module_data_p = []
                # When the ith is done, the ith obs is the first obs of the next trajectory
                state_data_p.append(obs[i])
                action_data_p.append(act[i])
                module_data_p.append(am[i])

            for i in trange(len(state_data)):
                self.trajectories.append({
                    'observations': np.array(state_data[i]),
                    'actions': np.array(action_data[i]),
                    'active_modules': np.array(module_data[i]),
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
        
        for traj in self.trajectories + self.real_trajectories:
            traj_len = traj['observations'].shape[0]
            min_len = min(min_len, traj_len)
            states.append(traj['observations'])

        # used for input normalization
        states = np.concatenate(states, axis=0)
        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        print(f"State mean: {self.state_mean}, State std: {self.state_std}")
        # normalize states
        for traj in self.trajectories + self.real_trajectories:
            traj['observations'] = (traj['observations'] - self.state_mean) / self.state_std

    def __getitem__(self, idx):
        # 30% probability to sample from self.real_trajectories if available
        assert self.real_trajectories
        if random.random() < 0.1:
            # print("Sampling from real trajectories...")
            traj = self.real_trajectories[random.randint(0, len(self.real_trajectories) - 1)]
        else:
            traj = self.trajectories[idx]
        traj_len = traj['observations'].shape[0]
        dtype = torch.float32

        if traj_len >= self.context_len:
            # sample random index to slice trajectory
            si = random.randint(0, traj_len - self.context_len)

            states = torch.from_numpy(traj['observations'][si : si + self.context_len]).to(dtype)
            actions = torch.from_numpy(traj['actions'][si : si + self.context_len]).to(dtype)
            active_modules = torch.from_numpy(traj['active_modules'][si : si + self.context_len]).to(torch.long)
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
            active_modules = torch.from_numpy(traj['active_modules']).to(torch.long)
            active_modules = torch.cat([active_modules,
                                torch.zeros(([padding_len] + list(active_modules.shape[1:])),
                                dtype=torch.long)],
                               dim=0)
            


            timesteps = torch.arange(start=0, end=self.context_len, step=1)

            traj_mask = torch.cat([torch.ones(traj_len, dtype=torch.long),
                                   torch.zeros(padding_len, dtype=torch.long)],
                                  dim=0)
        # print("traj_mask shape:", traj_mask.shape)
        # print("active_modules shape:", active_modules.shape)

        # B, T = 1, self.context_len
        return  timesteps, self._reshape_states(states), actions, traj_mask, active_modules
