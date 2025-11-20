# Get rid of MJX

import math
import pickle
import os
import pdb
import shutil
import time
import h5py
import numpy as np
import ray
from tqdm import trange, tqdm
from twist_controller import CONTROLLER_ROOT_DIR
from twist_controller.envs.env_sim import ZeroSim
from twist_controller.self_modeling.data_collection.utils import pipeidx_decode
from twist_controller.sim.evolution.utils import update_cfg_with_pipeline
from twist_controller.sim.evolution.vae.gen_dataset import gen_asym4x
from twist_controller.utils.files import load_cfg

# def _get_state(env):
#     quats = env.sensors["quat"].flatten()
#     ang_vel = env.sensors["gyro"].flatten()
#     dof_pos = env.data.qpos[7:]
#     dof_vel = env.data.qvel[6:]
#     state = np.concatenate([quats, ang_vel, dof_pos, dof_vel])
#     return state

def one_hot_encode(labels, flatten=True, num_classes=None):
    if num_classes is None:
        num_classes = np.max(labels) + 1
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    if flatten:
        return one_hot.flatten()
    return one_hot


robot_cfg = {
        "theta": 0.4625123,
        "R": 0.07,
        "r": 0.03,
        "l_": 0.236,
        "delta_l": 0,
        "stick_ball_l": 0.005,
        "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
        "stick_mass": 0.231, #0.26,
        "top_hemi_mass": 0.74,
        "bottom_hemi_mass": 0.534
    }

air_cfg = {
        "theta": 0.4625123,
        "R": 0.07,
        "r": 0.03,
        "l_": 0.236,
        "delta_l": 0,
        "stick_ball_l": 0.005,
        "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
        "stick_mass":0.154, # 0.231/2, #0.26,
        "top_hemi_mass": 0.106, #0.143, #0.49,# 0.74,
        "battery_mass": 0.122,
        "motor_mass": 0.317,
        "bottom_hemi_mass": 0.097, #0.132, #0.534
        "pcb_mass": 0.1
    }

mesh_dict_air = {
                "up": "top_lid.obj",
                "bottom": "bottom_lid.obj",
                "stick": "leg4.4.obj",
                "battery": "battery.obj",
                "pcb": "pcb.obj",
                "motor": "motor.obj"
            }

# Tese are important for determining the dynamics of the robot in data collection

num_modules = 5

cfg_table = {}
ctr = 1 # 0: empty pipeline
for parent_id in range(num_modules):
    for posa in range(18):
        for posb in range(9):
            for r in range(3):
                pipeline = [parent_id, posa, posb, r]
                cfg_table[tuple(pipeline)] = ctr
                ctr += 1
print(f"Total types of pipelines: {ctr}")

def pipeidx_encode(pipeline):

    pipeidx = []
    for step in np.reshape(pipeline, (-1, 4)):
        pipeidx.append(cfg_table[tuple(step)])
    if len(pipeidx) < num_modules-1:
        pipeidx += [0] * (num_modules-1 - len(pipeidx))
    return np.array(pipeidx)
        
@ray.remote
def rollout(designs, n_episodes=5000, episode_length=1000, input_tree=False, state_func=None):

    tape = {"states": [], "actions": [], "tree_nodes": []}
    # (B, T, S) and (B, T, A)
    legacy_tape = []

    # for i_p, pipeline in enumerate(pipelines):
    for i_p, design in enumerate(tqdm(designs, desc="Rollout")):

        if not input_tree:
            pipeline = design
        else:
            pipeline = pipeidx_decode(design).tolist()

        # pipeline_oh = one_hot_encode(pipeline, num_classes=16)
        conf = load_cfg("sim_train_quadrupedX4air", alg="sbx")
        conf.agent.obs_version = "sensed_proprioception"
        conf = update_cfg_with_pipeline(conf, pipeline, init_pose_type="original", robot_cfg=air_cfg, mesh_dict=mesh_dict_air)
        conf.agent.reward_version = "alive"
        conf.sim.fully_randomize_orientation = True
        conf.sim.init_pos = [0, 0, 2]
        conf.sim.noisy_actions = False
        conf.sim.noisy_observations = False 
        num_act = conf.agent.num_act
        conf.agent.default_dof_pos = [0]*num_act
        # Use the env state space
        assert state_func is None, "Debugging"
        conf.agent.include_history_steps = 1 

        n_act = conf.agent.num_act
        env = ZeroSim(conf)

        for i_ep in range(n_episodes):
            env.reset()
            # Drop the robot to the ground
            last_pos = np.array([i for i in conf.sim.init_pos])
            for _ in range(100):
                env.step(np.zeros(n_act))
                if np.linalg.norm(env.data.qpos[:3] - last_pos) < 1e-3:
                    break
                last_pos = np.array([i for i in env.data.qpos[:3]])

            # Perform the sampled sine wave
            amplitudes = np.random.uniform(-np.pi/np.pi,1, n_act)
            frequencies = np.random.uniform(0.2, 2, n_act)
            phases = np.random.uniform(0, 2*np.pi, n_act)

            state_rollout = []
            action_rollout = []

            state = env.brain.get_observations(insert=False)
            for t in range(episode_length):
                state_rollout.append(np.concatenate([state, np.zeros(num_modules*9 - len(state))]))
                action = amplitudes * np.sin(frequencies * t + phases)
                env.step(action)
                action_rollout.append(np.concatenate([action, np.zeros(num_modules-num_act)]))
                state = env.brain.get_observations(insert=False)
                assert state.shape[0]%9 == 0, f"State shape: {state.shape}"

            tape["states"].append(np.array(state_rollout))
            tape["actions"].append(np.array(action_rollout))
            tape["tree_nodes"].append(np.array([pipeidx_encode(pipeline)]*episode_length))

            # legacy_tape.append({"states": np.array(state_rollout), 
            #                     "actions": np.array(action_rollout),
            #                     # "pipeline_onehot": np.array([pipeline_oh]*episode_length),
            #                     "pipeline": np.array([pipeline]*episode_length),
            #                     "tree_nodes": np.array([pipeidx_encode(pipeline)]*episode_length)
            #                     })
    tape["states"] = np.array(tape["states"], dtype=np.float32)
    tape["actions"] = np.array(tape["actions"], dtype=np.float32)
    tape["tree_nodes"] = np.array(tape["tree_nodes"], dtype=np.int16)
    return tape
        

def gen_training_data():

    temp_dir = "/m9400/users/zmb8634/tmp/ray" 
    temp_dir = temp_dir if os.path.exists(temp_dir) else "/tmp/ray"

    p_file = os.path.join(CONTROLLER_ROOT_DIR, "twist_controller/sim/evolution/vae/designs_asym.pkl")
    with open(p_file, "rb") as f:
        pipelines_loded = pickle.load(f)

    pipelines = [[ 0,1,0,0, 0,3,0,0, 0,13,0,0, 0,15,0,0 ],
                 [ 0,0,0,2 ],
                 [ 0,1,0,0, 0,3,0,0 ],
                 [ 0,1,0,0, 0,3,0,0, 0,17,0,2 ],
                 []]
    pipelines += pipelines_loded
    
    

    pipe_offset = 0
    n_episodes_per_pipe = 100 # unchanged
    n_pipes_per_worker = 20  # how long to run each worker
    n_workers = 50
    n_epochs = 10
    
    tapes = []
    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch}")
        ray.init(_temp_dir=temp_dir)
        pipe_offset = epoch * n_pipes_per_worker * n_workers
        rollout_tasks = [rollout.remote(pipelines[(i)*n_pipes_per_worker+pipe_offset:(i+1)*n_pipes_per_worker+pipe_offset], n_episodes=n_episodes_per_pipe, episode_length=1000) for i in range(n_workers)]
        new_tapes = ray.get(rollout_tasks)
        print("Concatenating tapes...")
        tapes += new_tapes
        saved_tapes = {key: np.concatenate([d[key] for d in tapes]) for key in tapes[0]}
        print("Saving tapes...")
        data_name = f"mini{int(n_pipes_per_worker*n_workers*(epoch+1))}_legacy"
        np.savez_compressed(os.path.join(CONTROLLER_ROOT_DIR, "twist_controller", "self_modeling",  f"{data_name}.npz"), **saved_tapes)
        ray.shutdown()
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)



def gen_ood_testing_data():

    temp_dir = "/m9400/users/zmb8634/tmp/ray" 
    temp_dir = temp_dir if os.path.exists(temp_dir) else "/tmp/ray"

    print("Loading training data...")
    p_file = os.path.join(CONTROLLER_ROOT_DIR, "twist_controller/sim/evolution/vae/designs_asym.pkl")
    with open(p_file, "rb") as f:
        pipelines_loded = pickle.load(f)

    existed_pipelines = [[ 0,1,0,0, 0,3,0,0, 0,13,0,0, 0,15,0,0 ],
                        [ 0,0,0,2 ],
                        [ 0,1,0,0, 0,3,0,0 ],
                        [ 0,1,0,0, 0,3,0,0, 0,17,0,2 ],
                        []]
    existed_pipelines += pipelines_loded

    print("Loading OOD testing data...")

    p_file = os.path.join(CONTROLLER_ROOT_DIR, "twist_controller/sim/evolution/vae/designs_asym_v2.pkl")
    with open(p_file, "rb") as f:
        test_data = pickle.load(f)

    print("Filtering out existed pipelines...")
    pipelines = [p for p in test_data[:2000] if p not in existed_pipelines]
    print(f"Number of OOD pipelines: {len(pipelines)}")
    
    

    pipe_offset = 0
    n_episodes_per_pipe = 100 # unchanged
    n_pipes_per_worker = 10  # how long to run each worker
    n_workers = 50
    n_epochs = 2
    
    tapes = []
    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch}")
        ray.init(_temp_dir=temp_dir)
        pipe_offset = epoch * n_pipes_per_worker * n_workers
        rollout_tasks = [rollout.remote(pipelines[(i)*n_pipes_per_worker+pipe_offset:(i+1)*n_pipes_per_worker+pipe_offset], n_episodes=n_episodes_per_pipe, episode_length=1000) for i in range(n_workers)]
        new_tapes = ray.get(rollout_tasks)
        print("Concatenating tapes...")
        tapes += new_tapes
        saved_tapes = {key: np.concatenate([d[key] for d in tapes]) for key in tapes[0]}
        print("Saving tapes...")
        data_name = f"mini{int(n_pipes_per_worker*n_workers*(epoch+1))}_legacy"
        np.savez_compressed(os.path.join(CONTROLLER_ROOT_DIR, "twist_controller", "self_modeling",  f"{data_name}.npz"), **saved_tapes)
        ray.shutdown()
        shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

import concurrent.futures

@ray.remote
def gen_job(n_designs_per_worker, min_n_modules, max_n_modules, seed):
    np.random.seed(seed)  
    sub_pop = []
    while len(sub_pop) < n_designs_per_worker:
        design = gen_asym4x(min_n_modules, max_n_modules)
        tree = pipeidx_encode(design)
        # if tree not in sub_pop:
        sub_pop.append(tree)
        print(f"[{len(sub_pop)}/{n_designs_per_worker}] New design added: ", tree)
    return sub_pop

def fast_gen_dateset(min_n_modules=1, max_n_modules=3, data_name="designs_short", seed_offset=0, max_num_designs = 5e5, n_workers = 300):



    n_designs_per_worker = math.ceil(max_num_designs*1.5 / n_workers)


    remotes = [gen_job.remote(n_designs_per_worker, min_n_modules, max_n_modules, seed+seed_offset) for seed in range(n_workers)]
    designs_sub_pop = ray.get(remotes)
    all_designs = []
    for designs in tqdm(designs_sub_pop):
        all_designs.extend(designs)
   
    print("Adding hand-designed pipelines...")
    hand_designed_pipelines = [[ 0,1,0,0, 0,3,0,0, 0,13,0,0, 0,15,0,0 ],
                                [ 0,0,0,2 ],
                                [ 0,1,0,0, 0,3,0,0 ],
                                [ 0,1,0,0, 0,3,0,0, 0,17,0,2 ],
                                [],
                                [ 0,3,1,0, 1,2,4,0, 1,3,1,0, 1,4,2,0 ],
                                [ 0,1,0,0, 0,3,0,0, 2,17,3,0, 1,17,1,0 ],
                                [ 0,1,0,1, 0,2,0,2, 0,14,0,1, 0,13,0,2 ],
                                [ 0,9,0,2, 0,10,0,1, 0,13,0,0, 0,15,0,0 ],
                                [ 0,13,1,0, 1,14,1,0 ],
                                [ 0,9,0,2, 0,10,0,1 ],
                                [ 0,1,0,1, 0,2,0,2, 0,15,0,0 ],
                                [ 0,1,0,1, 0,2,0,2 ],
                                ]
    hand_designed_trees = [pipeidx_encode(p) for p in hand_designed_pipelines]
    all_designs.extend(hand_designed_trees)

    print("Removing duplicates...")
    all_designs_np = np.array(all_designs, dtype=np.int16)
    # Use numpy's unique function, this works for 1D arrays and axis=0 for 2D
    unique_designs = np.unique(all_designs_np, axis=0)


    print(f"Total designs: {len(all_designs)}, Unique designs: {len(unique_designs)}")

    if len(unique_designs) > max_num_designs:
        print(f"Truncating to {max_num_designs} designs")
        unique_designs = unique_designs[-max_num_designs:]

    design_file = os.path.join(CONTROLLER_ROOT_DIR, f"twist_controller/self_modeling/data_collection/{data_name}.npy")
    print(f"Saving designs to {design_file}")
    np.save(design_file, unique_designs)
    ray.shutdown()
    return unique_designs

# def get_state(env):
#     state = []
#     quats = env.sensors["quat"].flatten()
#     ang_vel = env.sensors["gyro"].flatten()
#     dof_pos = env.data.qpos[7:]
#     dof_vel = env.data.qvel[6:]
#     n_act = len(dof_vel)
#     for i in range(n_act):
#         state += list(quats[i*4:(i+1)*4])
#         state += list(ang_vel[i*3:(i+1)*3])
#         state.append(dof_pos[i])
#         state.append(dof_vel[i])
#     assert len(state) == n_act*9
#     return np.array(state)


def gen_bigger_training_data():
    tree_data_name = "designtrees_1m"
    tree_data_file = os.path.join(CONTROLLER_ROOT_DIR, f"twist_controller/self_modeling/data_collection/{tree_data_name}.npy")
    if os.path.exists(tree_data_file):
        print("Loading existing tree data...")
        designs = np.load(tree_data_file)
    else:
        designs = fast_gen_dateset(1,4,data_name=tree_data_name, seed_offset=999, max_num_designs=1e6, n_workers=1000)
    # designs = fast_gen_dateset(1,4,data_name="test", seed_offset=999, max_num_designs=100, n_workers=50)

    temp_dir = "/m9400/users/zmb8634/tmp/ray" 
    temp_dir = temp_dir if os.path.exists(temp_dir) else "/tmp/ray"


    pipe_offset = 0
    n_episodes_per_pipe = 10 # make it faster?
    episode_length = 200 # make it faster?
    # n_pipes_per_worker = 20  # how long to run each worker
    n_workers = 300
    # n_epochs = 10


    n_pipes_per_worker = len(designs) // n_workers
    print(f"n_pipes_per_worker: {n_pipes_per_worker}")
    remainder = len(designs) % n_workers
    jobs = []
    
    start = 0
    for i in range(n_workers):
        # Add an extra element to the first 'remainder' parts
        end = start + n_pipes_per_worker + (1 if i < remainder else 0)
        jobs.append(designs[start:end])
        start = end

    # for epoch in range(n_epochs):
    ray.init(_temp_dir=temp_dir)
    # pipe_offset = epoch * n_pipes_per_worker * n_workers
    rollout_tasks = [rollout.remote(jobs[i], n_episodes=n_episodes_per_pipe, episode_length=episode_length, input_tree=True, state_func=None) for i in range(n_workers)]
    tapes = ray.get(rollout_tasks)
    print("Concatenating tapes...")
    saved_tapes = {key: np.concatenate([d[key] for d in tapes]) for key in tapes[0]}
    print("Saving tapes...")
    data_name = f"1m_short_newstates"
    np.savez_compressed(os.path.join(CONTROLLER_ROOT_DIR, "twist_controller", "self_modeling",  f"{data_name}.npz"), **saved_tapes)

    with h5py.File(os.path.join(CONTROLLER_ROOT_DIR, "twist_controller", "self_modeling",  f"{data_name}.h5"), 'w') as h5_file:
        # Iterate over the saved_tapes dictionary and save each item as a dataset
        for key, value in saved_tapes.items():
            h5_file.create_dataset(key, data=value)

    ray.shutdown()
    shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)


if __name__ == "__main__":

    # gen_ood_testing_data()
    # gen_training_data()
    gen_bigger_training_data()

    # while True:
    #     gen_job(1000, 1, 4, 233)
