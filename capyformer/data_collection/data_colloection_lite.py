# Get rid of MJX

import pickle
import os
import pdb
import time
import numpy as np
from tqdm import trange
from twist_controller import CONTROLLER_ROOT_DIR
from twist_controller.envs.env_sim import ZeroSim
from twist_controller.sim.evolution.utils import update_cfg_with_pipeline
from twist_controller.utils.files import load_cfg

def _get_state(env):
    quats = env.sensors["quat"].flatten()
    ang_vel = env.sensors["gyro"].flatten()
    dof_pos = env.data.qpos[7:]
    dof_vel = env.data.qvel[6:]
    state = np.concatenate([quats, ang_vel, dof_pos, dof_vel])
    return state

def one_hot_encode(labels, flatten=True, num_classes=None):
    if num_classes is None:
        num_classes = np.max(labels) + 1
    one_hot = np.zeros((len(labels), num_classes))
    one_hot[np.arange(len(labels)), labels] = 1
    if flatten:
        return one_hot.flatten()
    return one_hot




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



cfg = {
        "theta": 0.4625123,
        "R": 0.07,
        "r": 0.03,
        "l_": 0.236,
        "delta_l": 0,
        "stick_ball_l": 0.005,
        "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
        "stick_mass": 0.231, #0.26,
        "top_hemi_mass": 0.74,
        "bottom_hemi_mass": 0.534,
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

pla_cfg = {
        "theta": 0.4625123,
        "R": 0.07,
        "r": 0.03,
        "l_": 0.236,
        "delta_l": 0,
        "stick_ball_l": 0.005,
        "a": 0.236/4, # 0.0380409255338946, # l/6 stick center to the dock center on the side
        "stick_mass":0.237,
        "top_hemi_mass": 0.225,
        "bottom_hemi_mass": 0.212,
        "battery_mass": 0.122,
        "motor_mass": 0.317,
        "pcb_mass": 0.1
    }

mesh_dict_legacy = {
                "up": "hemisphere014.obj",
                "bottom": "hemisphere014f.obj",
                "stick": "leg4.4.obj",
            }


def rollout(pipelines, n_episodes=5000, episode_length=1000, dataset_name="rollouts"):

    tape = {"states": [], "actions": [], "pipeline": []}
    # (B, T, S) and (B, T, A)
    legacy_tape = []

    for i_p, pipeline in enumerate(pipelines):

        pipeline_oh = one_hot_encode(pipeline, num_classes=16)
        # pdb.set_trace()
        conf = load_cfg("sim_train_quadrupedX4", alg="sbx")
        conf = update_cfg_with_pipeline(conf, pipeline, init_pose_type="original", robot_cfg=cfg, mesh_dict=mesh_dict_legacy)
        conf.agent.reward_version = "alive"
        # conf.sim.fully_randomize_orientation = True
        conf.sim.init_pos = [0, 0, 2]
        conf.sim.noisy_actions = False
        conf.sim.noisy_observations = False 

        conf.sim.render = True


        n_act = conf.agent.num_act
        env = ZeroSim(conf)

        

        for i_ep in trange(n_episodes):
                
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

            state = _get_state(env)
            for t in range(episode_length):
                t0 = time.time()
                state_rollout.append(state)
                action = amplitudes * np.sin(frequencies * t + phases)
                action_rollout.append(action)
                env.step(action)
                state = _get_state(env)
                time.sleep(max(0, env.dt - (time.time() - t0)))

            tape["states"].append(np.array(state_rollout))
            tape["actions"].append(np.array(action_rollout))
            tape["pipeline"].append(pipeline_oh)

            legacy_tape.append({"states": np.array(state_rollout), 
                                "actions": np.array(action_rollout),
                                "pipeline_onehot": np.array([pipeline_oh]*episode_length),
                                "pipeline_idx": np.array([[i_p]]*episode_length),
                                "tree_nodes": np.array([pipeidx_encode(pipeline)]*episode_length)
                                })

        quit()
        # if i_p % 10 == 0:
        np.savez_compressed(os.path.join(CONTROLLER_ROOT_DIR, "twist_controller", "self_modeling", f"{dataset_name}.npz"), **tape)
        with open(os.path.join(CONTROLLER_ROOT_DIR, "twist_controller", "self_modeling", f"{dataset_name}_legacy.pkl"), "wb") as f:
            pickle.dump(legacy_tape, f)


if __name__ == "__main__":

    pipelines = [[ 0,1,0,0, 0,3,0,0, 0,13,0,0, 0,15,0,0 ],
                [ 0,2,0,0, 0,3,0,1, 0,13,0,1, 0,15,0,1 ],
                [ 0,3,0,0, 0,4,0,1, 0,12,0,1, 0,14,0,1 ]]
    
    # rollout(pipelines, n_episodes=3000, dataset_name="mini3rollouts")

    rollout(pipelines, n_episodes=200, dataset_name="rollouts_test")
    

    # print(pipeidx_encode([ ]))
    # print(pipeidx_encode(pipelines[1]))
    # print(pipeidx_encode(pipelines[2]))