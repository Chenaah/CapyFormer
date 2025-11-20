

import pdb
import time
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
from rich.progress import Progress
from scipy.spatial import ConvexHull

from twist_controller.sim.evolution.mutation_meta import random_gen
# from twist_controller.sim.robot_metadesigner import MetaDesigner
from twist_controller.sim.robot_metadesigner import MetaDesignerAsym
from twist_controller.utils.math import euler_to_quaternion, quat_rotate_inverse, quat_rotate_inverse_jax_wxyz, wxyz_to_xyzw
from twist_controller.utils.model import XMLCompiler, get_joint_pos_addr


def is_degenerate(points):
    """Check if the points are degenerate (collinear or duplicate)."""
    # Check if all points are identical
    if np.all(points == points[0]):
        return True
    # Check if points are collinear
    vec1 = points[1] - points[0]
    for i in range(2, len(points)):
        vec2 = points[i] - points[0]
        if np.linalg.norm(np.cross(vec1, vec2)) > 1e-8:
            return False
    return True


def optimize_pose(pipeline, drop_steps=100, move_steps=500, optimization_type="multiply"):


    # d = MetaDesignerAsym(pipeline, mesh_mode="draft")
    d = MetaDesignerAsym(pipeline, mesh_mode="pretty")
    c = XMLCompiler(d.get_xml())
    xml = c.get_string()
    # Note that it is assumed that the accuator is position controlled
    mj_model = mujoco.MjModel.from_xml_string(xml)
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    joint_geom_ids = [mj_model.geom(f'left{i}').id for i in range(mj_model.nu)] + [mj_model.geom(f'right{i}').id for i in range(mj_model.nu)]
    joint_unique_geom_idx = [mj_model.geom(f'left{i}').id for i in range(mj_model.nu)]
    joint_body_idx = [mj_model.geom(i).bodyid.item() for i in joint_unique_geom_idx]
    tape = {}


    ############################################################################################
    # STEP1: Set a random qpos and throw the robot on the ground
    def set_random_qpos(rng):
        fixed_pos = jnp.array([0.0, 0.0, 0.4])
        quaternion = jax.random.uniform(rng, (4,), minval=-1.0, maxval=1.0)
        norm = jnp.linalg.norm(quaternion)
        quaternion = quaternion / norm
        qpos_len = mjx_data.qpos.shape[0]
        remaining_qpos = jnp.zeros(qpos_len - 7)
        new_qpos = jnp.concatenate([fixed_pos, quaternion, remaining_qpos])
        new_data = mjx_data.replace(qpos=new_qpos)
        return new_data

    rand_key = jax.random.PRNGKey(0)
    rngs = jax.random.split(rand_key, 4096)
    mjx_data = jax.vmap(set_random_qpos, in_axes=0)(rngs)
    joint_pos = mjx_data.qpos[:,7:]


    total_steps = drop_steps + move_steps  # Total steps in both loops combined
    progress = Progress()
    task = progress.add_task("[cyan]Optimizing pose...", total=total_steps)
    progress.start()

    last_heights = jnp.copy(mjx_data.qpos[:,2])
    jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    for i in range(drop_steps):
        mjx_data.ctrl.at[:].set(joint_pos)
        mjx_data = jit_step(mjx_model, mjx_data)
        # mean_height = mjx_data.geom_xpos.reshape(-1, mjx_model.ngeom, 3).mean(axis=1)[:,2]
        # print(f"[{i}] Height: ", mean_height, " Max: ", mean_height.max())
        progress.advance(task)

        delta_height = mjx_data.qpos[:,2] - last_heights
        last_heights = jnp.copy(mjx_data.qpos[:,2])
        if jnp.all(delta_height < 1e-3) and i > 20:
            progress.advance(task, advance=total_steps-i)
            break

    # Set the default pos / quat / joint_pos as the current state
    # Assume the current state is stable and valid
    stable_qpos = mjx_data.qpos.clone()
    default_joint_pos = mjx_data.qpos[:,get_joint_pos_addr(mj_model)] # Actuation order
    default_joint_pos = jnp.zeros(default_joint_pos.shape) # Set the default joint pos to 0
    num_joints = mj_model.nu
    
    joint_body_ids = [mj_model.body(f'l{i}').id for i in range(mj_model.nu)] # position of the left bodies should be the same as the right bodies
    stable_avg_joint_height = mjx_data.xpos[:,joint_body_ids,2].mean(axis=1)
    stable_avg_joint_pos = mjx_data.xpos[:,joint_body_ids,:2].mean(axis=1)
    stable_highest_joint_height = mjx_data.xpos[:,joint_body_ids,2].max(axis=1)
    stable_lowest_joint_height = mjx_data.xpos[:,joint_body_ids,2].min(axis=1)

    # contact_geom = mjx_data.contact.geom
    # dist = mjx_data.contact.dist
    # contact_pos = mjx_data.contact.pos
    # contact_floor = jnp.any(contact_geom == 0, axis=2)
    # contact_happen = dist < 0.02
    # contact_happen_with_floor = contact_floor & contact_happen
    # contact_happen_with_floor_bc = jnp.broadcast_to(contact_happen_with_floor[:, :, None], contact_pos.shape)
    # floor_contact_pos = jnp.where(contact_happen_with_floor_bc, contact_pos, 0.0)
    # contact_pos_sum = jnp.sum(floor_contact_pos, axis=1)
    # count_contact = jnp.sum(contact_happen_with_floor, axis=1, keepdims=True)
    # count_contact = jnp.maximum(count_contact, 1) # To avoid division by zero
    # avg_contact_pos = (contact_pos_sum / count_contact)[:,:2]

    # Calculate the convex hull area
    # areas = []
    # for cp, c in zip(contact_pos, contact_happen_with_floor):
    #     floor_points = cp[c]
    #     floor_points = np.asarray(floor_points)[:,:2]
    #     if len(floor_points) < 3 or is_degenerate(floor_points):
    #         area = 0
    #     else:
    #         area = ConvexHull(floor_points).volume
    #     areas.append(area)
    # convex_hull_areas = jnp.array(areas)
    # L = jnp.array(joint_geom_ids)
    # in_L = jnp.isin(contact_geom, L)
    # masked_in_L = in_L * contact_happen_with_floor[..., None]
    # any_in_L = jnp.any(masked_in_L, axis=-1)
    # joint_touch_floor = jnp.any(any_in_L, axis=-1)

    # squared_distances = jnp.sum((stable_avg_joint_pos - avg_contact_pos) ** 2, axis=1)
    # com_distances = jnp.sqrt(squared_distances)

    # Calculate the projected vectors
    
    projected_upward = quat_rotate_inverse_jax_wxyz(stable_qpos[:,3:7], jnp.array([[0,0,1]]))
    


    ############################################################################################
    # STEP2: Add random noise to the joint position and record the trajectories
    actions = "sin"
    def add_random_noise(joint_position, rng):
        noise = jax.random.uniform(rng, shape=(joint_position.shape), minval=-1.0, maxval=1.0)
        return joint_position + noise
    acc_speed = jnp.zeros(mjx_data.qpos.shape[0])
    acc_height = jnp.zeros(mjx_data.qpos.shape[0])
    last_com_pos = mjx_data.xpos[:,joint_body_idx,:2].mean(axis=1)
    acc_vel = jnp.zeros(mjx_data.qpos[:,:2].shape)
    fall_down = jnp.zeros(mjx_data.qpos.shape[0], dtype=bool)
    gravity_vec = jnp.array([[0, 0, -1]])

    for i in range(move_steps):
        if actions == "random":
            rngs = jax.random.split(rand_key, 4096)
            rand_key = rngs[0]
            joint_pos_with_noise = jax.vmap(add_random_noise)(default_joint_pos, rngs)
        elif actions == "sin":
            rngs = jax.random.split(rand_key, 4096)
            rand_key = rngs[0]
            subkeys = jax.random.split(rand_key, 3)

            amplitudes = jax.random.uniform(subkeys[0], (4096,num_joints), minval=-1, maxval=1)  # Random amplitudes for each of the B vectors
            frequencies = jax.random.uniform(subkeys[1], (4096,num_joints), minval=0.02, maxval=0.2)  # Random frequencies for each of the B vectors
            phases = jax.random.uniform(subkeys[2], (4096,num_joints), minval=0., maxval=2*np.pi/0.1)  # Random phases for each of the B vectors

            # Compute the sine function for each B vector
            # sine_values = amplitudes[:, None] * jnp.sin(frequencies[:, None] * i + phases[:, None])
            sine_values = amplitudes * jnp.sin(frequencies * i + phases) # Each joint has a different sine wave

            # Add the sine values to the original array
            # result = array + sine_values

            joint_pos_with_noise = default_joint_pos + sine_values

        # Step the simulation
        mjx_data.ctrl.at[:].set(joint_pos_with_noise)
        mjx_data = jit_step(mjx_model, mjx_data)
        # speed = jnp.linalg.norm(mjx_data.qvel[:,:2], axis=1)

        com_pos = mjx_data.xpos[:,joint_body_idx,:2].mean(axis=1) # 2D COM
        com_vel = (com_pos - last_com_pos) / mj_model.opt.timestep
        last_com_pos = com_pos.copy()
        print("Avg vel: ", com_vel)
        acc_vel += com_vel

        # Check if the robots fall down
        quat = mjx_data.qpos[:,3:7]
        projected_gravity = quat_rotate_inverse_jax_wxyz(quat, gravity_vec)
        dot_results = jnp.einsum('ij,ij->i', projected_upward, -projected_gravity)
        fall = dot_results < 0.1
        fall_down = jnp.logical_or(fall_down, fall)
        print("Fall Down: ", fall_down)

        # acc_speed += speed
        # mean_height = mjx_data.geom_xpos.reshape(-1, mjx_model.ngeom, 3).mean(axis=1)[:,2]
        # acc_height += mean_height
        # print(f"[{i}] Height: ", mean_height, " Max: ", mean_height.max(), "Speed: ", acc_speed)
        progress.advance(task)

    # avg_speed = acc_speed / move_steps
    # avg_height = acc_height / move_steps
    avg_vel = acc_vel / move_steps
    avg_speed = jnp.linalg.norm(avg_vel, axis=1)

    ############################################################################################


    progress.stop()


    return tape


def get_local_vectors(pipeline, init_pos, init_quat, init_joint, render=False):
    # Get the local vectors for reward / done calculation
    d = MetaDesignerAsym(pipeline, mesh_mode="default")
    c = XMLCompiler(d.get_xml())
    xml = c.get_string()

    # m = mujoco.MjModel.from_xml_path(file)
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    d.qpos[:] = 0
    d.qpos[0:3] = init_pos
    d.qpos[3:7] = init_quat
    d.qpos[get_joint_pos_addr(m)] = init_joint

    # Vectors in the global frame
    forward_vec = [1., 0, 0] # This can be an arbitrary vector as we require the robot to be omnidirectional
    updir_vec = [0, 0, 1]

    if render:
        viewer = mujoco.viewer.launch_passive(m, d)
        viewer.__enter__()
    else:
        viewer = None

    n_ctr = 0
    last_projected_forward = np.zeros(3)
    last_projected_upward = np.zeros(3)


    for t in range(100):
        step_start = time.time()

        d.ctrl[:] = init_joint
        mujoco.mj_step(m, d)

        # Example modification of a viewer option: toggle contact points every two seconds.
        # with viewer.lock():
        #     viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2) if vis_contact else 0
            # viewer.opt.frame = 1 # visulize axis
        # pdb.set_trace()
        n_self_collision = 0
        # for contact in d.contact:
        #     if contact.geom1 != 0 and contact.geom2 != 0:
        #         b1 = m.body(m.geom(contact.geom1).bodyid).name
        #         b2 = m.body(m.geom(contact.geom2).bodyid).name
        #         if not (((b1[0] == "l" and b2[0] == "r") or (b1[0] == "r" and b2[0] == "l")) and (b1[1] == b2[1])):
        #             n_self_collision += 1
        # print(n_self_collision)
        quat = d.qpos[3:7]
        projected_forward = quat_rotate_inverse(wxyz_to_xyzw(quat), np.array(forward_vec))
        print("Projected Forward: ", projected_forward)
        projected_upward = quat_rotate_inverse(wxyz_to_xyzw(quat), np.array(updir_vec))
        print("Projected Upward: ", projected_upward)

        # if t > 10 and np.mean(projected_forward - last_projected_forward) < 0.001 and np.mean(projected_upward - last_projected_upward) < 0.001:
        #     break

        # quat = d.qpos[3:7]
        # accurate_projected_gravity = quat_rotate_inverse(wxyz_to_xyzw(quat), np.array([0, 0, -1]))
        # print("DOT: ", np.dot(np.array([np.sin(np.pi/12), 0, np.cos(np.pi/12)]), -accurate_projected_gravity))
        

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        if render:
            viewer.sync()


        # Rudimentary time keeping, will drift relative to wall clock.
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0 and render:
            time.sleep(time_until_next_step)
        
        n_ctr += 1
    
    if render:    
        viewer.__exit__()

    return projected_forward, projected_upward





if __name__ == "__main__":
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2' 
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # p = random_gen(6, None)
    # # init_pos, init_quat, init_joint = optimize_pose(p, 100, 0)
    # # init_pos, init_quat, init_joint = optimize_pose(p, 150, 0, "longlegs")
    # init_pos, init_quat, init_joint, _ = optimize_pose(p, 150, 250, "fastbigbase")
    # print("Design: ", p)
    # print("Init Pos: ", init_pos)
    # print("Init Quat: ", init_quat)
    # print("Init Joint: ", init_joint)

    pipe = [ 0,1,0,0, 0,3,0,0, 0,13,0,0, 0,15,0,0 ]

    init_pos, init_quat, init_joint, _ = optimize_pose(pipe, 15, 250, "fastbigbase")
    # get_local_vectors(pipe,
    #                   init_pos=[0,0,0.1825061684846878],
    #                   init_quat=[0.04657600820064545, -0.8202893137931824, 0.5689055323600769, 0.03609202802181244],
    #                   init_joint=[0]*5,
    #                   render=True)
