"""
This file is for setting an environment for Elastica arm reaching to a target and matching arm
orientation with the target. Actuation torques acting on arm can generate torques in normal,
binormal and tangent direction. Environment set in this file is interfaced with stable-baselines
and OpenAI Gym.
"""

import copy
from collections import defaultdict

import gym
from gym import spaces
import numpy as np

from post_processing import plot_video_with_sphere, plot_video_with_sphere_2D
from MuscleTorquesWithBspline.BsplineMuscleTorques import (
    MuscleTorquesWithVaryingBetaSplines,
)

from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica import *
from elastica import AnalyticalLinearDamper
from elastica.boundary_conditions import FixedConstraint, OneEndFixedBC


# Set base simulator class
class BaseSimulator(BaseSystemCollection, Constraints, Connections, Forcing, CallBacks, Damping):
    pass


class Environment(gym.Env):
    """
    FOUR modes:
    1. fixed target position to be reached (default: need target_position parameter)
    2. random fixed target position to be reached (target changes every reset)
    3. fixed trajectory to be followed (moving target)
    4. random trajectory to be followed (moving target)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        final_time,
        num_steps_per_update,
        number_of_control_points,
        alpha,
        beta,
        target_position,
        COLLECT_DATA_FOR_POSTPROCESSING=False,
        sim_dt=2.5e-4,
        n_elem=40,
        mode=1,
        dim=3.5,
        *args,
        **kwargs,
    ):
        super(Environment, self).__init__()

        self.dim = float(dim)
        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        self.final_time = float(final_time)
        self.h_time_step = float(sim_dt)  # stable time step
        self.total_steps = int(self.final_time / self.h_time_step)
        self.time_step = np.float64(float(self.final_time) / self.total_steps)

        # Video speed / callback sampling
        self.rendering_fps = 60
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step))

        # Control params
        self.number_of_control_points = int(number_of_control_points)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # Target
        self.target_position = np.array(target_position, dtype=np.float64)

        # learning step define through num_steps_per_update
        self.num_steps_per_update = int(num_steps_per_update)
        self.total_learning_steps = int(self.total_steps / self.num_steps_per_update)

        # Action space depends on dim
        if self.dim == 2.0:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.number_of_control_points,), dtype=np.float64
            )
            self.action = np.zeros(self.number_of_control_points, dtype=np.float64)
        elif self.dim == 2.5 or self.dim == 3.0:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(2 * self.number_of_control_points,), dtype=np.float64
            )
            self.action = np.zeros(2 * self.number_of_control_points, dtype=np.float64)
        elif self.dim == 3.5:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(3 * self.number_of_control_points,), dtype=np.float64
            )
            self.action = np.zeros(3 * self.number_of_control_points, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported dim={self.dim}. Use one of: 2.0, 2.5, 3.0, 3.5")

        # Observation space (kept as original design)
        self.obs_state_points = 10
        num_points = int(n_elem / self.obs_state_points)
        num_rod_state = len(np.ones(n_elem + 1)[0::num_points])
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_rod_state * 3 + 8 + 11,),
            dtype=np.float64,
        )

        # Modes
        self.mode = int(mode)

        if self.mode == 2:
            assert "boundary" in kwargs, "need to specify boundary in mode 2"
            self.boundary = np.array(kwargs["boundary"], dtype=np.float64)

        if self.mode == 3:
            assert "target_v" in kwargs, "need to specify target_v in mode 3"
            self.target_v = float(kwargs["target_v"])

        if self.mode == 4:
            assert ("boundary" in kwargs) and ("target_v" in kwargs), "need to specify boundary and target_v in mode 4"
            self.boundary = np.array(kwargs["boundary"], dtype=np.float64)
            self.target_v = float(kwargs["target_v"])

        # Post-processing
        self.COLLECT_DATA_FOR_POSTPROCESSING = bool(COLLECT_DATA_FOR_POSTPROCESSING)

        # Time
        self.time_tracker = np.float64(0.0)

        # Activation smoothing / constraints
        self.acti_diff_coef = float(kwargs.get("acti_diff_coef", 9e-1))
        self.acti_coef = float(kwargs.get("acti_coef", 1e-1))
        self.max_rate_of_change_of_activation = float(
            kwargs.get("max_rate_of_change_of_activation", np.infty)
        )

        # Rod material parameters
        self.E = float(kwargs.get("E", 1e7))
        self.NU = float(kwargs.get("NU", 10))

        # Rod discretization
        self.n_elem = int(n_elem)

    def reset(self):
        self.simulator = BaseSimulator()

        # Rod setup
        n_elem = self.n_elem
        start = np.zeros((3,), dtype=np.float64)
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # pointing upwards
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        density = 1000.0
        E = self.E
        poisson_ratio = 0.5
        G = E / (2.0 * (1.0 + poisson_ratio))

        base_length = 1.0
        radius_tip = 0.05
        radius_base = 0.05
        radius_along_rod = np.linspace(radius_base, radius_tip, n_elem, dtype=np.float64)

        self.shearable_rod = CosseratRod.straight_rod(
            n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius=radius_along_rod,
            density=density,
            youngs_modulus=E,
            shear_modulus=G,
        )
        self.simulator.append(self.shearable_rod)

        # Damping: use AnalyticalLinearDamper consistently (do NOT also pass nu into straight_rod)
        self.simulator.dampen(self.shearable_rod).using(
            AnalyticalLinearDamper,
            damping_constant=self.NU,
            time_step=self.time_step,
        )

        # Target position selection
        if self.mode == 1 or self.mode == 3:
            target_position = self.target_position.copy()
        elif self.mode == 2 or self.mode == 4:
            # random target in boundary (original design)
            xmin, xmax, ymin, ymax, zmin, zmax = self.boundary
            t_x = np.random.uniform(xmin, xmax)
            t_y = np.random.uniform(ymin, ymax)
            if self.dim == 2.0 or self.dim == 2.5:
                t_z = 0.0
            else:
                t_z = np.random.uniform(zmin, zmax)
            target_position = np.array([t_x, t_y, t_z], dtype=np.float64)
        else:
            raise ValueError(f"Unknown mode={self.mode}")

        # Sphere (target)
        self.sphere = Sphere(center=target_position, base_radius=0.05, density=1000.0)

        # Moving target initialization
        if self.mode == 3:
            self.dir_indicator = 1
            self.sphere_initial_velocity = self.target_v
            self.sphere.velocity_collection[..., 0] = [self.sphere_initial_velocity, 0.0, 0.0]

        if self.mode == 4:
            self.trajectory_iteration = 0
            self.rand_direction_1 = np.pi * np.random.uniform(0, 2)
            if self.dim == 2.0 or self.dim == 2.5:
                self.rand_direction_2 = np.pi / 2.0
            else:
                self.rand_direction_2 = np.pi * np.random.uniform(0, 2)

            self.v_x = self.target_v * np.cos(self.rand_direction_1) * np.sin(self.rand_direction_2)
            self.v_y = self.target_v * np.sin(self.rand_direction_1) * np.sin(self.rand_direction_2)
            self.v_z = self.target_v * np.cos(self.rand_direction_2)

            self.sphere.velocity_collection[..., 0] = [self.v_x, self.v_y, self.v_z]
            self.boundaries = np.array(self.boundary, dtype=np.float64)

        # Target orientation (kept)
        if self.mode == 1:
            theta_x = 0.0
            theta_y = np.pi / 4
            theta_z = 0.0
        else:
            theta_x = 0.0
            theta_y = np.random.uniform(-np.pi / 2, np.pi / 2)
            theta_z = 0.0

        theta = np.array([theta_x, theta_y, theta_z], dtype=np.float64)
        R = np.array(
            [
                [
                    -np.sin(theta[1]),
                    np.sin(theta[0]) * np.cos(theta[1]),
                    np.cos(theta[0]) * np.cos(theta[1]),
                ],
                [
                    np.cos(theta[1]) * np.cos(theta[2]),
                    np.sin(theta[0]) * np.sin(theta[1]) * np.cos(theta[2]) - np.sin(theta[2]) * np.cos(theta[0]),
                    np.sin(theta[1]) * np.cos(theta[0]) * np.cos(theta[2]) + np.sin(theta[0]) * np.sin(theta[2]),
                ],
                [
                    np.sin(theta[2]) * np.cos(theta[1]),
                    np.sin(theta[0]) * np.sin(theta[1]) * np.sin(theta[2]) + np.cos(theta[0]) * np.cos(theta[2]),
                    np.sin(theta[1]) * np.sin(theta[2]) * np.cos(theta[0]) - np.sin(theta[0]) * np.cos(theta[2]),
                ],
            ],
            dtype=np.float64,
        )
        self.sphere.director_collection[..., 0] = R
        self.simulator.append(self.sphere)

        # Quaternion for target orientation
        Q = self.sphere.director_collection[..., 0]
        qw = np.sqrt(1 + Q[0, 0] + Q[1, 1] + Q[2, 2]) / 2
        qx = (Q[2, 1] - Q[1, 2]) / (4 * qw)
        qy = (Q[0, 2] - Q[2, 0]) / (4 * qw)
        qz = (Q[1, 0] - Q[0, 1]) / (4 * qw)
        self.target_tip_orientation = np.array([qw, qx, qy, qz], dtype=np.float64)

        # Sphere boundary for mode=4
        class WallBoundaryForSphere(FixedConstraint):
            def __init__(self, boundaries):
                self.x_boundary_low = boundaries[0]
                self.x_boundary_high = boundaries[1]
                self.y_boundary_low = boundaries[2]
                self.y_boundary_high = boundaries[3]
                self.z_boundary_low = boundaries[4]
                self.z_boundary_high = boundaries[5]

            def constrain_values(self, sphere, time):
                pos_x = sphere.position_collection[0]
                pos_y = sphere.position_collection[1]
                pos_z = sphere.position_collection[2]
                radius = sphere.radius
                vx = sphere.velocity_collection[0]
                vy = sphere.velocity_collection[1]
                vz = sphere.velocity_collection[2]

                if (pos_x - radius) < self.x_boundary_low or (pos_x + radius) > self.x_boundary_high:
                    sphere.velocity_collection[:] = np.array([-vx, vy, vz])
                if (pos_y - radius) < self.y_boundary_low or (pos_y + radius) > self.y_boundary_high:
                    sphere.velocity_collection[:] = np.array([vx, -vy, vz])
                if (pos_z - radius) < self.z_boundary_low or (pos_z + radius) > self.z_boundary_high:
                    sphere.velocity_collection[:] = np.array([vx, vy, -vz])

            def constrain_rates(self, sphere, time):
                pass

        if self.mode == 4:
            self.simulator.constrain(self.sphere).using(WallBoundaryForSphere, boundaries=self.boundaries)

        # Fix rod base
        self.simulator.constrain(self.shearable_rod).using(
            OneEndFixedBC, constrained_position_idx=(0,), constrained_director_idx=(0,)
        )

        # Actuation torques (normal, binormal, tangent)
        self.torque_profile_list_for_muscle_in_normal_dir = defaultdict(list)
        self.spline_points_func_array_normal_dir = np.zeros(self.number_of_control_points, dtype=np.float64)
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_normal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("normal"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_normal_dir,
        )

        self.torque_profile_list_for_muscle_in_binormal_dir = defaultdict(list)
        self.spline_points_func_array_binormal_dir = np.zeros(self.number_of_control_points, dtype=np.float64)
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_binormal_dir,
            muscle_torque_scale=self.alpha,
            direction=str("binormal"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_binormal_dir,
        )

        self.torque_profile_list_for_muscle_in_twist_dir = defaultdict(list)
        self.spline_points_func_array_twist_dir = np.zeros(self.number_of_control_points, dtype=np.float64)
        self.simulator.add_forcing_to(self.shearable_rod).using(
            MuscleTorquesWithVaryingBetaSplines,
            base_length=base_length,
            number_of_control_points=self.number_of_control_points,
            points_func_array=self.spline_points_func_array_twist_dir,
            muscle_torque_scale=self.beta,
            direction=str("tangent"),
            step_skip=self.step_skip,
            max_rate_of_change_of_activation=self.max_rate_of_change_of_activation,
            torque_profile_recorder=self.torque_profile_list_for_muscle_in_twist_dir,
        )

        # Callbacks (optional)
        class ArmMuscleBasisCallBack(CallBackBaseClass):
            def __init__(self, step_skip: int, callback_params: dict):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(system.position_collection.copy())
                    self.callback_params["directors"].append(system.director_collection.copy())
                    self.callback_params["radius"].append(system.radius.copy())
                    self.callback_params["com"].append(system.compute_position_center_of_mass())

        class RigidSphereCallBack(CallBackBaseClass):
            def __init__(self, step_skip: int, callback_params: dict):
                CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(self, system, time, current_step: int):
                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(system.position_collection.copy())
                    self.callback_params["directors"].append(system.director_collection.copy())
                    self.callback_params["radius"].append(copy.deepcopy(system.radius))
                    self.callback_params["com"].append(system.compute_position_center_of_mass())

        if self.COLLECT_DATA_FOR_POSTPROCESSING:
            self.post_processing_dict_rod = defaultdict(list)
            self.simulator.collect_diagnostics(self.shearable_rod).using(
                ArmMuscleBasisCallBack, step_skip=self.step_skip, callback_params=self.post_processing_dict_rod
            )

            self.post_processing_dict_sphere = defaultdict(list)
            self.simulator.collect_diagnostics(self.sphere).using(
                RigidSphereCallBack, step_skip=self.step_skip, callback_params=self.post_processing_dict_sphere
            )

        # Finalize
        self.simulator.finalize()
        self.do_step, self.stages_and_updates = extend_stepper_interface(self.StatefulStepper, self.simulator)

        # Reset trackers
        self.on_goal = 0
        self.current_step = 0
        self.time_tracker = np.float64(0.0)
        self.previous_action = None

        return self.get_state()

    def sampleAction(self):
        """Return a random action with correct dimension for current dim."""
        return self.action_space.sample()

    def get_state(self):
        rod_state = self.shearable_rod.position_collection
        r_s_a = rod_state[0]
        r_s_b = rod_state[1]
        r_s_c = rod_state[2]

        num_points = int(self.n_elem / self.obs_state_points)

        rod_compact_state = np.concatenate(
            (
                r_s_a[0 : len(r_s_a) + 1 : num_points],
                r_s_b[0 : len(r_s_b) + 1 : num_points],
                r_s_c[0 : len(r_s_c) + 1 : num_points],
            )
        )

        rod_compact_velocity = self.shearable_rod.velocity_collection[..., -1]
        rod_compact_velocity_norm = np.array([np.linalg.norm(rod_compact_velocity)])
        rod_compact_velocity_dir = np.where(
            rod_compact_velocity_norm != 0,
            rod_compact_velocity / rod_compact_velocity_norm,
            0.0,
        )

        sphere_compact_state = self.sphere.position_collection.flatten()
        sphere_compact_velocity = self.sphere.velocity_collection.flatten()
        sphere_compact_velocity_norm = np.array([np.linalg.norm(sphere_compact_velocity)])
        sphere_compact_velocity_dir = np.where(
            sphere_compact_velocity_norm != 0,
            sphere_compact_velocity / sphere_compact_velocity_norm,
            0.0,
        )

        Q = self.shearable_rod.director_collection[..., -1]
        qw = np.sqrt(1 + Q[0, 0] + Q[1, 1] + Q[2, 2]) / 2
        qx = (Q[2, 1] - Q[1, 2]) / (4 * qw)
        qy = (Q[0, 2] - Q[2, 0]) / (4 * qw)
        qz = (Q[1, 0] - Q[0, 1]) / (4 * qw)
        self.rod_tip_orientation = np.array([qw, qx, qy, qz])

        state = np.concatenate(
            (
                rod_compact_state,
                rod_compact_velocity_norm,
                rod_compact_velocity_dir,
                self.rod_tip_orientation,
                sphere_compact_state,
                sphere_compact_velocity_norm,
                sphere_compact_velocity_dir,
                self.target_tip_orientation,
            )
        )
        return state

    def step(self, action):
        self.action = np.array(action, dtype=np.float64)

        # dispatch control points into directions
        ncp = self.number_of_control_points
        if self.dim == 2.0:
            self.spline_points_func_array_normal_dir[:] = self.action[:ncp]
            self.spline_points_func_array_binormal_dir[:] = 0.0
            self.spline_points_func_array_twist_dir[:] = 0.0
        elif self.dim == 2.5:
            self.spline_points_func_array_normal_dir[:] = self.action[:ncp]
            self.spline_points_func_array_binormal_dir[:] = 0.0
            self.spline_points_func_array_twist_dir[:] = self.action[ncp:]
        elif self.dim == 3.0:
            self.spline_points_func_array_normal_dir[:] = self.action[:ncp]
            self.spline_points_func_array_binormal_dir[:] = self.action[ncp:]
            self.spline_points_func_array_twist_dir[:] = 0.0
        elif self.dim == 3.5:
            self.spline_points_func_array_normal_dir[:] = self.action[:ncp]
            self.spline_points_func_array_binormal_dir[:] = self.action[ncp : 2 * ncp]
            self.spline_points_func_array_twist_dir[:] = self.action[2 * ncp :]
        else:
            raise ValueError(f"Unsupported dim={self.dim}")

        # integrate num_steps_per_update steps
        for _ in range(self.num_steps_per_update):
            self.time_tracker = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time_tracker,
                self.time_step,
            )

        # moving target update (kept)
        if self.mode == 3:
            if (self.current_step % (1.0 / (self.h_time_step * self.num_steps_per_update))) == 0:
                if self.dir_indicator == 1:
                    self.sphere.velocity_collection[..., 0] = [0.0, -self.sphere_initial_velocity, 0.0]
                    self.dir_indicator = 2
                elif self.dir_indicator == 2:
                    self.sphere.velocity_collection[..., 0] = [-self.sphere_initial_velocity, 0.0, 0.0]
                    self.dir_indicator = 3
                elif self.dir_indicator == 3:
                    self.sphere.velocity_collection[..., 0] = [0.0, +self.sphere_initial_velocity, 0.0]
                    self.dir_indicator = 4
                elif self.dir_indicator == 4:
                    self.sphere.velocity_collection[..., 0] = [+self.sphere_initial_velocity, 0.0, 0.0]
                    self.dir_indicator = 1

        if self.mode == 4:
            self.trajectory_iteration += 1
            if self.trajectory_iteration == 500:
                self.rand_direction_1 = np.pi * np.random.uniform(0, 2)
                if self.dim == 2.0 or self.dim == 2.5:
                    self.rand_direction_2 = np.pi / 2.0
                else:
                    self.rand_direction_2 = np.pi * np.random.uniform(0, 2)

                self.v_x = self.target_v * np.cos(self.rand_direction_1) * np.sin(self.rand_direction_2)
                self.v_y = self.target_v * np.sin(self.rand_direction_1) * np.sin(self.rand_direction_2)
                self.v_z = self.target_v * np.cos(self.rand_direction_2)
                self.sphere.velocity_collection[..., 0] = [self.v_x, self.v_y, self.v_z]
                self.trajectory_iteration = 0

        self.current_step += 1

        # observe state
        state = self.get_state()

        # reward (kept)
        dist = np.linalg.norm(self.shearable_rod.position_collection[..., -1] - self.sphere.position_collection[..., 0])
        reward_dist = -np.square(dist).sum()

        orientation_dist = 1.0 - np.dot(self.rod_tip_orientation, self.target_tip_orientation) ** 2
        orientation_penalty = -((orientation_dist) ** 2)
        reward = 1.0 * reward_dist + 0.5 * orientation_penalty

        done = False

        # NaN check
        invalid_pos = _isnan_check(self.shearable_rod.position_collection)
        if invalid_pos:
            self.shearable_rod.position_collection[:] = 0.0
            reward = -10000.0
            state = self.get_state()
            done = True

        if self.current_step >= self.total_learning_steps:
            done = True

        self.previous_action = self.action

        invalid_state = _isnan_check(state)
        if invalid_state:
            reward = -10000.0
            state = np.zeros(state.shape, dtype=np.float64)
            done = True

        return state, float(reward), bool(done), {"ctime": self.time_tracker}

    def render(self, mode="human"):
        return

    def post_processing(self, filename_video, SAVE_DATA=False, **kwargs):
        if not self.COLLECT_DATA_FOR_POSTPROCESSING:
            raise RuntimeError("No callback data collected. Set COLLECT_DATA_FOR_POSTPROCESSING=True")

        plot_video_with_sphere_2D(
            [self.post_processing_dict_rod],
            [self.post_processing_dict_sphere],
            video_name="2d_" + filename_video,
            fps=self.rendering_fps,
            step=1,
            vis2D=False,
            **kwargs,
        )

        plot_video_with_sphere(
            [self.post_processing_dict_rod],
            [self.post_processing_dict_sphere],
            video_name="3d_" + filename_video,
            fps=self.rendering_fps,
            step=1,
            vis2D=False,
            **kwargs,
        )

        if SAVE_DATA:
            import os
            save_folder = os.path.join(os.getcwd(), "data")
            os.makedirs(save_folder, exist_ok=True)

            position_rod = np.array(self.post_processing_dict_rod["position"])
            position_rod = 0.5 * (position_rod[..., 1:] + position_rod[..., :-1])

            np.savez(
                os.path.join(save_folder, "arm_data.npz"),
                position_rod=position_rod,
                radii_rod=np.array(self.post_processing_dict_rod["radius"]),
                n_elems_rod=self.shearable_rod.n_elems,
                position_sphere=np.array(self.post_processing_dict_sphere["position"]),
                radii_sphere=np.array(self.post_processing_dict_sphere["radius"]),
            )

            np.savez(
                os.path.join(save_folder, "arm_activation.npz"),
                torque_mag=np.array(self.torque_profile_list_for_muscle_in_normal_dir["torque_mag"]),
                torque_muscle=np.array(self.torque_profile_list_for_muscle_in_normal_dir["torque"]),
            )