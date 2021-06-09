import logging
import multiprocessing as mp
import queue
from timeit import default_timer as timer

import numpy as np
import scipy.spatial
import torch
import trimesh.transformations as tra
from autolab_core import Logger
from tracikpy import TracIKSolver

from .collision_checker import (
    FCLMultiSceneCollisionChecker,
    FCLSelfCollisionChecker,
    NNSceneCollisionChecker,
    NNSelfCollisionChecker,
)
from .constants import ROBOT_LABEL, TABLE_LABEL
from .robot import Robot

np.set_printoptions(suppress=True)


class InfeasibleTargetException(Exception):
    pass


class IKProc(mp.Process):
    """
    Used for finding ik in parallel.
    """

    def __init__(
        self,
        output_queue,
    ):
        """
        Args:
        output_queue: mp.Queue, the queue that all the output data
            that is computed is added to.
        """
        super().__init__()
        self.output_queue = output_queue
        self.input_queue = mp.Queue()
        self.ik_solver = TracIKSolver(
            "data/panda/panda.urdf",
            "panda_link0",
            "right_gripper",
            timeout=0.05,
        )

    def _ik(self, ee_pose, qinit):
        qout = self.ik_solver.ik(
            ee_pose, qinit[: self.ik_solver.number_of_joints]
        )
        # Add fingers if not None
        if qout is not None:
            qout = np.append(qout, np.zeros(2))
        return qout

    def run(self):
        """
        the main function of each path collector process.
        """
        while True:
            try:
                request = self.input_queue.get(timeout=1)
            except queue.Empty:
                continue
            if request[0] == "ik":
                self.output_queue.put(
                    (request[3], self._ik(request[1], request[2]))
                )

    def ik(self, grasp, init_q, ind=None):
        self.input_queue.put(("ik", grasp, init_q, ind))


class MPPIPolicy:
    def __init__(
        self,
        num_proc,
        num_path,
        horizon,
        noise_std,
        max_step=0.02,
        lift_height=0.5,
        lift_steps=100,
        collision_steps=10,
        transition_threshold=0.05,
        self_coll_nn=None,
        scene_coll_nn=None,
        cam_type="ws",
        device=0,
        log_file=None,
    ):
        """
        Args:
          mode: string, 'qspace' or 'tspace' for configuration space and task space.
          num_proc: number of parallel processes used for IK solutions.
          num_path: number of desired paths.
          horizon: int, horizon of unrolling
          noise_std: standard deviation of noise in qspace.
          max_step: float, distance to move along trajectory at each timestep in rollout
          lift_height: height in m to lift after grasping, placing
          lift_steps: int, number of steps in lifting plan
          collision_steps: int, number of discrete intervals to check collisions for between each timestep
          transition_threshold: float, threshold for moving between states
          self_coll_nn: path to a network for checking self collisions, defaults to FCL checker
          scene_coll_nn: path to scene collision network, defaults to FCL checker
          cam_type: "ws" or "hand" determines which camera to use
          device: int, compute device for rollouts
          safe: bool, pause before grasping and placing and wait for confirmation
          log_file: str, path to a logging file
        """
        self.num_path = num_path
        self.horizon = horizon
        self.noise_std = noise_std
        self.max_step = max_step
        self.lift_height = lift_height
        self.lift_steps = lift_steps
        self.collision_steps = collision_steps
        self.transition_threshold = transition_threshold
        self.device = device

        if cam_type not in ["ws", "hand"]:
            raise ValueError("Invalid cam_type (ws or hand)")
        self.cam_type = 0 if cam_type == "hand" else 1

        self.num_proc = num_proc
        self.output_queue = mp.Queue()
        self.ik_procs = []
        for i in range(num_proc):
            self.ik_procs.append(
                IKProc(
                    self.output_queue,
                )
            )
            self.ik_procs[-1].daemon = True
            self.ik_procs[-1].start()

        self.robot = Robot(
            "data/panda/panda.urdf",
            "right_gripper",
            device=torch.device("cuda:{:d}".format(self.device)),
        )

        # Set up self collision checkers
        self.self_collision_checker = (
            NNSelfCollisionChecker(
                self_coll_nn, device=torch.device(f"cuda:{self.device}")
            )
            if self_coll_nn is not None
            else FCLSelfCollisionChecker(self.robot)
        )
        for i in range(len(self.robot.links) - 1):
            self.self_collision_checker.set_allowed_collisions(
                self.robot.links[i].name, self.robot.links[i + 1].name
            )
        self.self_collision_checker.set_allowed_collisions(
            "panda_hand", "panda_rightfinger"
        )
        self.self_collision_checker.set_allowed_collisions(
            "panda_link7", "panda_hand"
        )

        self.scene_collision_checker = (
            NNSceneCollisionChecker(
                scene_coll_nn,
                self.robot,
                device=torch.device(f"cuda:{self.device}"),
                use_knn=False,
            )
            if scene_coll_nn is not None
            else FCLMultiSceneCollisionChecker(self.robot, use_scene_pc=True)
        )

        self.target_object_name = None
        self.current_target_index = -1
        self.target_object_names = None

        self.logger = Logger.get_logger(
            "MPPIPolicy", log_file=log_file, log_level=logging.DEBUG
        )

        self.state = 0
        self.grasps = None
        self.grasp_scores = None
        self.cur_grasps_init = None
        self.cur_grasps_final = None
        self.ik_cur_grasps_init = None
        self.ik_cur_grasps_final = None
        self.lift_plan = None
        self.reach_plan = None
        self.prev_plan_ind = None
        self.ee_offset = None
        self.prev_rollout_lens = []
        self._reset()

    def _reset(self):
        self.grasps = None
        self.grasp_scores = None
        self.lift_plan = None
        self.reach_plan = None
        self.prev_plan_ind = None

    def _update_state(self, obs):
        """
        syncs up the scene with the latest observation.
        Args:
        obs: np.array, pointcloud of the scene
        state: dict, this is the gym_state_dict coming from scene managers.
          contains info about robot and object."""
        self.robot_q = obs["robot_q"].astype(np.float64).copy()
        label_map = {
            "table": TABLE_LABEL,
            "target": 1,
            "objs": 2,
            "robot": ROBOT_LABEL,
        }
        rtm = np.eye(4)
        rtm[:3, 3] = (-0.6, 0, 0.1)
        in_obs = {
            "pc": obs["pc"][self.cam_type],
            "pc_label": obs["pc_label"][self.cam_type],
            "label_map": label_map,
            "camera_pose": tra.euler_matrix(np.pi / 2, 0, 0)
            @ obs["camera_pose"][self.cam_type],
            "robot_to_model": rtm,
            "model_to_robot": np.linalg.inv(rtm),
        }

        # Segment and set object for placement
        if self.state == 3:
            # Get ee position and mask for points
            self.robot.set_joint_cfg(self.robot_q)
            ee_pose = self.robot.ee_pose[0].cpu().numpy()

            if self.ik_cur_grasps_init is None:
                obj_in_hand = not (self.robot_q[-1] < 0.001 > self.robot_q[-2])
                if obj_in_hand:
                    self.ee_offset = (
                        ee_pose[:3, 3]
                        - tra.transform_points(
                            in_obs["pc"][
                                in_obs["pc_label"] == label_map["target"]
                            ],
                            in_obs["camera_pose"],
                        ).mean(axis=0)
                    )

                    self.scene_collision_checker.set_object(in_obs)

                else:
                    self.logger.error(
                        "************** OBJECT DROPPED ***********"
                    )
                    self.state = 0

        self.scene_collision_checker.set_scene(in_obs)

    # Tries to find a lifting plan given an initial configuration and given EE offsets
    def _compute_lift_plan(self, init_q, lift_offsets, num_plans=1):
        """computes a line in configuration space that lift the object."""
        if not isinstance(lift_offsets, list) or len(lift_offsets) != 3:
            raise ValueError(
                "lift_offsets should be list of length 3. {}".format(
                    lift_offsets
                )
            )
        cur_q = init_q.copy()
        self.robot.set_joint_cfg(cur_q)
        self.logger.debug("looking for lift offset {}".format(lift_offsets))
        final_poses = self.robot.ee_pose.repeat(num_plans, 1, 1).cpu().numpy()
        for i in range(3):
            if isinstance(lift_offsets[i], list):
                final_poses[:, i, 3] += np.random.uniform(
                    *lift_offsets[i], size=num_plans
                )
            else:
                final_poses[:, i, 3] = lift_offsets[i]

        # Check ik for all final grasps distributed between processes
        for i, p in enumerate(final_poses):
            self.ik_procs[i % self.num_proc].ik(p, cur_q)

        # collect computed iks
        final_ik = []
        for _ in range(len(final_poses)):
            output = self.output_queue.get(True)
            assert isinstance(output, tuple)
            assert len(output) == 2
            if output[1] is None:
                continue
            final_ik.append(output[1])

        # If no IK or IK is wildly different
        final_ik = np.asarray(final_ik)
        if len(final_ik) == 0:
            return None
        final_ik = final_ik[
            np.linalg.norm(final_ik - cur_q, ord=np.inf, axis=-1) < 1
        ]
        if len(final_ik) == 0:
            return None

        # keep the fingers closed during lift
        final_ik[:, -2:] = 0
        cur_q[-2:] = 0
        alpha = np.linspace(0, 1, self.lift_steps).reshape(1, -1, 1)
        plans = final_ik[:, None, :] * alpha + cur_q.reshape(1, 1, -1) * (
            1 - alpha
        )
        valid = ~(
            self.self_collision_checker(plans.reshape(-1, self.robot.dof))
            .reshape(len(plans), self.lift_steps)
            .any(axis=1)
        )
        plans = plans[valid.cpu().numpy()]
        return plans[0] if len(plans) > 0 else None

    def _compute_batch_reward(self, qs):
        """
        Args:
          paths: tensor, shape (num_paths, horizon, dof)

        Returns:
          closest distance, tensor float, (batch_size)
          closest_grasp_index min index
        """
        if qs.shape[-1] != self.ik_cur_grasps_init.shape[-1]:
            raise ValueError(
                "last dim should be equal {} {}".format(
                    qs.shape, self.ik_cur_grasps_init.shape
                )
            )
        distance = torch.norm(
            qs[..., None, :7]
            - torch.from_numpy(self.ik_cur_grasps_init[:, :7]).to(self.device),
            dim=-1,
        )
        output = torch.min(distance, dim=-1)
        return -output[0], output[1]

    # Trim rollouts by their collisions and max rewards
    def _trim_rollouts(self, rewards, collisions):
        # Array of connected points collision free along path
        connected = torch.cat(
            (
                torch.ones(
                    self.num_path, 1, dtype=torch.bool, device=self.device
                ),
                collisions.sum(dim=-1).cumsum(dim=-1) == 0,
            ),
            dim=1,
        )
        rewards = rewards * connected - 10000 * ~connected
        rollout_values, rollout_lengths = rewards.max(dim=1)
        return rollout_values, rollout_lengths

    def _collect_rollouts(self):
        """
        Unrolls self.horizon steps and push the trajectory value + trajectory
        in output_queue.

        Rollouts are list of configurations or (configuration, bool) where the
        bool variable indicates whether you want the controller to accurately get
        to that way point. Otherwise the contoller will proceed to next waypoints
        if it is roughly close.

        For context, for one environment you can provide a list of waypoints and
        the controller only execs the first set of actions that can be done in
        1/--control-frequency.
        """
        init_q = self.robot_q.copy()

        # Setup rollouts array and initialize with first q
        rewards = torch.empty(
            (self.num_path, self.horizon), device=self.device
        )
        rollouts = torch.empty(
            (self.num_path, self.horizon, self.robot.dof), device=self.device
        )
        rollouts[:, 0] = (
            torch.from_numpy(init_q)
            .to(self.device)
            .reshape([1, -1])
            .repeat([self.num_path, 1])
        )

        # Find straight line trajectory to goal
        _, closest_g_ind = self._compute_batch_reward(rollouts[:, 0])
        closest_g_q = torch.from_numpy(self.ik_cur_grasps_init).to(
            self.device
        )[closest_g_ind]
        greedy_dir = torch.nn.functional.normalize(
            closest_g_q - rollouts[:, 0], dim=-1
        )

        # Perturb the greedy direction and renormalize (keep one greedy)
        noise_dir = torch.empty(
            (self.num_path, self.robot.dof),
            device=self.device,
        ).normal_(mean=0, std=self.noise_std)
        noise_dir[0] = 0.0
        rollout_dirs = torch.nn.functional.normalize(greedy_dir + noise_dir)

        # Generate rollouts
        step_sizes = torch.empty(
            (self.num_path, self.horizon - 1, 1), device=self.device
        ).fill_(self.max_step)
        rollouts[:, 1:] = rollout_dirs[:, None] * step_sizes
        rollouts = rollouts.cumsum(dim=1)

        # Clip actions to joint limits
        rollouts = torch.max(
            rollouts,
            self.robot.min_joints,
        )
        rollouts = torch.min(
            rollouts,
            self.robot.max_joints,
        )

        # Set fingers to open or closed depending on desired cfg
        rollouts[..., -2:] = closest_g_q[0, -2:]
        rewards = self._compute_batch_reward(rollouts)[0]
        return rollouts, rewards

    # Check collisions between the robot and optionally the object in hand with the scene
    # for a batch of rollouts
    def _check_collisions(self, rollouts, check_obj=False):
        alpha = (
            torch.linspace(0, 1, self.collision_steps)
            .reshape([1, 1, -1, 1])
            .to(self.device)
        )
        waypoints = (
            alpha * rollouts[:, 1:, None]
            + (1.0 - alpha) * rollouts[:, :-1, None]
        ).reshape(-1, self.robot.dof)

        if isinstance(self.self_collision_checker, FCLSelfCollisionChecker):
            coll_mask = np.zeros(len(waypoints), dtype=np.bool)
            for i, q in enumerate(waypoints):
                coll_mask[i] = self.self_collision_checker(q)
        else:
            coll_mask = self.self_collision_checker(waypoints)

        coll_mask |= self.scene_collision_checker(waypoints, threshold=0.45)

        if check_obj:
            obj_trs = torch.cat(
                (
                    self.robot.ee_pose[:, :3, 3]
                    - torch.from_numpy(self.ee_offset).float().to(self.device),
                    torch.ones(len(self.robot.ee_pose), 1, device=self.device),
                ),
                dim=1,
            )
            model_obj_trs = (
                self.scene_collision_checker.robot_to_model @ obj_trs.T
            )
            obj_coll = self.scene_collision_checker.check_object_collisions(
                model_obj_trs[:3].T, threshold=0.45
            )
            coll_mask |= obj_coll.reshape(coll_mask.shape)
        return coll_mask.reshape(
            self.num_path, self.horizon - 1, self.collision_steps
        )

    # localizing in the prev rollout by computing the closest point of rollout.
    def _localize_in_plan(self, cur_q, plan):
        distances = np.linalg.norm(plan - cur_q, axis=1)
        return np.argmin(distances)

    # Main policy call, returns a rollout based on the current observation
    def get_action(self, obs):
        if (
            "keyboard_next_target_event" in obs
            and obs["keyboard_next_target_event"]
        ):
            self.state = 0
            self._reset()
            return None

        self._update_state(obs)

        # Reach if not at a final grasp
        if self.state == 0:
            cur_grasps_init = obs["grasps_init"]
            cur_grasps_final = obs["grasps_final"]
            if len(cur_grasps_init) == 0:
                self.logger.warning("No grasps found on target!")
                self._reset()
                return None

            # Check ik for all init grasps distributed between processes
            ik_time = timer()
            num_grasps = len(cur_grasps_init)
            for i, g in enumerate(cur_grasps_init):
                self.ik_procs[i % self.num_proc].ik(g, self.robot_q, ind=i)

            # collect computed iks
            cur_grasp_init_inds = []
            ik_cur_grasps_init = []
            for _ in range(num_grasps):
                output = self.output_queue.get(True)
                assert isinstance(output, tuple)
                assert len(output) == 2
                if output[1] is None:
                    continue
                cur_grasp_init_inds.append(output[0])
                ik_cur_grasps_init.append(output[1])

            cur_grasps_init = cur_grasps_init[cur_grasp_init_inds]
            cur_grasps_final = cur_grasps_final[cur_grasp_init_inds]

            if len(cur_grasps_init) == 0:
                self.logger.warning("No grasp IKs found!")
                self._reset()
                return None

            # Check ik for all final grasps distributed between processes
            for i, (g, q) in enumerate(
                zip(cur_grasps_final, ik_cur_grasps_init)
            ):
                self.ik_procs[i % self.num_proc].ik(g, q, ind=i)

            # collect computed iks
            cur_grasp_final_inds = []
            ik_cur_grasps_final = []
            for _ in range(len(cur_grasps_final)):
                output = self.output_queue.get(True)
                assert isinstance(output, tuple)
                assert len(output) == 2
                if output[1] is None:
                    continue
                cur_grasp_final_inds.append(output[0])
                ik_cur_grasps_final.append(output[1])

            cur_grasps_final = cur_grasps_final[cur_grasp_final_inds]
            cur_grasps_init = cur_grasps_init[cur_grasp_final_inds]
            ik_cur_grasps_init = np.asarray(ik_cur_grasps_init)[
                cur_grasp_final_inds
            ]

            if len(cur_grasps_final) == 0:
                self.logger.warning("No grasp IKs found!")
                self._reset()
                return None

            init_final_mask = (
                np.linalg.norm(
                    ik_cur_grasps_init - np.asarray(ik_cur_grasps_final),
                    axis=-1,
                    ord=np.inf,
                )
                < 0.25
            )
            # import pdb; pdb.set_trace()
            self.logger.debug("ik_time = {}".format(timer() - ik_time))
            self.logger.debug(
                "{}/{} grasps have ik solution".format(
                    init_final_mask.sum(), num_grasps
                )
            )

            # Check collisions for each grasp with self and with scene
            cfree_cur_grasps_init = np.asarray(cur_grasps_init)[
                init_final_mask
            ]
            cfree_ik_cur_grasps_init = np.asarray(ik_cur_grasps_init)[
                init_final_mask
            ]
            cfree_cur_grasps_final = np.asarray(cur_grasps_final)[
                init_final_mask
            ]
            cfree_ik_cur_grasps_final = np.asarray(ik_cur_grasps_final)[
                init_final_mask
            ]

            if isinstance(
                self.self_collision_checker, FCLSelfCollisionChecker
            ):
                self_coll_mask = np.zeros(
                    len(cfree_cur_grasps_init), dtype=np.bool
                )
                for i in range(len(cfree_ik_cur_grasps_init)):
                    self_coll_mask[i] = not self.self_collision_checker(
                        cfree_ik_cur_grasps_init[i]
                    )
            else:
                self_coll_mask = (
                    ~self.self_collision_checker(cfree_ik_cur_grasps_init)
                    .cpu()
                    .numpy()
                )
            cfree_cur_grasps_init = cfree_cur_grasps_init[self_coll_mask]
            cfree_cur_grasps_final = cfree_cur_grasps_final[self_coll_mask]
            cfree_ik_cur_grasps_init = cfree_ik_cur_grasps_init[self_coll_mask]
            cfree_ik_cur_grasps_final = cfree_ik_cur_grasps_final[
                self_coll_mask
            ]
            if len(cfree_cur_grasps_init) == 0:
                self.logger.warning("No grasps found!")
                self._reset()
                return None

            scene_coll_mask = (
                ~self.scene_collision_checker(
                    cfree_ik_cur_grasps_init,
                    threshold=0.45,
                )
                .cpu()
                .numpy()
            )

            cfree_cur_grasps_init = cfree_cur_grasps_init[scene_coll_mask]
            cfree_cur_grasps_final = cfree_cur_grasps_final[scene_coll_mask]
            cfree_ik_cur_grasps_init = cfree_ik_cur_grasps_init[
                scene_coll_mask
            ]
            cfree_ik_cur_grasps_final = cfree_ik_cur_grasps_final[
                scene_coll_mask
            ]

            if len(cur_grasps_init) == 0 or len(cfree_cur_grasps_init) == 0:
                self.logger.warning("No grasps found!")
                self._reset()
                return None

            self.cur_grasps_init = cfree_cur_grasps_init
            self.cur_grasps_final = cfree_cur_grasps_final
            self.ik_cur_grasps_init = cfree_ik_cur_grasps_init
            self.ik_cur_grasps_final = cfree_ik_cur_grasps_final
            self.ik_cur_grasps_init[:, -2:] = 0.04  # Set fingers open
            self.ik_cur_grasps_final[:, -2:] = 0.04

        # Reach toward the final grasp pose
        elif self.state == 1:
            closest_init_grasp_ind = np.linalg.norm(
                self.robot_q - self.ik_cur_grasps_init, ord=np.inf, axis=-1
            ).argmin()

            init_q = self.robot_q.copy()
            final_q = self.ik_cur_grasps_final[closest_init_grasp_ind]
            alpha = np.linspace(0, 1, self.lift_steps)[:, None]
            if self.reach_plan is None:
                self.reach_plan = (
                    alpha * final_q[None, :] + (1 - alpha) * init_q[None, :]
                )
            plan_ind = self._localize_in_plan(self.robot_q, self.reach_plan)
            index = min(plan_ind + 1, len(self.reach_plan) - 1)
            self.logger.debug(
                "index = {}, reach_plan = {}".format(
                    index, len(self.reach_plan)
                )
            )

            if index == len(self.reach_plan) - 1 or (
                self.prev_plan_ind is not None and self.prev_plan_ind >= index
            ):
                self.state = 2
                last_reach_plan = self.reach_plan[-1]
                self.reach_plan = None
                self.prev_plan_ind = None
                self.cur_grasps_init = None
                self.cur_grasps_final = None
                self.ik_cur_grasps_init = None
                self.ik_cur_grasps_final = None
                return [(last_reach_plan, True)]
            else:
                self.prev_plan_ind = index
            # precision reaching for last 10% of plan
            return list(
                zip(
                    self.reach_plan[index:],
                    [True if index > 0.5 * self.lift_steps else False]
                    * len(self.reach_plan[index:]),
                )
            )

        # Plan lift if at the final grasp pose
        elif self.state == 2:
            attempts = 0
            MAX_ATTEMPTS = 100
            while self.lift_plan is None and attempts < MAX_ATTEMPTS:
                lift_height_offset = (
                    self.lift_height
                    - self.robot.ee_pose[0, 2, 3].cpu().numpy()
                )
                self.lift_plan = self._compute_lift_plan(
                    self.robot_q,
                    [
                        [
                            -0.2 * attempts / MAX_ATTEMPTS,
                            0.2 * attempts / MAX_ATTEMPTS,
                        ],
                        [
                            -0.2 * attempts / MAX_ATTEMPTS,
                            0.2 * attempts / MAX_ATTEMPTS,
                        ],
                        [
                            lift_height_offset - 0.1 * attempts / MAX_ATTEMPTS,
                            lift_height_offset + 0.1 * attempts / MAX_ATTEMPTS,
                        ],
                    ],
                    num_plans=100,
                )
                if self.lift_plan is None:
                    attempts += 1
                else:
                    self.logger.info("found lift plan")

            # unable to lift (try again next time)
            if attempts == MAX_ATTEMPTS:
                self.logger.warning("Could not find lift plan")
                return [(self.robot_q.copy(), True)]
            plan_ind = self._localize_in_plan(self.robot_q, self.lift_plan)
            index = min(plan_ind + 1, len(self.lift_plan) - 1)
            self.logger.debug(
                "index = {}, lift_plan = {}".format(index, len(self.lift_plan))
            )

            if index == len(self.lift_plan) - 1:
                self.state = 3
                last_lift_plan = (self.lift_plan[-1], False)
                self.lift_plan = None
                return [last_lift_plan]

            # precision lifting for first 20% of plan
            return list(
                zip(
                    self.lift_plan[index:],
                    [True if index < 0.1 * self.lift_steps else False]
                    * len(self.lift_plan[index:]),
                )
            )

        # Plan placement when lifting is complete
        elif self.state == 3:
            if self.ik_cur_grasps_init is None:
                # Convert point mask into placement box
                if "placement_mask" in obs:
                    # project points onto table plane
                    pts = tra.transform_points(
                        obs["pc"][obs["placement_mask"] > 0],
                        obs["robot_to_model"] @ obs["camera_pose"],
                    )
                    cvx_hull = scipy.spatial.Delaunay(
                        pts[:, :2], qhull_options="QbB Pp"
                    )
                    in_placement_box = lambda p: torch.from_numpy(
                        cvx_hull.find_simplex(p.cpu().numpy()) >= 0
                    ).to(self.device)
                # distance condition
                else:
                    # Limit placements to different area of workspace
                    in_placement_box = (
                        lambda p: torch.norm(
                            p
                            - torch.from_numpy(
                                self.scene_collision_checker.obj_pc[
                                    :, :2
                                ].mean(axis=0)
                            ).to(self.device),
                            dim=-1,
                        )
                        > 0.25
                    )
                    # Otherwise, assume whole space is fair game
                    # in_placement_box = lambda p: torch.ones(
                    #     len(p), dtype=torch.bool, device=self.device
                    # )

                NUM_IK_ATTEMPTS = 20
                cur_places = []
                for attempt in range(NUM_IK_ATTEMPTS):

                    # Sample candidate placements within the workspace (model frame)
                    placement_pts = (
                        self.scene_collision_checker.sample_in_bounds(
                            num=10000, offset=0.05
                        )
                    )
                    placement_pts[:, 2] = (
                        torch.empty(
                            len(placement_pts),
                            device=self.device,
                            dtype=torch.float32,
                        ).uniform_(0.0, 0.3)
                        + 0.3
                    )
                    placement_mask = in_placement_box(placement_pts[:, :2])
                    placement_mask &= (
                        ~self.scene_collision_checker.check_object_collisions(
                            placement_pts
                        )
                    )
                    placement_pts = placement_pts[placement_mask]
                    if len(placement_pts) == 0:
                        self.logger.debug(
                            "No sampled placements are collision-free {:d}/{:d}".format(
                                attempt, NUM_IK_ATTEMPTS
                            )
                        )
                        continue

                    placement_pts = placement_pts[
                        torch.argsort(placement_pts[:, 2])
                    ]

                    placement_pts_robot = tra.transform_points(
                        placement_pts.cpu().numpy(),
                        self.scene_collision_checker.model_to_robot.cpu().numpy(),
                    )
                    self.robot.set_joint_cfg(self.robot_q)
                    placement_poses = (
                        self.robot.ee_pose.repeat(100, 1, 1).cpu().numpy()
                    )
                    placement_poses[:, :3, 3] = (
                        placement_pts_robot[:100] + self.ee_offset
                    )
                    placement_poses = placement_poses[
                        : len(placement_pts_robot[:100])
                    ]
                    self.placement_poses = placement_poses.copy()

                    # Get IK solutions for lowest points
                    ik_time = timer()
                    for i, p in enumerate(placement_poses):
                        self.ik_procs[i % self.num_proc].ik(
                            p, self.robot_q, ind=i
                        )
                        for _ in range(4):
                            rand_q = np.random.uniform(
                                self.robot.min_joints.cpu().numpy(),
                                self.robot.max_joints.cpu().numpy(),
                            )
                            self.ik_procs[i % self.num_proc].ik(
                                p, rand_q.reshape(-1), ind=i
                            )

                    # collect computed iks
                    place_inds = []
                    cur_places = []
                    ik_cur_places = []
                    for _ in range(len(placement_poses) * 5):
                        output = self.output_queue.get(True)
                        assert isinstance(output, tuple)
                        assert len(output) == 2
                        if output[1] is None:
                            continue
                        place_inds.append(output[0])
                        ik_cur_places.append(output[1])
                        cur_places.append(placement_poses[place_inds[-1]])

                    self.logger.debug("ik_time = {}".format(timer() - ik_time))
                    self.logger.debug(
                        "{}/{} placements have ik solution".format(
                            len(ik_cur_places), len(placement_poses)
                        )
                    )

                    if len(ik_cur_places) == 0:
                        continue

                    place_inds = np.asarray(place_inds)
                    place_inds = (
                        place_inds[None, :] == np.unique(place_inds)[:, None]
                    ).argmax(1)
                    ik_cur_places = np.asarray(ik_cur_places)[place_inds]
                    cur_places = np.asarray(cur_places)[place_inds]
                    ik_cur_places[:, -2:] = 0.0

                    if len(ik_cur_places) > 0:
                        break

                if len(cur_places) == 0:
                    self.logger.warning("No placements IKs found!")
                    self.state = 4
                    closed_q = self.robot_q.copy()
                    closed_q[-2:] = 0.0
                    return [(closed_q, True)]
                self.cur_grasps_final = cur_places
                self.ik_cur_grasps_final = ik_cur_places

            # Get robot collisions for placements
            coll_mask = self.scene_collision_checker(
                self.ik_cur_grasps_final
            ) | self.self_collision_checker(self.ik_cur_grasps_final)
            coll_mask |= self.scene_collision_checker.check_object_collisions(
                torch.from_numpy(
                    tra.transform_points(
                        self.cur_grasps_final[:, :3, 3] - self.ee_offset,
                        self.scene_collision_checker.robot_to_model.cpu().numpy(),
                    )
                )
                .float()
                .to(self.device)
            )
            self.ik_cur_grasps_init = self.ik_cur_grasps_final[
                ~coll_mask.cpu().numpy()
            ]
            self.cur_grasps_init = self.cur_grasps_final[
                ~coll_mask.cpu().numpy()
            ]
            if len(self.ik_cur_grasps_init) == 0:
                self.logger.warning("No collision-free placements found!")
                self.state = 4
                closed_q = self.robot_q.copy()
                closed_q[-2:] = 0.0
                return [(closed_q, True)]

        # Plan lift if at the final placement pose
        elif self.state == 4:
            attempts = 0
            MAX_ATTEMPTS = 100
            while self.lift_plan is None and attempts < MAX_ATTEMPTS:
                lift_height_offset = (
                    self.lift_height
                    - self.robot.ee_pose[0, 2, 3].cpu().numpy()
                )
                self.lift_plan = self._compute_lift_plan(
                    self.robot_q,
                    [
                        [
                            -0.2 * attempts / MAX_ATTEMPTS,
                            0.2 * attempts / MAX_ATTEMPTS,
                        ],
                        [
                            -0.2 * attempts / MAX_ATTEMPTS,
                            0.2 * attempts / MAX_ATTEMPTS,
                        ],
                        [
                            lift_height_offset - 0.1 * attempts / MAX_ATTEMPTS,
                            lift_height_offset + 0.1 * attempts / MAX_ATTEMPTS,
                        ],
                    ],
                    num_plans=100,
                )
                if self.lift_plan is None:
                    attempts += 1
                else:
                    self.logger.info("found lift plan")
                    self.lift_plan[:, -2:] = 0.04

            # unable to lift, try again next time
            if attempts == MAX_ATTEMPTS:
                self.logger.warning("Could not find lift plan")
                return [(self.robot_q.copy(), True)]
            plan_ind = self._localize_in_plan(self.robot_q, self.lift_plan)
            index = min(plan_ind + 1, len(self.lift_plan) - 1)
            self.logger.debug(
                "index = {}, lift_plan = {}".format(index, len(self.lift_plan))
            )

            # We are done, go back to reaching for the next grasp
            if index == len(self.lift_plan) - 1:
                self.cur_grasps_init = None
                self.cur_grasps_final = None
                self.ik_cur_grasps_init = None
                self.ik_cur_grasps_final = None
                last_lift_plan = (self.lift_plan[-1], False)
                self.lift_plan = None
                self.state = 0
                self._reset()
                return None

            return list(
                zip(
                    self.lift_plan[index:],
                    [True if index < 0.1 * self.lift_steps else False]
                    * len(self.lift_plan[index:]),
                )
            )

        self.logger.debug(
            "Joint Dist to Target: {:.4f}".format(
                np.linalg.norm(
                    self.robot_q[:7] - self.ik_cur_grasps_init[:, :7],
                    ord=np.inf,
                    axis=-1,
                ).min()
            )
        )

        # Check to see if we are within threshold of our goal; if so, transition to next state
        if (self.state == 0 or self.state == 3) and np.linalg.norm(
            self.robot_q - self.ik_cur_grasps_init, ord=np.inf, axis=-1
        ).min() < self.transition_threshold:
            self.state += 1
            return [(self.robot_q.copy(), True)]

        # Collect rollouts and find best collision free trajectory
        self.logger.info("collecting data")
        rollouts, rewards = self._collect_rollouts()
        collisions = self._check_collisions(
            rollouts, check_obj=(self.state == 3)
        )

        rollout_values, rollout_lengths = self._trim_rollouts(
            rewards, collisions
        )

        best_rollout_val, best_rollout_ind = rollout_values.max(dim=0)
        self.logger.debug(
            f"==============> best_rollout_val {best_rollout_val}"
        )
        best_rollout_len = rollout_lengths[best_rollout_ind]
        self.prev_rollout_lens.append(best_rollout_len)
        if len(self.prev_rollout_lens) > 10:
            self.prev_rollout_lens.pop(0)
            if sum(self.prev_rollout_lens) == 0:
                self.state = 4  # enter recovery policy
        best_rollout = (
            rollouts[best_rollout_ind, : best_rollout_len + 1].cpu().numpy()
        )
        self.logger.info("reward = {:.4f}".format(best_rollout_val))
        self.logger.info("best rollout length = {:d}".format(best_rollout_len))

        self.rollouts = (rollouts, rollout_values, rollout_lengths)
        return best_rollout
