import argparse
import random

import numpy as np
from autolab_core import Logger

from scenecollisionnet.policy import MPPIPolicy, RobotEnvironment


def rollout(args):
    logger = Logger.get_logger("tools/rollout_policy.py")

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    env = RobotEnvironment(args)
    policy = MPPIPolicy(
        num_proc=args.mppi_nproc,
        num_path=args.mppi_num_rollouts,
        horizon=args.mppi_horizon,
        collision_steps=args.mppi_collision_steps,
        max_step=args.mppi_max_step,
        lift_height=args.mppi_lift_height,
        lift_steps=args.mppi_lift_steps,
        transition_threshold=args.mppi_transition_threshold,
        noise_std=args.mppi_q_noise_std,
        self_coll_nn=args.self_coll_nn,
        scene_coll_nn=args.scene_coll_nn,
        cam_type=args.mppi_cam,
        device=args.compute_device_id,
        log_file=args.log_file,
    )

    input("press any key to start")
    o, info = env.step()
    while True:
        a = policy.get_action(o)
        next_o, info = env.step(a)
        if info["done"]:
            break

        o = next_o

    logger.info(f"Finished {2 * args.num_objects} attempts. Exiting...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--compute-device-id", type=int, default=0)
    parser.add_argument("--graphics-device-id", type=int, default=1)
    parser.add_argument(
        "--robot-asset-root",
        type=str,
        default="extern/isaacgym/assets/",
    )
    parser.add_argument(
        "--robot-asset-file",
        type=str,
        default="urdf/franka_description/robots/franka_panda.urdf",
    )
    parser.add_argument("--ee-link-name", type=str, default="panda_hand")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="datasets/shapenet",
    )
    parser.add_argument(
        "--urdf-cache", type=str, default="datasets/scene_cache"
    )
    parser.add_argument("--num-objects", type=int, default=10)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-width", type=int, default=640)

    parser.add_argument("--seed", type=int)
    parser.add_argument("--control-frequency", type=float, default=20)
    parser.add_argument("--npoints", type=int, default=8192)
    parser.add_argument("--mppi-nproc", type=int, default=20)
    parser.add_argument("--mppi-horizon", type=int, default=40)
    parser.add_argument("--mppi-num-rollouts", type=int, default=200)
    parser.add_argument("--mppi-q-noise-std", type=float, default=0.08)
    parser.add_argument("--mppi-max-step", type=float, default=0.05)
    parser.add_argument("--mppi-lift-height", type=float, default=0.6)
    parser.add_argument("--mppi-lift-steps", type=int, default=20)
    parser.add_argument("--mppi-collision-steps", type=int, default=10)
    parser.add_argument(
        "--mppi-transition-threshold", type=float, default=0.05
    )
    parser.add_argument("--mppi-cam", type=str, default="ws")
    parser.add_argument("--scene-coll-nn", type=str)
    parser.add_argument("--self-coll-nn", type=str)
    parser.add_argument("--log-file", type=str)

    args = parser.parse_args()
    rollout(args)
