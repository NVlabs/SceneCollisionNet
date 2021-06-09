import itertools
from timeit import default_timer as timer

import numpy as np
import trimesh.transformations as tra
from autolab_core import CameraIntrinsics
from torch.utils.data import IterableDataset
from trimesh.collision import CollisionManager
from urdfpy import URDF

from .scene import SceneManager, SceneRenderer
from .utils import compute_camera_pose


class IterableSceneCollisionDataset(IterableDataset):
    def __init__(
        self,
        meshes,
        batch_size,
        query_size,
        intrinsics,
        extrinsics,
        bounds,
        n_obj_points,
        n_scene_points,
        rotations=True,
        trajectories=1,
    ):
        self.meshes = meshes
        self.batch_size = batch_size
        self.query_size = query_size
        self.cam_intr = CameraIntrinsics(**intrinsics)
        self.cam_pose = extrinsics
        self.bounds = np.array(bounds)

        self.n_obj_points = n_obj_points
        self.n_scene_points = n_scene_points
        self.rotations = rotations
        self.trajectories = trajectories

    # Generator that yields batches of training tuples
    def __iter__(self):
        """
        Generator that yields batches of training tuples
        Outputs:
            scene_points (self.batch_size, self.n_scene_points, 3): scene point cloud batch
            obj_points (self.n_obj_points, 3): object point cloud batch
            trans (self.query_size, 3): translations of object in scene
            rots (self.query_size, 6): rotations of object in scene (first two cols of rotation matrix)
            colls (self.query_size,): boolean array of GT collisions along trajectories
            scene_manager (utils.SceneManager): underlying scene manager for GT collisions
        """
        while True:
            (
                obj_points,
                obj_centroid,
                obj_mesh,
                obj_pose,
                camera_pose,
            ) = self.get_obj()
            scene_points, scene_manager = self.get_scene(
                camera_pose=camera_pose
            )
            np.random.seed()
            trans, rots, colls = self.get_colls(
                scene_manager,
                obj_mesh,
                obj_pose,
                obj_centroid,
            )
            del scene_manager

            yield scene_points, obj_points, trans, rots, colls

    def get_scene(self, low=10, high=20, camera_pose=None):
        """
        Generate a scene point cloud by placing meshes on a tabletop
        Inputs:
            low (int): minimum number of objects to place
            high (int): maximum number of objects to place
            camera_pose (4, 4): optional camera pose
        Outputs:
            points_batch (self.batch_size, self.n_scene_points, 3): scene point cloud batch
            scene_manager (utils.SceneManager): underlying scene manager for GT collisions
        """
        # Create scene with random number of objects btw low and high
        num_objs = np.random.randint(low, high)
        scene_manager = self._create_scene()
        scene_manager.arrange_scene(num_objs)

        # Render points from batch_size different angles
        points_batch = np.zeros(
            (self.batch_size, self.n_scene_points, 3), dtype=np.float32
        )
        for i in range(self.batch_size):
            if camera_pose is None:
                camera_pose = self.sample_camera_pose()
            scene_manager.camera_pose = camera_pose

            points = scene_manager.render_points()
            points = points[
                (points[:, :3] > self.bounds[0] + 1e-4).all(axis=1)
            ]
            points = points[
                (points[:, :3] < self.bounds[1] - 1e-4).all(axis=1)
            ]
            pt_inds = np.random.choice(
                points.shape[0], size=self.n_scene_points
            )
            sample_points = points[pt_inds]
            points_batch[i] = sample_points

        return points_batch, scene_manager

    def get_obj(self):
        """
        Generate an object point cloud
        Outputs:
            points_batch (self.n_obj_points, 3): object point cloud batch
            points_center (3,): centroid of object point cloud
            obj_mesh (trimesh.Trimesh): object underlying mesh
            pose (4, 4): GT object stable pose (w/ z rotation)
            camera_pose (4, 4): Pose matrix of the camera
        """
        obj_scene = self._create_scene()
        camera_pose = self.sample_camera_pose()
        obj_scene.camera_pose = camera_pose
        points = np.array([])
        while not points.any():
            obj_scene.reset()
            obj_mesh, obj_info = obj_scene.sample_obj()
            if obj_mesh is None:
                continue
            stps = obj_info["stps"]
            probs = obj_info["probs"]
            pose = stps[np.random.choice(len(stps), p=probs)].copy()
            z_rot = tra.rotation_matrix(
                2 * np.pi * np.random.rand(), [0, 0, 1], point=pose[:3, 3]
            )
            pose = z_rot @ pose
            obj_scene.add_object("obj", obj_mesh, pose)
            points = obj_scene.render_points()

        pt_inds = np.random.choice(
            points.shape[0], size=self.n_obj_points, replace=True
        )
        points_batch = np.repeat(
            points[None, pt_inds], self.batch_size, axis=0
        )
        points_center = np.mean(points_batch[0, :, :3], axis=0)
        del obj_scene

        return points_batch, points_center, obj_mesh, pose, camera_pose

    def get_colls(
        self,
        scene_manager,
        obj,
        obj_pose,
        obj_centroid,
    ):
        """
        Generate object/scene collision trajectories
        Inputs:
            scene_manager (utils.SceneManager): Underlying scene manager
            obj (trimesh.Trimesh): Underlying object mesh
            obj_pose (4, 4): Object GT pose matrix
            obj_centroid (3,): Centroid of object point cloud
        Outputs:
            trans (self.query_size, 3): translations of object in scene
            rots (self.query_size, 6): rotations of object in scene (first two cols of rotation matrix)
            colls (self.query_size,): boolean array of GT collisions along trajectories
        """
        # Generate trajectory translations and clip to workspace bounds
        trans_start, trans_end = np.random.uniform(
            self.bounds[0],
            self.bounds[1],
            size=(2, self.trajectories, len(self.bounds[0])),
        )
        trans = (
            np.linspace(
                trans_start, trans_end, self.query_size // self.trajectories
            )
            .transpose(1, 0, 2)
            .reshape(self.query_size, 3)
        )
        trans[trans < self.bounds[0] + 1e-4] = (
            self.bounds[0, np.where(trans < self.bounds[0] + 1e-4)[1]] + 1e-4
        )
        trans[trans > self.bounds[1] - 1e-4] = (
            self.bounds[1, np.where(trans > self.bounds[1] - 1e-4)[1]] - 1e-4
        )

        # Sample trajectory rotations (if desired) and interpolate
        mesh_trans = trans - (obj_centroid - obj_pose[:3, 3])
        mesh_tfs = np.repeat(np.eye(4)[None, ...], self.query_size, axis=0)
        mesh_tfs[:, :3, 3] = mesh_trans
        if self.rotations:
            rots = np.random.randn(2 * self.trajectories, 2, 3).astype(
                np.float32
            )
            b1 = rots[:, 0] / np.linalg.norm(
                rots[:, 0], axis=-1, keepdims=True
            )
            b2 = (
                rots[:, 1]
                - np.einsum("ij,ij->i", b1, rots[:, 1])[:, None] * b1
            )
            b2 /= np.linalg.norm(b2, axis=1, keepdims=True)
            b3 = np.cross(b1, b2)
            rot_mats = np.stack((b1, b2, b3), axis=-1)
            step = self.query_size // self.trajectories
            for i in range(self.trajectories):
                quats = [
                    tra.quaternion_from_matrix(rm)
                    for rm in rot_mats[2 * i : 2 * (i + 1)]
                ]
                d = np.dot(*quats)
                if d < 0.0:
                    d = -d
                    np.negative(quats[1], quats[1])
                ang = np.arccos(d)
                t = np.linspace(0, 1, step, endpoint=True)
                quats_slerp = quats[0][None, :] * np.sin((1.0 - t) * ang)[
                    :, None
                ] / np.sin(ang) + quats[1][None, :] * np.sin(t * ang)[
                    :, None
                ] / np.sin(
                    ang
                )
                mesh_tfs[
                    i * step : (i + 1) * step, :3, :3
                ] = tra.quaternion_matrix(quats_slerp)[:, :3, :3]
            rots = mesh_tfs[:, :3, :2].transpose(0, 2, 1)
        else:
            rots = np.repeat(
                np.eye(3, dtype=np.float32)[None, :, :2],
                self.query_size,
                axis=0,
            ).transpose(0, 2, 1)

        # Apply to object within scene and find GT collision status
        new_obj_poses = mesh_tfs @ obj_pose
        colls = np.zeros(self.query_size, dtype=np.bool)
        scene_manager._collision_manager.add_object("query_obj", obj)
        for i in range(self.query_size):
            scene_manager._collision_manager.set_transform(
                "query_obj", new_obj_poses[i]
            )
            colls[i] = scene_manager.collides()
        scene_manager._collision_manager.remove_object("query_obj")
        return (
            trans,
            rots.reshape(self.query_size, -1),
            np.repeat(
                colls.reshape(1, self.query_size), self.batch_size, axis=0
            ),
        )

    # Creates scene manager and renderer
    def _create_scene(self):
        r = SceneRenderer()
        r.create_camera(self.cam_intr, znear=0.04, zfar=5)
        s = SceneManager(self.meshes, renderer=r)
        return s

    # Samples a camera pose that looks at the center of the scene
    def sample_camera_pose(self, mean=False):
        if mean:
            az = np.mean(self.cam_pose["azimuth"])
            elev = np.mean(self.cam_pose["elevation"])
            radius = np.mean(self.cam_pose["radius"])
        else:
            az = np.random.uniform(*self.cam_pose["azimuth"])
            elev = np.random.uniform(*self.cam_pose["elevation"])
            radius = np.random.uniform(*self.cam_pose["radius"])

        sample_pose, _ = compute_camera_pose(radius, az, elev)

        return sample_pose


class BenchmarkSceneCollisionDataset(IterableSceneCollisionDataset):
    def __init__(
        self,
        meshes,
        batch_size,
        query_size,
        intrinsics,
        extrinsics,
        bounds,
        n_obj_points=1024,
        n_scene_points=4096,
        rotations=True,
        trajectories=0,
        vis=True,
        **kwargs
    ):
        super().__init__(
            meshes,
            batch_size,
            query_size,
            intrinsics,
            extrinsics,
            bounds,
            n_obj_points,
            n_scene_points,
            rotations=rotations,
            trajectories=trajectories,
        )
        self.vis = vis

    # Generator that yields batches of training tuples
    def __iter__(self):
        while True:
            (
                obj_points,
                obj_centroid,
                obj_mesh,
                obj_pose,
                camera_pose,
            ) = self.get_obj()
            scene_points, scene_manager = self.get_scene(
                camera_pose=camera_pose
            )
            np.random.seed()
            coll_start = timer()
            trans, rots, colls = self.get_colls(
                scene_manager,
                obj_mesh,
                obj_pose,
                obj_centroid,
            )
            coll_time = timer() - coll_start

            if not self.vis:
                del scene_manager

            if self.vis:
                yield scene_manager, obj_mesh, obj_pose, scene_points, obj_points, trans, rots, colls, coll_time
            else:
                yield scene_points, obj_points, trans, rots, colls, coll_time


class IterableRobotCollisionDataset(IterableDataset):
    def __init__(self, robot_urdf, batch_size):
        self.robot = URDF.load(robot_urdf)
        self.batch_size = batch_size

        # Get link poses for a series of random configurations
        low_joint_limits, high_joint_limits = self.robot.joint_limit_cfgs
        self.low_joint_vals = np.fromiter(
            low_joint_limits.values(), dtype=float
        )
        self.high_joint_vals = np.fromiter(
            high_joint_limits.values(), dtype=float
        )

        meshes = self.robot.collision_trimesh_fk().keys()
        self.link_meshes = list(meshes)
        self.num_links = len(self.link_meshes)
        self.link_combos = list(itertools.combinations(range(len(meshes)), 2))

        # Add the meshes to the collision managers
        self.collision_managers = []
        for m in self.link_meshes:
            collision_manager = CollisionManager()
            collision_manager.add_object("link", m)
            self.collision_managers.append(collision_manager)

        # Preprocess - find some collision-free configs and set min distances
        rand_cfgs = (
            np.random.rand(self.batch_size, len(self.low_joint_vals))
            * (self.high_joint_vals - self.low_joint_vals)
            + self.low_joint_vals
        )
        mesh_poses = self.robot.collision_trimesh_fk_batch(rand_cfgs)
        colls = np.zeros(self.batch_size, dtype=np.bool)
        for k in range(self.batch_size):
            colls[k] = self.check_pairwise_distances(
                mesh_poses, k, boolean=True
            )

        self.original_dists = np.inf * np.ones(
            (self.num_links, self.num_links)
        )
        for k, c in enumerate(colls):
            if not c:
                dists = self.check_pairwise_distances(
                    mesh_poses, k, normalize=False
                )
                self.original_dists = np.minimum(self.original_dists, dists)

    # Generator that yields batches of training tuples
    def __iter__(self):
        while True:
            rand_cfgs = (
                np.random.rand(self.batch_size, len(self.low_joint_vals))
                * (self.high_joint_vals - self.low_joint_vals)
                + self.low_joint_vals
            )
            mesh_poses = self.robot.collision_trimesh_fk_batch(rand_cfgs)
            colls = np.zeros(self.batch_size)
            for k in range(self.batch_size):
                colls[k] = np.sum(self.check_pairwise_distances(mesh_poses, k))

            yield rand_cfgs.reshape(self.batch_size, -1), colls

    def check_pairwise_distances(
        self, mesh_poses, ind, boolean=False, normalize=False
    ):
        if boolean:
            coll = False
        else:
            dists = np.zeros((self.num_links, self.num_links))
        for _, (i, j) in enumerate(self.link_combos):
            if abs(i - j) < 2 or ((i, j) == (6, 8)) or ((i, j) == (8, 10)):
                continue
            i_tf = mesh_poses[self.link_meshes[i]][ind]
            self.collision_managers[i].set_transform("link", i_tf)
            j_tf = mesh_poses[self.link_meshes[j]][ind]
            self.collision_managers[j].set_transform("link", j_tf)
            if boolean:
                coll |= self.collision_managers[i].in_collision_other(
                    self.collision_managers[j]
                )
                if coll:
                    return coll
            else:
                dists[i, j] = dists[j, i] = self.get_dist_pair(
                    i, j, normalize=normalize
                )
        if boolean:
            return coll
        return dists

    def get_dist_pair(self, i, j, normalize=False):
        min_dist = self.collision_managers[i].min_distance_other(
            self.collision_managers[j]
        )
        if normalize:
            min_dist = np.e ** (
                -2.0 * (min_dist / self.original_dists[i, j]) ** 2
            )
        return min_dist
