import abc
import itertools
import os
import os.path as osp

import numpy as np
import torch
import torch_scatter
import trimesh
import trimesh.transformations as tra
from knn_cuda import KNN

try:
    import kaolin as kal
except ImportError:
    print("skipping kaolin import, only needed for baselines")
import multiprocessing as mp
import queue as Queue

import ruamel.yaml as yaml

from ..collision_models import RobotCollisionNet, SceneCollisionNet


class CollisionChecker(abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def __call__(self):
        pass


class SceneCollisionChecker(CollisionChecker):
    def __init__(self, robot, use_knn=True, **kwargs):
        super().__init__(**kwargs)
        self.robot = robot
        self.device = self.robot.device
        if use_knn:
            self.knn = KNN(k=8, transpose_mode=True).to(self.device)
        else:
            self.knn = None
        self.cur_scene_pc = None
        self.robot_to_model = None
        self.model_to_robot = None

    def set_scene(self, obs):
        orig_scene_pc = obs["pc"]
        scene_labels = obs["pc_label"]
        label_map = obs["label_map"]

        # Remove robot points plus excluded
        self.scene_pc_mask = np.logical_and(
            scene_labels != label_map["robot"],
            scene_labels != label_map["target"],
        )

        # Transform into robot frame (z up)
        self.camera_pose = obs["camera_pose"]
        self.scene_pc = tra.transform_points(orig_scene_pc, self.camera_pose)

    def set_object(self, obs):
        scene_pc = obs["pc"]
        scene_labels = obs["pc_label"]
        label_map = obs["label_map"]
        obj_pc = scene_pc[scene_labels == label_map["target"]]

        self.obj_pc = tra.transform_points(
            obj_pc,
            obs["camera_pose"],
        )

    def _aggregate_pc(self, cur_pc, new_pc):

        # Filter tiny clusters of points and split into vis/occluding
        cam_model_tr = (
            torch.from_numpy(self.camera_pose[:3, 3]).float().to(self.device)
        )
        new_pc = torch.from_numpy(new_pc).float().to(self.device)
        if self.knn is not None:
            nearest = self.knn(new_pc[None, ...], new_pc[None, ...])[0][
                0, :, -1
            ]
        vis_mask = torch.from_numpy(self.scene_pc_mask).to(self.device)
        dists = torch.norm(new_pc - cam_model_tr, dim=1) ** 2
        dists /= dists.max()
        if self.knn is None:
            nearest = torch.zeros_like(vis_mask)
        occ_scene_pc = new_pc[~vis_mask & (nearest < 0.1 * dists)]
        scene_pc = new_pc[vis_mask & (nearest < 0.1 * dists)]

        if cur_pc is not None:
            if self.knn is None:
                pass
            else:
                dists = torch.norm(cur_pc - cam_model_tr, dim=1) ** 2
                dists /= dists.max()

                nearest = self.knn(cur_pc[None, ...], cur_pc[None, ...])[0][
                    0, :, -1
                ]
                cur_pc = cur_pc[nearest < 0.1 * dists].float()

            # Group points by rays; get mapping from points to unique rays
            cur_pc_rays = cur_pc - cam_model_tr
            cur_pc_dists = torch.norm(cur_pc_rays, dim=1, keepdim=True) + 1e-12
            cur_pc_rays /= cur_pc_dists
            occ_pc_rays = occ_scene_pc - cam_model_tr
            occ_pc_dists = torch.norm(occ_pc_rays, dim=1, keepdim=True) + 1e-12
            occ_pc_rays /= occ_pc_dists
            occ_rays = (
                (torch.cat((cur_pc_rays, occ_pc_rays), dim=0) * 50)
                .round()
                .long()
            )
            _, occ_uniq_inv, occ_uniq_counts = torch.unique(
                occ_rays, dim=0, return_inverse=True, return_counts=True
            )

            # Build new point cloud from previous now-occluded points and new pc
            cur_occ_inv = occ_uniq_inv[: len(cur_pc_rays)]
            cur_occ_counts = torch.bincount(
                cur_occ_inv, minlength=len(occ_uniq_counts)
            )
            mean_occ_dists = torch_scatter.scatter_max(
                occ_pc_dists.squeeze(),
                occ_uniq_inv[-len(occ_pc_rays) :],
                dim_size=occ_uniq_inv.max() + 1,
            )[0]
            occ_mask = (occ_uniq_counts > cur_occ_counts) & (
                cur_occ_counts > 0
            )
            occ_pc = cur_pc[
                occ_mask[cur_occ_inv]
                & (cur_pc_dists.squeeze() > mean_occ_dists[cur_occ_inv] + 0.01)
            ]
            return torch.cat((occ_pc, scene_pc), dim=0)
        else:
            return scene_pc

    def _compute_model_tfs(self, obs):
        if "robot_to_model" in obs:
            self.robot_to_model = obs["robot_to_model"]
            self.model_to_robot = obs["model_to_robot"]
            assert len(self.robot_to_model) > 0
            assert len(self.model_to_robot) > 0
        else:
            scene_labels = obs["pc_label"]
            label_map = obs["label_map"]

            # Extract table transform from points
            tab_pts = self.scene_pc[scene_labels == label_map["table"]]
            if len(tab_pts) == 0:
                tab_pts = self.scene_pc[scene_labels == label_map["objs"]]
            tab_height = tab_pts.mean(axis=0)[2]
            tab_tf_2d = trimesh.bounds.oriented_bounds_2D(tab_pts[:, :2])[0]
            tab_tf = np.eye(4)
            tab_tf[:3, :2] = tab_tf_2d[:, :2]
            tab_tf[:3, 3] = np.append(tab_tf_2d[:2, 2], 0.3 - tab_height)

            # Fix "long" side of table by rotating
            self.robot_to_model = tra.euler_matrix(0, 0, np.pi / 2) @ tab_tf
            if self.robot_to_model[0, 0] < 0:
                self.robot_to_model = (
                    tra.euler_matrix(0, 0, -np.pi) @ self.robot_to_model
                )
            self.model_to_robot = np.linalg.inv(self.robot_to_model)

        self.robot_to_model = (
            torch.from_numpy(self.robot_to_model).float().to(self.device)
        )
        self.model_to_robot = (
            torch.from_numpy(self.model_to_robot).float().to(self.device)
        )


class FCLSelfCollisionChecker(CollisionChecker):
    def __init__(self, robot):
        self.robot = robot
        self._link_combos = list(
            itertools.combinations(robot.link_map.keys(), 2)
        )
        self._link_managers = {}
        for link in robot.links:
            if link.collision_mesh is not None:
                self._link_managers[
                    link.name
                ] = trimesh.collision.CollisionManager()
                self._link_managers[link.name].add_object(
                    link.name, link.collision_mesh
                )

    def set_allowed_collisions(self, l1, l2):
        if (l1, l2) in self._link_combos:
            self._link_combos.remove((l1, l2))
        elif (l2, l1) in self._link_combos:
            self._link_combos.remove((l2, l1))

    def __call__(self, q):
        self.robot.set_joint_cfg(q)
        for (i, j) in self._link_combos:
            if i not in self._link_managers or j not in self._link_managers:
                continue
            i_tf = (
                self.robot.link_poses[self.robot.link_map[i]]
                .squeeze()
                .cpu()
                .numpy()
            )
            self._link_managers[i].set_transform(i, i_tf)
            j_tf = (
                self.robot.link_poses[self.robot.link_map[j]]
                .squeeze()
                .cpu()
                .numpy()
            )
            self._link_managers[j].set_transform(j, j_tf)
            if self._link_managers[i].in_collision_other(
                self._link_managers[j]
            ):
                return True
        return False


class FCLSceneCollisionChecker(SceneCollisionChecker):
    def __init__(self, robot, use_scene_pc=False):
        super().__init__(robot=robot)
        self._use_scene_pc = use_scene_pc
        self.robot_manager = trimesh.collision.CollisionManager()

        for link in robot.mesh_links:
            self.robot_manager.add_object(link.name, link.collision_mesh)

    def set_scene(self, obs, scene=None):
        if self._use_scene_pc:
            super().set_scene(obs)
            self.cur_scene_pc = self._aggregate_pc(
                self.cur_scene_pc, self.scene_pc
            )
            pc = trimesh.PointCloud(self.scene_pc)
            scene_mesh = trimesh.voxel.ops.points_to_marching_cubes(
                pc.vertices, pitch=0.01
            )
            self.scene_manager = trimesh.collision.CollisionManager()
            self.scene_manager.add_object("scene", scene_mesh)
        else:
            self.scene_manager = scene

    def set_object(self, obs, scene=None):
        if self._use_scene_pc:
            super().set_scene(obs)
            pc = trimesh.PointCloud(self.scene_pc)
            pitch = pc.extents.max() / 100
            scene_mesh = trimesh.voxel.ops.points_to_marching_cubes(
                pc.vertices, pitch=pitch
            )
            self.scene_manager = trimesh.collision.CollisionManager()
            self.scene_manager.add_object("scene", scene_mesh)
        else:
            self.scene_manager = scene

    def __call__(self, q, by_link=False, threshold=None):
        if q is None or len(q) == 0:
            return np.zeros((len(self.robot.mesh_links), 0), dtype=np.bool)
        coll = (
            np.zeros((len(self.robot.mesh_links), len(q)), dtype=np.bool)
            if by_link
            else np.zeros(len(q), dtype=np.bool)
        )
        self.robot.set_joint_cfg(q)
        for i in range(len(q)):
            for link in self.robot.mesh_links:
                pose = self.robot.link_poses[link][i].squeeze().cpu().numpy()
                import pdb

                pdb.set_trace()
                self.robot_manager.set_transform(link.name, pose)

            coll_q = self.robot_manager.in_collision_other(
                self.scene_manager, return_names=by_link
            )
            if by_link:
                for j, link in enumerate(self.robot.mesh_links):
                    coll[j, i] = np.any(
                        [link.name in pair for pair in coll_q[1]]
                    )
            else:
                coll[i] = coll_q
        return coll


class FCLProc(mp.Process):
    """
    Used for finding collisions in parallel using FCL.
    """

    def __init__(self, links, output_queue, use_scene_pc=True):
        """
        Args:
        output_queue: mp.Queue, the queue that all the output data
            that is computed is added to.
        """
        super().__init__()
        self.links = links
        self.output_queue = output_queue
        self.input_queue = mp.Queue()
        self.use_scene_pc = use_scene_pc

    def _collides(self, link_poses, inds, by_link):
        """computes collisions."""
        coll = (
            np.zeros((len(self.links), len(inds)), dtype=np.bool)
            if by_link
            else np.zeros(len(inds), dtype=np.bool)
        )
        for k, i in enumerate(inds):
            for link in self.links:
                pose = link_poses[link.name][i].squeeze()
                self.robot_manager.set_transform(link.name, pose)

            coll_q = self.robot_manager.in_collision_other(
                self.scene_manager, return_names=by_link
            )
            if by_link:
                for j, link in enumerate(self.links):
                    coll[j, k] = np.any(
                        [link.name in pair for pair in coll_q[1]]
                    )
            else:
                coll[k] = coll_q
        return coll

    def _object_collides(self, poses, inds):
        """computes collisions."""
        coll = np.zeros(len(inds), dtype=np.bool)
        for k, i in enumerate(inds):
            self.object_manager.set_transform("obj", poses[i])
            coll[k] = self.object_manager.in_collision_other(
                self.scene_manager
            )
        return coll

    def _set_scene(self, scene):
        if self.use_scene_pc:
            self.scene_manager = trimesh.collision.CollisionManager()
            self.scene_manager.add_object("scene", scene)
        else:
            self.scene_manager = scene

    def _set_object(self, obj):
        if self.use_scene_pc:
            self.object_manager = trimesh.collision.CollisionManager()
            self.object_manager.add_object("obj", obj)

    def run(self):
        """
        the main function of each FCL process.
        """
        self.robot_manager = trimesh.collision.CollisionManager()
        for link in self.links:
            self.robot_manager.add_object(link.name, link.collision_mesh)
        while True:
            try:
                request = self.input_queue.get(timeout=1)
            except Queue.Empty:
                continue
            if request[0] == "set_scene":
                self._set_scene(request[1]),
            elif request[0] == "set_object":
                self._set_object(request[1]),
            elif request[0] == "collides":
                self.output_queue.put(
                    (
                        request[4],
                        self._collides(*request[1:4]),
                    )
                )
            elif request[0] == "obj_collides":
                self.output_queue.put(
                    (
                        request[3],
                        self._object_collides(*request[1:3]),
                    )
                )

    def set_scene(self, scene):
        self.input_queue.put(("set_scene", scene))

    def set_object(self, obj):
        self.input_queue.put(("set_object", obj))

    def collides(self, link_poses, inds, by_link, pind=None):
        self.input_queue.put(("collides", link_poses, inds, by_link, pind))

    def object_collides(self, poses, inds, pind=None):
        self.input_queue.put(("obj_collides", poses, inds, pind))


class FCLMultiSceneCollisionChecker(SceneCollisionChecker):
    def __init__(self, robot, n_proc=10, use_scene_pc=True):
        super().__init__(robot=robot)
        self._n_proc = n_proc
        self._use_scene_pc = use_scene_pc

        self.output_queue = mp.Queue()
        self.coll_procs = []
        for _ in range(self._n_proc):
            self.coll_procs.append(
                FCLProc(
                    self.robot.mesh_links,
                    self.output_queue,
                    use_scene_pc=self._use_scene_pc,
                )
            )
            self.coll_procs[-1].daemon = True
            self.coll_procs[-1].start()

    def set_scene(self, obs, scene=None):
        if self._use_scene_pc:
            super().set_scene(obs)
            self.cur_scene_pc = self._aggregate_pc(
                self.cur_scene_pc, self.scene_pc
            )
            pc = trimesh.PointCloud(self.cur_scene_pc.cpu().numpy())
            self.scene_mesh = trimesh.voxel.ops.points_to_marching_cubes(
                pc.vertices, pitch=0.01
            )
            for proc in self.coll_procs:
                proc.set_scene(self.scene_mesh)
        else:
            for proc in self.coll_procs:
                proc.set_scene(scene)

    def set_object(self, obs):
        if self.robot_to_model is None:
            self._compute_model_tfs(obs)
        if self._use_scene_pc:
            super().set_object(obs)
            obj_pc = self.obj_pc - self.obj_pc.mean()
            pc = trimesh.PointCloud(obj_pc)
            self.obj_mesh = trimesh.voxel.ops.points_to_marching_cubes(
                pc.vertices, pitch=0.025
            )
            self.obj_mesh.vertices -= self.obj_mesh.centroid
            for proc in self.coll_procs:
                proc.set_object(self.obj_mesh)

    def sample_in_bounds(self, num=20000, offset=0.0):
        return (
            torch.rand((num, 3), dtype=torch.float32, device=self.device)
            * (torch.tensor([1.0, 1.6, 0.4], device=self.device) - 2 * offset)
            + torch.tensor([-0.5, -0.8, 0.2], device=self.device)
            + offset
        )

    def check_object_collisions(self, translations, threshold=None):
        coll = torch.zeros(
            len(translations), dtype=torch.bool, device=self.device
        )
        tr = tra.transform_points(
            translations.cpu().numpy(), self.model_to_robot.cpu().numpy()
        )
        poses = np.repeat(np.eye(4)[None, ...], len(tr), axis=0)
        poses[:, :3, 3] = tr
        for i in range(self._n_proc):
            self.coll_procs[i].object_collides(
                poses,
                np.arange(
                    i * len(tr) // self._n_proc,
                    (i + 1) * len(tr) // self._n_proc,
                ),
                pind=i,
            )

        # collect computed iks
        for _ in range(self._n_proc):
            i, proc_coll = self.output_queue.get(True)
            coll[
                i * len(tr) // self._n_proc : (i + 1) * len(tr) // self._n_proc
            ] = torch.from_numpy(proc_coll).to(self.device)

        return coll

    def __call__(self, q, by_link=False, threshold=None):
        if q is None or len(q) == 0:
            return torch.zeros(
                (len(self.robot.mesh_links), 0),
                dtype=torch.bool,
                device=self.device,
            )
        coll = (
            torch.zeros(
                (len(self.robot.mesh_links), len(q)),
                dtype=torch.bool,
                device=self.device,
            )
            if by_link
            else torch.zeros(len(q), dtype=np.bool, device=self.device)
        )
        self.robot.set_joint_cfg(q)
        poses = {
            k.name: v.cpu().numpy() for k, v in self.robot.link_poses.items()
        }
        for i in range(self._n_proc):
            self.coll_procs[i].collides(
                poses,
                np.arange(
                    i * len(q) // self._n_proc,
                    (i + 1) * len(q) // self._n_proc,
                ),
                by_link,
                pind=i,
            )

        # collect computed iks
        for _ in range(self._n_proc):
            i, proc_coll = self.output_queue.get(True)
            if by_link:
                coll[
                    :,
                    i
                    * len(q)
                    // self._n_proc : (i + 1)
                    * len(q)
                    // self._n_proc,
                ] = torch.from_numpy(proc_coll).to(self.device)
            else:
                coll[
                    i
                    * len(q)
                    // self._n_proc : (i + 1)
                    * len(q)
                    // self._n_proc
                ] = torch.from_numpy(proc_coll).to(self.device)

        return coll


class SDFSceneCollisionChecker(SceneCollisionChecker):
    def __init__(self, robot, n_pts=1000, device=torch.device("cuda:0")):
        self.robot = robot
        self.device = device

        self.link_pts = {}
        for link in self.robot.mesh_links:
            self.link_pts[link] = (
                torch.from_numpy(link.collision_mesh.sample(n_pts))
                .float()
                .to(self.device)
            )

        self.scene_mesh = None

    def set_scene(self, obs):
        super().set_scene(obs)
        pc = trimesh.PointCloud(self.scene_pc)
        pitch = pc.extents.max() / 100
        scene_mesh = trimesh.voxel.ops.points_to_marching_cubes(
            pc.vertices, pitch=pitch
        )
        verts = torch.from_numpy(scene_mesh.vertices).float().to(self.device)
        faces = torch.from_numpy(scene_mesh.faces).long().to(self.device)
        self.scene_sdf = kal.conversions.trianglemesh_to_sdf(
            kal.rep.TriangleMesh.from_tensors(verts, faces)
        )

    def __call__(self, q, by_link=False, threshold=None):
        if q is None or len(q) == 0:
            return np.zeros((len(self.robot.mesh_links), 0), dtype=np.bool)
        self.robot.set_joint_cfg(q)
        coll = torch.zeros(
            (len(self.robot.mesh_links), len(q)),
            dtype=torch.bool,
            device=self.device,
        )
        for i, link in enumerate(self.robot.mesh_links):
            poses = self.robot.link_poses[link]
            link_pts = torch.cat(
                (
                    self.link_pts[link],
                    torch.ones(
                        (len(self.link_pts[link]), 1), device=self.device
                    ),
                ),
                dim=1,
            )
            tf_link_pts = torch.matmul(poses, link_pts.T).transpose(2, 1)[
                ..., :3
            ]
            sdf_vals = self.scene_sdf(tf_link_pts.reshape(-1, 3)).reshape(
                len(poses), -1
            )
            coll[i] = (sdf_vals <= 0).any(dim=1)

        return coll if by_link else coll.any(dim=0)


class SDFRobotCollisionChecker(SceneCollisionChecker):
    def __init__(self, robot, device=torch.device("cuda:0")):
        self.robot = robot
        self.device = device

        self.link_sdfs = {}
        for link in self.robot.mesh_links:
            link_mesh = link.collision_mesh
            verts = (
                torch.from_numpy(link_mesh.vertices).float().to(self.device)
            )
            faces = torch.from_numpy(link_mesh.faces).long().to(self.device)
            self.link_sdfs[link] = kal.conversions.trianglemesh_to_sdf(
                kal.rep.TriangleMesh.from_tensors(verts, faces)
            )

        self.scene_pts = None

    def set_scene(self, obs):
        super().set_scene(obs)
        self.scene_pts = (
            torch.from_numpy(self.scene_pc).float().to(self.device)
        )

    def __call__(self, q, by_link=False, threshold=None):
        if q is None or len(q) == 0:
            return np.zeros((len(self.robot.mesh_links), 0), dtype=np.bool)
        self.robot.set_joint_cfg(q)
        coll = torch.zeros(
            (len(self.robot.mesh_links), len(q)),
            dtype=torch.bool,
            device=self.device,
        )
        for i, link in enumerate(self.robot.mesh_links):
            poses = torch.inverse(self.robot.link_poses[link])
            scene_pts = torch.cat(
                (
                    self.scene_pts,
                    torch.ones((len(self.scene_pts), 1), device=self.device),
                ),
                dim=1,
            )
            tf_scene_pts = torch.matmul(poses, scene_pts.T).transpose(2, 1)[
                ..., :3
            ]
            sdf_vals = self.link_sdfs[link](
                tf_scene_pts.reshape(-1, 3)
            ).reshape(len(poses), -1)
            coll[i] = (sdf_vals <= 0).any(dim=1)

        return coll if by_link else coll.any(dim=0)


class NNCollisionChecker(CollisionChecker):
    def __init__(self, model_path, device=torch.device("cuda:0"), **kwargs):
        super().__init__(**kwargs)
        self.model_path = model_path
        self.device = device
        with open(osp.join(self.model_path, "train.yaml")) as f:
            self.cfg = yaml.safe_load(f)

    def _setup(self):
        chk = torch.load(
            os.path.join(self.model_path, "model.pth.tar"),
            map_location=self.device,
        )
        # Bounds and vox_size parameters can be loaded earlier
        self.model.load_state_dict(chk["model_state_dict"], strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

    @abc.abstractmethod
    def __call__(self):
        pass


class NNSelfCollisionChecker(NNCollisionChecker):
    def __init__(self, model_path, threshold=3.0, **kwargs):
        super().__init__(model_path, **kwargs)
        self.model = RobotCollisionNet(self.cfg["model"]["num_joints"])
        self.threshold = threshold
        self._setup()

    def set_allowed_collisions(self, l1, l2):
        pass

    def __call__(self, qs):
        if len(qs) == 0:
            return torch.cuda.BoolTensor([], device=self.device)
        if isinstance(qs, np.ndarray):
            qs = torch.from_numpy(qs).float().to(self.device)
        with torch.no_grad():
            out = self.model(qs[:, :8])
        return out.squeeze(-1) > self.threshold


class NNSceneCollisionChecker(NNCollisionChecker, SceneCollisionChecker):
    def __init__(self, model_path, robot, **kwargs):
        super().__init__(model_path=model_path, robot=robot, **kwargs)
        self.model = SceneCollisionNet(
            bounds=self.cfg["model"]["bounds"],
            vox_size=self.cfg["model"]["vox_size"],
        )
        self._setup()

        # Get features for robot links
        mesh_links = self.robot.mesh_links[1:]
        n_pts = self.cfg["dataset"]["n_obj_points"]
        self.link_pts = np.zeros((len(mesh_links), n_pts, 3), dtype=np.float32)
        for i, link in enumerate(mesh_links):
            pts = link.collision_mesh.sample(n_pts)
            l_pts = pts[None, ...]
            self.link_pts[i] = l_pts
        with torch.no_grad():
            self.link_features = self.model.get_obj_features(
                torch.from_numpy(self.link_pts).to(self.device)
            )

    def set_scene(self, obs):
        super().set_scene(obs)

        if self.robot_to_model is None:
            self._compute_model_tfs(obs)

        if self.cur_scene_pc is not None:
            self.cur_scene_pc = self._aggregate_pc(
                self.cur_scene_pc, self.scene_pc
            )
        else:
            self.cur_scene_pc = self._aggregate_pc(None, self.scene_pc)
        model_scene_pc = (
            self.robot_to_model
            @ torch.cat(
                (
                    self.cur_scene_pc,
                    torch.ones(
                        (len(self.cur_scene_pc), 1), device=self.device
                    ),
                ),
                dim=1,
            ).T
        )
        model_scene_pc = model_scene_pc[:3].T

        if self.model.bounds[0].device != self.device:
            self.model.bounds = [b.to(self.device) for b in self.model.bounds]
            self.model.vox_size = self.model.vox_size.to(self.device)
            self.model.num_voxels = self.model.num_voxels.to(self.device)

        # Clip points to model bounds and feed in for features
        in_bounds = (
            model_scene_pc[..., :3] > self.model.bounds[0] + 1e-5
        ).all(dim=-1)
        in_bounds &= (
            model_scene_pc[..., :3] < self.model.bounds[1] - 1e-5
        ).all(dim=-1)
        self.scene_features = self.model.get_scene_features(
            model_scene_pc[in_bounds].unsqueeze(0)
        ).squeeze(0)

    def set_object(self, obs):
        super().set_object(obs)
        if self.robot_to_model is None:
            self._compute_model_tfs(obs)

        obj_pc = tra.transform_points(
            self.obj_pc,
            self.robot_to_model.cpu().numpy(),
        )

        obj_tensor = torch.from_numpy(obj_pc.astype(np.float32)).to(
            self.device
        )
        obj_tensor -= obj_tensor.mean(dim=0)
        self.obj_features = self.model.get_obj_features(
            obj_tensor.unsqueeze(0)
        ).squeeze(0)

    def sample_in_bounds(self, num=20000, offset=0.0):
        return (
            torch.rand((num, 3), dtype=torch.float32, device=self.device)
            * (-torch.sub(*self.model.bounds) - 2 * offset)
            + self.model.bounds[0]
            + offset
        )

    def check_object_collisions(self, translations, threshold=0.45):
        res = torch.zeros(len(translations), dtype=bool, device=self.device)
        in_bounds = (translations > self.model.bounds[0] + 1e-5).all(dim=-1)
        in_bounds &= (translations < self.model.bounds[1] - 1e-5).all(dim=-1)
        if in_bounds.any():
            tr_in = translations[in_bounds]
            rots = np.repeat(
                np.eye(4)[:3, :2].flatten()[None, :], len(tr_in), axis=0
            )
            with torch.no_grad():
                out = self.model.classify_tfs(
                    self.obj_features[None, :],
                    self.scene_features[None, ...],
                    tr_in,
                    torch.from_numpy(rots).float().to(self.device),
                )
                res[in_bounds] = torch.sigmoid(out).squeeze() > threshold
        return res

    # Objs is an (n, m) array of points, tfs is an (n, t, 4, 4) array of
    # object transforms
    def __call__(self, qs, by_link=False, thresholded=True, threshold=0.4):
        self.robot.set_joint_cfg(qs)
        colls = torch.zeros(
            (len(self.link_features), len(qs)),
            dtype=torch.bool if thresholded else torch.float32,
            device=self.device,
        )
        trans = torch.empty(
            (len(self.link_features), len(qs), 3),
            dtype=torch.float32,
            device=self.device,
        )
        rots = torch.empty(
            (len(self.link_features), len(qs), 6),
            dtype=torch.float32,
            device=self.device,
        )

        for i, link in enumerate(self.robot.mesh_links[1:]):
            poses_tf = self.robot_to_model @ self.robot.link_poses[link]
            trans[i] = poses_tf[:, :3, 3]
            rots[i] = poses_tf[:, :3, :2].reshape(len(qs), -1)

        # filter translations that are out of bounds
        in_bounds = (trans > self.model.bounds[0] + 1e-5).all(dim=-1)
        in_bounds &= (trans < self.model.bounds[1] - 1e-5).all(dim=-1)
        if in_bounds.any():
            trans[~in_bounds] = 0.0  # Make inputs safe
            with torch.no_grad():
                out = self.model.classify_multi_obj_tfs(
                    self.link_features,
                    self.scene_features,
                    trans,
                    rots,
                )
                res = torch.sigmoid(out).squeeze(-1)
                res = res > threshold if thresholded else res
                colls = res * in_bounds

        if thresholded:
            return colls if by_link else colls.any(dim=0)
        else:
            return colls if by_link else colls.max(dim=0)[0]
