import multiprocessing as mp
import queue as Queue

import numpy as np
import torch
import trimesh

try:
    import kaolin as kal
except ModuleNotFoundError:
    print("Kaolin not installed, SDF baselines not available!")


class SDFProc(mp.Process):
    """
    Used for finding collisions in parallel using SDFs.
    """

    def __init__(self, output_queue, device):
        """
        Args:
        output_queue: mp.Queue, the queue that all the output data
            that is computed is added to.
        """
        super().__init__()
        self.output_queue = output_queue
        self.device = device
        self.input_queue = mp.Queue()

    def _set_sdf(self, verts, faces):
        self.v = torch.from_numpy(verts).float().to(self.device)
        self.f = torch.from_numpy(faces).long().to(self.device)

    def _collides(self, poses, inds):
        """computes collisions."""
        ind_poses = torch.from_numpy(poses[inds]).float().to(self.device)
        tf_pts = torch.cat(
            (
                self.points,
                torch.ones((len(self.points), 1), device=self.device),
            ),
            dim=1,
        )
        tf_pts = torch.matmul(ind_poses, tf_pts.T).transpose(2, 1)[..., :3]
        sdf_vals = kal.ops.mesh.check_sign(
            self.v.expand(len(inds), -1, -1), self.f, tf_pts
        )
        return sdf_vals.any(dim=-1).cpu().numpy()

    def run(self):
        """
        the main function of each process.
        """
        while True:
            try:
                request = self.input_queue.get(timeout=1)
            except Queue.Empty:
                continue
            if request[0] == "set_sdf":
                self._set_sdf(*request[1:])
            elif request[0] == "set_points":
                self.points = request[1].float().to(self.device)
            elif request[0] == "collides":
                self.output_queue.put(
                    (
                        request[3],
                        self._collides(*request[1:3]),
                    )
                )

    def set_sdf(self, verts, faces):
        self.input_queue.put(("set_sdf", verts, faces))

    def set_points(self, pts):
        self.input_queue.put(("set_points", pts))

    def collides(self, poses, inds, pind=None):
        self.input_queue.put(("collides", poses, inds, pind))


class SDFMultiSceneCollisionChecker:
    def __init__(self, device="cuda:0", n_proc=2):
        self.device = torch.device(device)
        self._n_proc = n_proc

        if self._n_proc > 0:
            self.output_queue = mp.Queue()
            self.coll_procs = []
            for _ in range(self._n_proc):
                self.coll_procs.append(SDFProc(self.output_queue, self.device))
                self.coll_procs[-1].daemon = True
                self.coll_procs[-1].start()

    def set_points(self, pts):
        if self._n_proc == 0:
            self.points = pts.float().to(self.device)
        else:
            for proc in self.coll_procs:
                proc.set_points(pts)

    def set_sdf(self, verts, faces):
        if self._n_proc == 0:
            self.v = torch.from_numpy(verts).float().to(self.device)
            self.f = torch.from_numpy(faces).long().to(self.device)
        else:
            for proc in self.coll_procs:
                proc.set_sdf(verts, faces)

    def _points_to_mesh(self, pts):
        pc = trimesh.PointCloud(pts)
        mesh = trimesh.voxel.ops.points_to_marching_cubes(
            pc.vertices, pitch=0.01
        )
        return mesh.vertices, mesh.faces

    def _to_poses(self, trans, rots):
        poses = np.repeat(np.eye(4)[None, ...], len(trans), axis=0)
        poses[:, :3, :2] = rots.numpy().reshape(-1, 3, 2)
        poses[:, :3, 2] = np.cross(rots[:, :3].numpy(), rots[:, 3:].numpy())
        poses[:, :3, 3] = trans
        return poses

    def __call__(self, scene, obj, trans, rots):
        self.set_scene(scene.squeeze())
        self.set_object(obj.squeeze())
        poses = self._to_poses(trans, rots)

        if self._n_proc > 0:
            coll = np.zeros(len(trans), dtype=np.bool)
            for i in range(self._n_proc):
                self.coll_procs[i].collides(
                    poses,
                    np.arange(
                        i * len(trans) // self._n_proc,
                        (i + 1) * len(trans) // self._n_proc,
                    ),
                    pind=i,
                )

            # collect computed collisions
            for _ in range(self._n_proc):
                i, proc_coll = self.output_queue.get(True)
                coll[
                    i
                    * len(trans)
                    // self._n_proc : (i + 1)
                    * len(trans)
                    // self._n_proc
                ] = proc_coll

        else:
            poses = torch.from_numpy(poses).float().to(self.device)
            tf_pts = torch.cat(
                (
                    self.points,
                    torch.ones((len(self.points), 1), device=self.device),
                ),
                dim=1,
            )
            tf_pts = torch.matmul(poses, tf_pts.T).transpose(2, 1)[..., :3]
            sdf_vals = kal.ops.mesh.check_sign(
                self.v.expand(len(poses), -1, -1), self.f, tf_pts
            )
            coll = sdf_vals.any(dim=-1).cpu().numpy()

        return coll


class SDFSceneCollisionChecker(SDFMultiSceneCollisionChecker):
    def set_scene(self, scene_pts):
        self.set_sdf(*self._points_to_mesh(scene_pts))

    def set_object(self, obj_pts):
        self.set_points(obj_pts)


class SDFObjCollisionChecker(SDFMultiSceneCollisionChecker):
    def set_scene(self, scene_pts):
        self.set_points(scene_pts)

    def set_object(self, obj_pts):
        self.set_sdf(*self._points_to_mesh(obj_pts))

    def _to_poses(self, trans, rots):
        poses = super()._to_poses(trans, rots)
        return np.linalg.inv(poses)
