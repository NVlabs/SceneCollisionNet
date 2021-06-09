import multiprocessing as mp
import queue as Queue

import numpy as np
import trimesh


class FCLProc(mp.Process):
    """
    Used for finding collisions in parallel using FCL.
    """

    def __init__(self, output_queue):
        """
        Args:
        output_queue: mp.Queue, the queue that all the output data
            that is computed is added to.
        """
        super().__init__()
        self.output_queue = output_queue
        self.input_queue = mp.Queue()

    def _collides(self, poses, inds):
        """computes collisions."""
        coll = np.zeros(len(inds), dtype=np.bool)
        for k, pose in enumerate(poses[inds]):
            self.obj_manager.set_transform("obj", pose)
            coll[k] = self.scene_manager.in_collision_other(self.obj_manager)
        return coll

    def _set_scene(self, scene):
        self.scene_manager = trimesh.collision.CollisionManager()
        self.scene_manager.add_object("scene", scene)

    def _set_object(self, obj):
        self.obj_manager = trimesh.collision.CollisionManager()
        self.obj_manager.add_object("obj", obj)

    def run(self):
        """
        the main function of each FCL process.
        """
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
                        request[3],
                        self._collides(*request[1:3]),
                    )
                )

    def set_scene(self, scene):
        self.input_queue.put(("set_scene", scene))

    def set_object(self, obj):
        self.input_queue.put(("set_object", obj))

    def collides(self, poses, inds, pind=None):
        self.input_queue.put(("collides", poses, inds, pind))


class FCLMultiSceneCollisionChecker:
    def __init__(self, n_proc=10):
        self._n_proc = n_proc

        self.output_queue = mp.Queue()
        self.coll_procs = []
        for _ in range(self._n_proc):
            self.coll_procs.append(
                FCLProc(
                    self.output_queue,
                )
            )
            self.coll_procs[-1].daemon = True
            self.coll_procs[-1].start()

    def set_scene(self, scene_pc):
        pc = trimesh.PointCloud(scene_pc)
        scene_mesh = trimesh.voxel.ops.points_to_marching_cubes(
            pc.vertices, pitch=0.01
        )
        for proc in self.coll_procs:
            proc.set_scene(scene_mesh)

    def set_object(self, obj_pc):
        pc = trimesh.PointCloud(obj_pc)
        obj_mesh = trimesh.voxel.ops.points_to_marching_cubes(
            pc.vertices, pitch=0.01
        )
        for proc in self.coll_procs:
            proc.set_object(obj_mesh)

    def __call__(self, scene, obj, trans, rots):
        self.set_scene(scene.squeeze().numpy())
        self.set_object(obj.squeeze().numpy())

        poses = np.repeat(np.eye(4)[None, ...], len(trans), axis=0)
        poses[:, :3, :2] = rots.numpy().reshape(-1, 3, 2)
        poses[:, :3, 2] = np.cross(rots[:, :3].numpy(), rots[:, 3:].numpy())
        poses[:, :3, 3] = trans
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

        return coll
