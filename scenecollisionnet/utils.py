import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Queue
from queue import Empty
from typing import Any, Callable, Dict, Iterable

import h5py
import numpy as np
import trimesh
import trimesh.transformations as tra


class MeshLoader(object):
    """A tool for loading meshes from a base directory.
    Attributes
    ----------
    basedir : str
        basedir containing mesh files
    """

    def __init__(self, basedir, only_ext=None):
        self.basedir = basedir
        self._map = {}
        for root, _, fns in os.walk(basedir):
            for fn in fns:
                full_fn = os.path.join(root, fn)
                f, ext = os.path.splitext(fn)
                if only_ext is not None and ext != only_ext:
                    continue
                if basedir != root:
                    f = os.path.basename(root) + "~" + f
                if ext[1:] not in trimesh.available_formats():
                    continue
                if f in self._map:
                    continue
                    # raise ValueError('Duplicate file named {}'.format(f))
                self._map[f] = full_fn

    def meshes(self):
        return self._map.keys()

    def get_path(self, name):
        if name in self._map:
            return self._map[name]
        raise ValueError(
            "Could not find mesh with name {} in directory {}".format(
                name, self.basedir
            )
        )

    def load(self, name):
        m = trimesh.load(self.get_path(name))
        m.metadata["name"] = name
        return m


class ProcessKillingExecutor:
    """
    The ProcessKillingExecutor works like an `Executor
    <https://docs.python.org/dev/library/concurrent.futures.html#executor-objects>`_
    in that it uses a bunch of processes to execute calls to a function with
    different arguments asynchronously.

    But other than the `ProcessPoolExecutor
    <https://docs.python.org/dev/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor>`_,
    the ProcessKillingExecutor forks a new Process for each function call that
    terminates after the function returns or if a timeout occurs.

    This means that contrary to the Executors and similar classes provided by
    the Python Standard Library, you can rely on the fact that a process will
    get killed if a timeout occurs and that absolutely no side can occur
    between function calls.

    Note that descendant processes of each process will not be terminated –
    they will simply become orphaned.
    """

    def __init__(self, max_workers: int = None):
        self.processes = max_workers or os.cpu_count()

    def map(
        self,
        func: Callable,
        iterable: Iterable,
        timeout: float = None,
        callback_timeout: Callable = None,
        daemon: bool = True,
    ) -> Iterable:
        """
        :param func: the function to execute
        :param iterable: an iterable of function arguments
        :param timeout: after this time, the process executing the function
                will be killed if it did not finish
        :param callback_timeout: this function will be called, if the task
                times out. It gets the same arguments as the original function
        :param daemon: define the child process as daemon
        """
        executor = ProcessPoolExecutor(max_workers=self.processes)
        params = (
            {
                "func": func,
                "fn_args": p_args,
                "p_kwargs": {},
                "timeout": timeout,
                "callback_timeout": callback_timeout,
                "daemon": daemon,
            }
            for p_args in iterable
        )
        return executor.map(self._submit_unpack_kwargs, params)

    def _submit_unpack_kwargs(self, params):
        """unpack the kwargs and call submit"""

        return self.submit(**params)

    def submit(
        self,
        func: Callable,
        fn_args: Any,
        p_kwargs: Dict,
        timeout: float,
        callback_timeout: Callable[[Any], Any],
        daemon: bool,
    ):
        """
        Submits a callable to be executed with the given arguments.
        Schedules the callable to be executed as func(*args, **kwargs) in a new
         process.
        :param func: the function to execute
        :param fn_args: the arguments to pass to the function. Can be one argument
                or a tuple of multiple args.
        :param p_kwargs: the kwargs to pass to the function
        :param timeout: after this time, the process executing the function
                will be killed if it did not finish
        :param callback_timeout: this function will be called with the same
                arguments, if the task times out.
        :param daemon: run the child process as daemon
        :return: the result of the function, or None if the process failed or
                timed out
        """
        p_args = fn_args if isinstance(fn_args, tuple) else (fn_args,)
        queue = Queue()
        p = Process(
            target=self._process_run,
            args=(
                queue,
                func,
                *p_args,
            ),
            kwargs=p_kwargs,
        )

        if daemon:
            p.daemon = True

        p.start()
        try:
            ret = queue.get(block=True, timeout=timeout)
            if ret is None:
                callback_timeout(*p_args, **p_kwargs)
            return ret
        except Empty:
            callback_timeout(*p_args, **p_kwargs)
        if p.is_alive():
            p.terminate()
            p.join()

    @staticmethod
    def _process_run(
        queue: Queue, func: Callable[[Any], Any] = None, *args, **kwargs
    ):
        """
        Executes the specified function as func(*args, **kwargs).
        The result will be stored in the shared dictionary
        :param func: the function to execute
        :param queue: a Queue
        """
        queue.put(func(*args, **kwargs))


def compute_camera_pose(distance, azimuth, elevation):
    cam_tf = tra.euler_matrix(np.pi / 2, 0, 0).dot(
        tra.euler_matrix(0, np.pi / 2, 0)
    )

    extrinsics = np.eye(4)
    extrinsics[0, 3] += distance
    extrinsics = tra.euler_matrix(0, -elevation, azimuth).dot(extrinsics)

    cam_pose = extrinsics.dot(cam_tf)
    frame_pose = cam_pose.copy()
    frame_pose[:, 1:3] *= -1.0
    return cam_pose, frame_pose


def process_mesh(in_path, out_path, scale, grasps, return_stps=True):
    mesh = trimesh.load(in_path, force="mesh", skip_materials=True)
    cat = (
        ""
        if os.path.basename(os.path.dirname(out_path)) == "meshes"
        else os.path.basename(os.path.dirname(out_path))
    )
    if not os.path.exists(os.path.dirname(out_path)):
        os.makedirs(os.path.dirname(out_path))

    if not mesh.is_watertight or len(mesh.faces) > 1000:
        obj_path = os.path.splitext(in_path)[0] + ".obj"
        is_obj = os.path.exists(obj_path)
        if not is_obj:
            mesh.export(obj_path)

        simplify_path = "{}_{:d}.obj".format(
            os.path.splitext(out_path)[0],
            np.random.RandomState().randint(2 ** 16),
        )
        manifold_cmd = [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../extern/Manifold/build/manifold",
            ),
            obj_path,
            simplify_path,
        ]
        simplify_cmd = [
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../extern/Manifold/build/simplify",
            ),
            "-i",
            simplify_path,
            "-o",
            simplify_path,
            "-m",
            "-f 1000",
        ]
        try:
            subprocess.check_output(manifold_cmd)
        except subprocess.CalledProcessError:
            if not is_obj:
                os.remove(obj_path)
            if os.path.exists(simplify_path):
                os.remove(simplify_path)
            return None
        if not is_obj:
            os.remove(obj_path)
        try:
            subprocess.check_output(simplify_cmd)
        except subprocess.CalledProcessError:
            if os.path.exists(simplify_path):
                os.remove(simplify_path)
            return None

        mesh = trimesh.load(simplify_path)
        os.remove(simplify_path)

    # Create final scaled and transformed mesh
    if not mesh.is_watertight:
        mesh.center_mass = mesh.centroid

    mesh.apply_scale(scale)
    mesh_offset = mesh.center_mass
    mesh.apply_transform(
        trimesh.transformations.translation_matrix(-mesh_offset)
    )
    m_scale = "{}_{}.obj".format(
        os.path.splitext(os.path.basename(out_path))[0], scale
    )
    s_out_path = os.path.join(os.path.dirname(out_path), m_scale)
    mesh.export(s_out_path)

    m_info = {
        "path": os.path.join(cat, m_scale),
        "scale": scale,
        "category": cat,
    }

    # Calculate stable poses and add grasps
    if return_stps:
        try:
            stps, probs = mesh.compute_stable_poses()
            if not probs.any():
                os.remove(s_out_path)
                return None
            m_info.update({"stps": stps, "probs": probs / probs.sum()})
        except Exception:
            os.remove(s_out_path)
            return None

    if grasps is not None:
        grasp_data = h5py.File(grasps, "r")["grasps"]
        positive_grasps = grasp_data["transforms"][:][
            grasp_data["qualities/flex/object_in_gripper"][:] > 0
        ]
        positive_grasps[:, :3, 3] -= mesh_offset
        m_info["grasps"] = positive_grasps

    return os.path.splitext(m_scale)[0], m_info
