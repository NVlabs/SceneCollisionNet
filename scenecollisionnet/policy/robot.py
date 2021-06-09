import numpy as np
import torch

from .urdf import TorchURDF


# Thin wrapper around urdfpy class that adds cfg sampling
class Robot:
    def __init__(self, urdf_path, ee_name, device=None):
        if device is None:
            device = torch.device("cpu")
        self._robot = TorchURDF.load(urdf_path, device=device)
        self.dof = len(self.joint_names)
        self._ee_name = ee_name
        self.device = device
        self.min_joints = torch.tensor(
            [
                j.limit.lower
                for j in self._robot.joints
                if j.joint_type != "fixed"
            ],
            device=self.device,
        ).unsqueeze_(0)
        self.max_joints = torch.tensor(
            [
                j.limit.upper
                for j in self._robot.joints
                if j.joint_type != "fixed"
            ],
            device=self.device,
        ).unsqueeze_(0)

    @property
    def joint_names(self):
        return [j.name for j in self._robot.joints if j.joint_type != "fixed"]

    @property
    def links(self):
        return self._robot.links

    @property
    def mesh_links(self):
        return [
            link
            for link in self._robot.links
            if link.collision_mesh is not None
        ]

    @property
    def link_map(self):
        return self._robot.link_map

    @property
    def link_poses(self):
        return self._link_poses

    @property
    def ee_pose(self):
        return self.link_poses[self.link_map[self._ee_name]]

    def set_joint_cfg(self, q):
        if q is not None:
            if isinstance(q, np.ndarray):
                q = torch.from_numpy(q).float().to(self.device)
            if q.device != self.device:
                q = q.to(self.device)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            if q.ndim > 2:
                raise ValueError("Tensor is wrong shape, must have 2 dims")
            link_poses = self._robot.link_fk_batch(q[:, : self.dof])
            self._link_poses = link_poses
        else:
            self._link_poses = None

    def sample_cfg(self, num=1):
        alpha = torch.rand(num, self.dof, device=self.device)
        return alpha * self.min_joints + (1 - alpha) * self.max_joints
