import numpy as np
import torch
import torch.nn as nn
import torch_scatter
from pointnet2.pointnet2_modules import PointnetSAModule
from pointnet2.pytorch_utils import FC, Conv1d, Conv3d
from torch.cuda.amp import autocast

OBJ_NPOINTS = [256, 64, None]
OBJ_RADII = [0.02, 0.04, None]
OBJ_NSAMPLES = [64, 128, None]
OBJ_MLPS = [[0, 64, 128], [128, 128, 256], [256, 256, 512]]
SCENE_PT_MLP = [3, 128, 256]
SCENE_VOX_MLP = [256, 512, 1024, 512]
CLS_FC = [2057, 1024, 256]


class SceneCollisionNet(nn.Module):
    def __init__(self, bounds, vox_size):
        super().__init__()
        self.bounds = nn.Parameter(
            torch.from_numpy(np.asarray(bounds)).float(), requires_grad=False
        )
        self.vox_size = nn.Parameter(
            torch.from_numpy(np.asarray(vox_size)).float(), requires_grad=False
        )
        self.num_voxels = nn.Parameter(
            ((self.bounds[1] - self.bounds[0]) / self.vox_size).long(),
            requires_grad=False,
        )

        self.scene_pt_mlp = nn.Sequential()
        for i in range(len(SCENE_PT_MLP) - 1):
            self.scene_pt_mlp.add_module(
                "pt_layer{}".format(i),
                Conv1d(SCENE_PT_MLP[i], SCENE_PT_MLP[i + 1], first=(i == 0)),
            )

        self.scene_vox_mlp = nn.ModuleList()
        for i in range(len(SCENE_VOX_MLP) - 1):
            scene_conv = nn.Sequential()
            if SCENE_VOX_MLP[i + 1] > SCENE_VOX_MLP[i]:
                scene_conv.add_module(
                    "3d_conv_layer{}".format(i),
                    Conv3d(
                        SCENE_VOX_MLP[i],
                        SCENE_VOX_MLP[i + 1],
                        kernel_size=3,
                        padding=1,
                    ),
                )
                scene_conv.add_module(
                    "3d_max_layer{}".format(i), nn.MaxPool3d(2, stride=2)
                )
            else:
                scene_conv.add_module(
                    "3d_convt_layer{}".format(i),
                    nn.ConvTranspose3d(
                        SCENE_VOX_MLP[i],
                        SCENE_VOX_MLP[i + 1],
                        kernel_size=2,
                        stride=2,
                    ),
                )
            self.scene_vox_mlp.append(scene_conv)

        self.obj_SA_modules = nn.ModuleList()
        for k in range(OBJ_NPOINTS.__len__()):
            self.obj_SA_modules.append(
                PointnetSAModule(
                    npoint=OBJ_NPOINTS[k],
                    radius=OBJ_RADII[k],
                    nsample=OBJ_NSAMPLES[k],
                    mlp=OBJ_MLPS[k],
                    bn=False,
                    first=(i == 0),
                )
            )

        self.obj_FCs = nn.ModuleList(
            [
                FC(OBJ_MLPS[-1][-1], 1024),
                FC(1024, 1024),
            ]
        )

        self.classifier = nn.Sequential(
            FC(CLS_FC[0], CLS_FC[1], first=True),
            FC(CLS_FC[1], CLS_FC[2]),
            FC(CLS_FC[2], 1, activation=None),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3
            else None
        )

        return xyz, features

    def _inds_to_flat(self, inds, scale=1):
        flat_inds = inds * torch.cuda.IntTensor(
            [
                self.num_voxels[1:].prod() // (scale ** 2),
                self.num_voxels[2] // scale,
                1,
            ],
            device=self.num_voxels.device,
        )
        return flat_inds.sum(axis=-1)

    def _inds_from_flat(self, flat_inds, scale=1):
        ind0 = flat_inds // (self.num_voxels[1:].prod() // (scale ** 2))
        ind1 = (flat_inds % (self.num_voxels[1:].prod() // (scale ** 2))) // (
            self.num_voxels[2] // scale
        )
        ind2 = (flat_inds % (self.num_voxels[1:].prod() // (scale ** 2))) % (
            self.num_voxels[2] // scale
        )
        return torch.stack((ind0, ind1, ind2), dim=-1)

    def voxel_inds(self, xyz, scale=1):
        inds = ((xyz - self.bounds[0]) // (scale * self.vox_size)).int()
        return self._inds_to_flat(inds, scale=scale)

    def get_obj_features(self, obj_pc):
        obj_xyz, obj_features = self._break_up_pc(obj_pc)

        # Featurize obj
        for i in range(len(self.obj_SA_modules)):
            obj_xyz, obj_features = self.obj_SA_modules[i](
                obj_xyz, obj_features
            )
        for i in range(len(self.obj_FCs)):
            obj_features = self.obj_FCs[i](obj_features.squeeze(axis=-1))

        return obj_features

    def get_scene_features(self, scene_pc):
        scene_xyz, scene_features = self._break_up_pc(scene_pc)
        scene_inds = self.voxel_inds(scene_xyz)

        # Featurize scene points and max pool over voxels
        scene_vox_centers = (
            self._inds_from_flat(scene_inds) * self.vox_size
            + self.vox_size / 2
            + self.bounds[0]
        )
        scene_xyz_centered = (scene_pc[..., :3] - scene_vox_centers).transpose(
            2, 1
        )
        if scene_features is not None:
            scene_features = self.scene_pt_mlp(
                torch.cat((scene_xyz_centered, scene_features), dim=1)
            )
        else:
            scene_features = self.scene_pt_mlp(scene_xyz_centered)
        max_vox_features = torch.zeros(
            (*scene_features.shape[:2], self.num_voxels.prod())
        ).to(scene_pc.device)
        if scene_inds.max() >= self.num_voxels.prod():
            print(
                scene_xyz[range(len(scene_pc)), scene_inds.max(axis=-1)[1]],
                scene_inds.max(),
            )
        assert scene_inds.max() < self.num_voxels.prod()
        assert scene_inds.min() >= 0

        with autocast(enabled=False):
            max_vox_features[
                ..., : scene_inds.max() + 1
            ] = torch_scatter.scatter_max(
                scene_features.float(), scene_inds[:, None, :]
            )[
                0
            ]
        max_vox_features = max_vox_features.reshape(
            *max_vox_features.shape[:2], *self.num_voxels.int()
        )

        # 3D conv over voxels
        l_vox_features = [max_vox_features]
        for i in range(len(self.scene_vox_mlp)):
            li_vox_features = self.scene_vox_mlp[i](l_vox_features[i])
            l_vox_features.append(li_vox_features)

        # Stack features from different levels
        stack_vox_features = torch.cat(
            (l_vox_features[1], l_vox_features[-1]), dim=1
        )
        stack_vox_features = stack_vox_features.reshape(
            *stack_vox_features.shape[:2], -1
        )
        return stack_vox_features

    # scene_pc: (b, n_scene_fts); obj_pc: (b, n_obj_fts);
    # trans: (q, 3); rots: (q, 6)
    def classify_tfs(self, obj_features, scene_features, trans, rots):
        b = len(scene_features)
        q = len(trans)

        # Get voxel indices for translations
        trans_inds = self.voxel_inds(trans, scale=2).long()
        if trans_inds.max() >= scene_features.shape[2]:
            print(trans[trans_inds.argmax()], trans_inds.max())
        assert trans_inds.max() < scene_features.shape[2]
        assert trans_inds.min() >= 0
        vox_trans_features = scene_features[..., trans_inds].transpose(2, 1)

        # Calculate translation offsets from centers of voxels
        tr_vox_centers = (
            self._inds_from_flat(trans_inds, scale=2) * self.vox_size * 2
            + self.vox_size / 2
            + self.bounds[0]
        )
        trans_offsets = trans - tr_vox_centers.float()

        # Send concatenated features to classifier
        class_in = torch.cat(
            (
                obj_features.unsqueeze(1).expand(b, q, obj_features.shape[-1]),
                vox_trans_features,
                trans_offsets.unsqueeze(0).expand(b, q, 3),
                rots.unsqueeze(0).expand(b, q, 6),
            ),
            dim=-1,
        )

        return self.classifier(class_in)

    # scene_pc: (n_scene_fts); obj_pc: (b, n_obj_fts);
    # trans: (b, q, 3); rots: (b, q, 6)
    def classify_multi_obj_tfs(
        self, obj_features, scene_features, trans, rots
    ):
        b, q = trans.shape[:2]

        # Get voxel indices for translations
        trans_inds = self.voxel_inds(trans, scale=2).long()
        if trans_inds.max() >= scene_features.shape[-1]:
            print(trans[trans_inds.argmax()], trans_inds.max())
        assert trans_inds.max() < scene_features.shape[-1]
        assert trans_inds.min() >= 0
        vox_trans_features = scene_features[..., trans_inds].permute(1, 2, 0)

        # Calculate translation offsets from centers of voxels
        tr_vox_centers = (
            self._inds_from_flat(trans_inds, scale=2) * self.vox_size * 2
            + self.vox_size / 2
            + self.bounds[0]
        )
        trans_offsets = trans - tr_vox_centers.float()

        # Send concatenated features to classifier
        class_in = torch.cat(
            (
                obj_features.unsqueeze(1).expand(-1, q, -1),
                vox_trans_features,
                trans_offsets,
                rots,
            ),
            dim=-1,
        )

        return self.classifier(class_in)

    # scene_pc: (b, n_scene_pts, 6); obj_pc: (b, n_obj_pts, 6);
    # trans: (b, q, 3); rots: (b, q, 6)
    def forward(self, scene_pc, obj_pc, trans, rots):

        obj_features = self.get_obj_features(obj_pc)
        scene_features = self.get_scene_features(scene_pc)
        return self.classify_tfs(obj_features, scene_features, trans, rots)


class RobotCollisionNet(nn.Module):
    def __init__(self, num_joints):
        super().__init__()
        self.classifier = nn.Sequential(
            FC(num_joints, 64),
            FC(64, 64),
            FC(64, 64),
            FC(64, 64),
            FC(64, 64),
            FC(64, 1, activation=None),
        )

    def forward(self, centers):
        return self.classifier(centers)
