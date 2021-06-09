import torch
import torch.nn as nn
from pointnet2.pointnet2_modules import PointnetSAModuleMSG
from pointnet2.pytorch_utils import Conv1d

OBJ_NPOINTS = [512, 256, 128, 1]
OBJ_MLPS = [
    [[16, 16, 32], [32, 32, 64]],
    [[64, 64, 128], [64, 96, 128]],
    [[128, 196, 256], [128, 196, 256]],
    [[256, 256, 512], [256, 384, 512]],
]
OBJ_RADIUS = [[0.005, 0.01], [0.01, 0.02], [0.02, 0.04], [0.04, 0.08]]
OBJ_NSAMPLES = [[32, 32], [32, 32], [32, 32], [32, 32]]

SCENE_NPOINTS = [8192, 2048, 512, 128, 32, 1]
SCENE_MLPS = [
    [[16, 16, 32], [32, 32, 64]],
    [[64, 64, 128], [64, 96, 128]],
    [[128, 196, 256], [128, 196, 256]],
    [[256, 256, 512], [256, 384, 512]],
    [[512, 768, 1024], [512, 768, 1024]],
    [[1024, 1024, 2048], [1024, 1536, 2048]],
]
SCENE_RADIUS = [
    [0.005, 0.01],
    [0.01, 0.05],
    [0.1, 0.2],
    [0.2, 0.4],
    [0.4, 0.8],
    [0.8, 1.6],
]
SCENE_NSAMPLE = [[32, 32], [32, 32], [32, 32], [32, 32], [32, 32], [32, 32]]
CLS_FC = [2048, 1024, 512, 512]


class PointNetGrid(nn.Module):
    def __init__(self, bounds, vox_size):
        super().__init__()

        self.bounds = nn.Parameter(
            torch.FloatTensor(bounds), requires_grad=False
        )
        self.vox_size = nn.Parameter(
            torch.FloatTensor(vox_size), requires_grad=False
        )
        if torch.allclose(self.vox_size, self.bounds[1] - self.bounds[0]):
            self.num_voxels = nn.Parameter(
                torch.ones(3, dtype=torch.long), requires_grad=False
            )
        else:
            self.num_voxels = nn.Parameter(
                (self.bounds[1] - self.bounds[0]) / (self.vox_size / 2) + 1,
                requires_grad=False,
            ).long()
        self.vox_centers = torch.meshgrid(
            [
                torch.linspace(self.bounds[0, i], self.bounds[1, i], n)
                for i, n in enumerate(self.num_voxels)
            ]
        )
        self.vox_centers = nn.Parameter(
            torch.stack([vc.flatten() for vc in self.vox_centers], dim=1),
            requires_grad=False,
        )

        # Scene pointnet layers
        channel_in = 0
        self.scene_SA_modules = nn.ModuleList()
        for k in range(SCENE_NPOINTS.__len__()):
            mlps = SCENE_MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.scene_SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=SCENE_NPOINTS[k],
                    radii=SCENE_RADIUS[k],
                    nsamples=SCENE_NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=False,
                )
            )
            channel_in = channel_out

        # Obj pointnet layers
        channel_in = 0
        self.obj_SA_modules = nn.ModuleList()
        for k in range(OBJ_NPOINTS.__len__()):
            mlps = OBJ_MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.obj_SA_modules.append(
                PointnetSAModuleMSG(
                    npoint=OBJ_NPOINTS[k],
                    radii=OBJ_RADIUS[k],
                    nsamples=OBJ_NSAMPLES[k],
                    mlps=mlps,
                    use_xyz=True,
                    bn=False,
                )
            )
            channel_in = channel_out

        cls_layers = []
        pre_channel = 4096 + 1024 + 3 + 6
        for k in range(0, CLS_FC.__len__()):
            cls_layers.append(Conv1d(pre_channel, CLS_FC[k]))
            pre_channel = CLS_FC[k]
        cls_layers.append(Conv1d(pre_channel, 1, activation=None))
        cls_layers.insert(1, nn.Dropout(0.5))
        self.cls_layer = nn.Sequential(*cls_layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3
            else None
        )

        return xyz, features

    def get_obj_features(self, obj):
        xyz, features = self._break_up_pc(obj)
        for i in range(len(self.obj_SA_modules)):
            xyz, features = self.obj_SA_modules[i](xyz, features)
        return features.squeeze(dim=1)

    def get_scene_features(self, scene):
        xyz, features = self._break_up_pc(scene)
        for i in range(len(self.scene_SA_modules)):
            xyz, features = self.scene_SA_modules[i](xyz, features)
        return features.squeeze(dim=1)

    def forward(self, scene, obj, tr, rot):

        # Voxelize scene points
        scene_vox_mask = (
            torch.abs(scene.transpose(1, 0) - self.vox_centers)
            < self.vox_size / 2
        ).all(dim=-1)
        scene_vox_counts = scene_vox_mask.sum(dim=0)
        nonzero_inds = torch.where(scene_vox_counts > 0)[0]
        val_vox_counts = scene_vox_counts[nonzero_inds]
        val_vox_centers = self.vox_centers[nonzero_inds]

        # Sample points and normalize per voxel - super janky but works
        scene_smpl_inds = (
            torch.randint(
                scene.size(1),
                size=(1024, len(nonzero_inds)),
                device=scene.device,
            )
            % val_vox_counts
        )
        offsets = torch.roll(val_vox_counts.cumsum(dim=0), 1, 0)
        offsets[0] = 0
        scene_smpl_inds += offsets
        scene_pts = scene[
            :, torch.where(scene_vox_mask.T)[1][scene_smpl_inds.flatten()]
        ].reshape(*scene_smpl_inds.shape, -1)
        norm_scene_pts = (scene_pts - val_vox_centers).transpose(1, 0)

        # Normalize translations
        tr_vox_mask = (
            torch.abs(tr.unsqueeze(1) - val_vox_centers) < self.vox_size / 2
        ).all(dim=-1)
        tr_val_inds, tr_vox_inds = torch.where(tr_vox_mask)
        norm_trans = tr[tr_val_inds] - val_vox_centers[tr_vox_inds]

        q = len(norm_trans)
        obj_feats = self.get_obj_features(obj)
        scene_feats = self.get_scene_features(norm_scene_pts)
        cls_in = torch.cat(
            (
                obj_feats.expand(q, -1, -1),
                scene_feats[tr_vox_inds],
                norm_trans.unsqueeze(2),
                rot[tr_val_inds].unsqueeze(2),
            ),
            dim=1,
        )
        pred_cls = self.cls_layer(cls_in).squeeze(dim=-1).squeeze(dim=-1)
        return (
            torch.zeros(q, device=tr.device).scatter_add(
                0, tr_val_inds, pred_cls
            )
            / 8.0
        )
