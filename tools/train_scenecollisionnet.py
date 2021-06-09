import argparse
import os
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn as nn
from autolab_core import YamlConfig, keyboard_input
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from scenecollisionnet.collision_models import PointNetGrid, SceneCollisionNet
from scenecollisionnet.datasets import IterableSceneCollisionDataset
from scenecollisionnet.trainer import Trainer


class SceneCollisionNetTrainer(Trainer):
    def to_binary(self, pred, true):
        return torch.sigmoid(pred), true


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a SceneCollisionNet")
    parser.add_argument(
        "--cfg",
        type=str,
        default="cfg/train_scenecollisionnet.yaml",
        help="config file with training params",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training"
    )
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-q", "--query_size", type=int)
    parser.add_argument("-o", "--obj_points", type=int)
    parser.add_argument("-s", "--scene_points", type=int)
    parser.add_argument("-t", "--trajectories", type=int)
    parser.add_argument("-r", "--rotations", type=int)
    parser.add_argument("-n", "--model_name", type=str)
    parser.add_argument(
        "-l", "--loss_pct", type=float, help="top k hard negative loss"
    )
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--device", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    # Replace config with args
    config = YamlConfig(args.cfg)
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size
    if args.query_size is not None:
        config["dataset"]["query_size"] = args.query_size
    if args.obj_points is not None:
        config["dataset"]["n_obj_points"] = args.obj_points
    if args.scene_points is not None:
        config["dataset"]["n_scene_points"] = args.scene_points
    if args.trajectories is not None:
        config["dataset"]["trajectories"] = args.trajectories
    if args.rotations is not None:
        config["dataset"]["rotations"] = args.rotations
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
    if args.dataset_path is not None:
        config["dataset"]["meshes"] = args.dataset_path
    if args.device is not None:
        config["trainer"]["device"] = args.device
    if args.loss_pct is not None:
        config["trainer"]["loss_pct"] = args.loss_pct
    resume = args.resume

    # Create output directory for model
    out = osp.join(config["model"]["path"], config["model"]["name"])
    if osp.exists(out) and not resume:
        response = keyboard_input(
            "A model exists at {}. Would you like to overwrite?".format(out),
            yesno=True,
        )
        if response.lower() == "n":
            sys.exit(0)
    elif osp.exists(out) and resume:
        resume = resume and osp.exists(osp.join(out, "checkpoint.pth.tar"))
    else:
        resume = False
        os.makedirs(out)
    config.save(osp.join(out, "train.yaml"))

    # Check whether GPU available
    cuda = torch.cuda.is_available()
    if not cuda:
        print("No CUDA available!")
        sys.exit(0)

    # 1. Model
    start_epoch = 0
    start_iteration = 0
    model_type = (
        SceneCollisionNet
        if config["model"]["type"] == "SceneCollisionNet"
        else PointNetGrid
    )
    model = model_type(
        bounds=config["model"]["bounds"],
        vox_size=config["model"]["vox_size"],
    )
    if resume:
        checkpoint = torch.load(osp.join(out, "checkpoint.pth.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_iteration = checkpoint["iteration"]

    # 2. Dataset
    kwargs = {
        "num_workers": config["trainer"]["num_workers"]
        if "num_workers" in config["trainer"]
        else os.cpu_count(),
        "pin_memory": True,
        "worker_init_fn": lambda _: np.random.seed(),
    }
    train_set = IterableSceneCollisionDataset(
        **config["dataset"],
        **config["camera"],
        bounds=config["model"]["bounds"]
    )
    train_loader = DataLoader(train_set, batch_size=None, **kwargs)

    # 3. Optimizer and loss
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["trainer"]["lr"],
        momentum=config["trainer"]["momentum"],
    )
    scaler = GradScaler()
    if resume:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        scaler.load_state_dict(checkpoint["amp"])

    trainer = SceneCollisionNetTrainer(
        device=torch.device(config["trainer"]["device"]),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        criterion=criterion,
        train_loader=train_loader,
        train_iterations=config["trainer"]["train_iterations"],
        val_iterations=config["trainer"]["val_iterations"],
        epochs=config["trainer"]["epochs"],
        out=out,
        loss_pct=config["trainer"]["loss_pct"],
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
