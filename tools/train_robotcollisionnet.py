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

from scenecollisionnet.collision_models import RobotCollisionNet
from scenecollisionnet.datasets import IterableRobotCollisionDataset
from scenecollisionnet.trainer import Trainer


class RobotCollisionTrainer(Trainer):
    def to_binary(self, pred, true):
        return torch.clamp(pred, 0.0, 1.0), true > 1.0


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a SelfCollisionNet")
    parser.add_argument(
        "--cfg",
        type=str,
        default="cfg/train_robotcollisionnet.yaml",
        help="config file with training params",
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume training"
    )
    parser.add_argument("-b", "--batch_size", type=int)
    parser.add_argument("-n", "--model_name", type=str)
    parser.add_argument(
        "-d", "--device", type=int, default=0, help="GPU index"
    )
    args = parser.parse_args()

    # Replace config with args
    config = YamlConfig(args.cfg)
    if args.batch_size is not None:
        config["dataset"]["batch_size"] = args.batch_size
    if args.model_name is not None:
        config["model"]["name"] = args.model_name
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

    # Check whether GPU available
    cuda = torch.cuda.is_available()
    if not cuda:
        print("No CUDA available!")
        sys.exit(0)

    # 1. Dataset
    kwargs = {
        "num_workers": config["trainer"]["num_workers"]
        if "num_workers" in config["trainer"]
        else os.cpu_count(),
        "pin_memory": True,
        "worker_init_fn": lambda _: np.random.seed(),
    }
    train_set = IterableRobotCollisionDataset(
        **config["dataset"],
    )
    train_loader = DataLoader(train_set, batch_size=None, **kwargs)

    # 2. Model
    start_epoch = 0
    start_iteration = 0
    config["model"]["num_joints"] = len(train_set.robot.actuated_joints)
    model = RobotCollisionNet(num_joints=config["model"]["num_joints"])
    if resume:
        checkpoint = torch.load(osp.join(out, "checkpoint.pth.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        start_iteration = checkpoint["iteration"]

    # 3. Optimizer and loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config["trainer"]["lr"],
        momentum=config["trainer"]["momentum"],
    )
    scaler = GradScaler()
    if resume:
        optimizer.load_state_dict(checkpoint["optim_state_dict"])
        scaler.load_state_dict(checkpoint["amp"])

    # Save out training config
    config.save(osp.join(out, "train.yaml"))

    trainer = RobotCollisionTrainer(
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
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()
