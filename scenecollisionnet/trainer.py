import datetime
import os
import os.path as osp
import shutil
import sys
from abc import abstractmethod

import numpy as np
import torch
import tqdm
from autolab_core import BinaryClassificationResult
from torch.autograd import Variable
from torch.cuda.amp import autocast


class Trainer(object):
    def __init__(
        self,
        device,
        model,
        optimizer,
        scaler,
        criterion,
        train_loader,
        train_iterations,
        val_iterations,
        epochs,
        out,
        loss_pct=1.0,
    ):
        self.device = device
        self.model = model
        self.optim = optimizer
        self.scaler = scaler
        self.criterion = criterion

        self.data_loader = iter(train_loader)
        self.train_iterations = train_iterations
        self.val_iterations = val_iterations
        self.loss_pct = loss_pct

        self.timestamp_start = datetime.datetime.now()

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            "epoch",
            "iteration",
            "train/loss",
            "train/acc",
            "train/f1",
            "train/tpr",
            "train/ap",
            "valid/loss",
            "valid/acc",
            "valid/f1",
            "valid/tpr",
            "valid/ap",
            "elapsed_time",
        ]
        if not osp.exists(osp.join(self.out, "log.csv")):
            with open(osp.join(self.out, "log.csv"), "w") as f:
                f.write(",".join(self.log_headers) + "\n")

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = epochs
        self.best_ap = 0.0

    def validate(self):
        self.model.eval()

        val_loss = 0
        val_preds, val_trues = np.array([]), np.array([])
        for _ in tqdm.trange(
            self.val_iterations,
            desc="Valid iteration={:d}".format(self.iteration),
            leave=False,
        ):
            data = next(self.data_loader)
            data = [Variable(d.float().to(self.device)) for d in data]
            with torch.no_grad():
                out = self.model(*data[:-1]).squeeze(dim=-1)

            coll = data[-1].type(out.type())
            loss = self.criterion(out, coll).mean()

            out, coll = self.to_binary(out, coll)
            val_preds = np.append(val_preds, out.cpu().numpy().flatten())
            val_trues = np.append(
                val_trues, coll.float().cpu().numpy().flatten()
            )
            val_loss += loss.item()

        val_loss /= self.val_iterations
        bcr = BinaryClassificationResult(val_preds, val_trues)

        with open(osp.join(self.out, "log.csv"), "a") as f:
            elapsed_time = (
                datetime.datetime.now() - self.timestamp_start
            ).total_seconds()
            log = (
                [self.epoch, self.iteration]
                + [""] * 5
                + [val_loss]
                + [bcr.accuracy, bcr.f1_score, bcr.tpr, bcr.ap_score]
                + [elapsed_time]
            )
            log = map(str, log)
            f.write(",".join(log) + "\n")

        is_best = bcr.ap_score > self.best_ap
        save_dict = {
            "epoch": self.epoch + 1,
            "iteration": self.iteration,
            "arch": self.model.__class__.__name__,
            "optim_state_dict": self.optim.state_dict(),
            "model_state_dict": self.model.state_dict(),
            "best_ap": self.best_ap,
        }
        save_dict.update({"amp": self.scaler.state_dict()})
        torch.save(save_dict, osp.join(self.out, "checkpoint.pth.tar"))

        if is_best:
            shutil.copy(
                osp.join(self.out, "checkpoint.pth.tar"),
                osp.join(self.out, "model.pth.tar"),
            )

    def train_epoch(self):
        self.model.train()

        train_bar = tqdm.trange(
            self.train_iterations,
            desc="Train epoch={:d}".format(self.epoch),
            leave=False,
        )

        for batch_idx in train_bar:

            iteration = batch_idx + self.epoch * self.train_iterations
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            assert self.model.training

            try:
                data = next(self.data_loader)
            except StopIteration:
                print("BAD NEWS BEARS")
                sys.exit(1)
            data = [Variable(d.float().to(self.device)) for d in data]
            self.optim.zero_grad()

            with autocast():
                out = self.model(*data[:-1]).squeeze(dim=-1)

            coll = data[-1].type(out.type())
            losses = self.criterion(out, coll)
            if self.loss_pct < 1.0:
                top_losses, _ = torch.topk(
                    losses, int(losses.size(1) * self.loss_pct), sorted=False
                )
                rand_losses = losses[
                    :,
                    torch.randint(
                        losses.size(1), (int(losses.size(1) * self.loss_pct),)
                    ),
                ]
                loss = 0.5 * (top_losses.mean() + rand_losses.mean())
            else:
                loss = losses.mean()
            loss_data = loss.item()
            if torch.isnan(loss.data):
                raise ValueError("loss is nan while training")
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()

            with torch.no_grad():
                out, coll = self.to_binary(out, coll)
                bcr = BinaryClassificationResult(
                    out.data.cpu().numpy().flatten(),
                    coll.data.cpu().numpy().flatten(),
                )
            train_bar.set_postfix_str("Loss: {:.5f}".format(loss_data))

            with open(osp.join(self.out, "log.csv"), "a") as f:
                elapsed_time = (
                    datetime.datetime.now() - self.timestamp_start
                ).total_seconds()
                log = (
                    [self.epoch, self.iteration]
                    + [loss_data]
                    + [bcr.accuracy, bcr.f1_score, bcr.tpr, bcr.ap_score]
                    + [""] * 5
                    + [elapsed_time]
                )
                log = map(str, log)
                f.write(",".join(log) + "\n")

    @abstractmethod
    def to_binary(self, pred, true):
        pass

    def train(self):
        self.model.to(self.device)
        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc="Train"):
            self.epoch = epoch
            self.train_epoch()
            self.validate()
