import argparse
import os
import os.path as osp
import sys
from timeit import default_timer as timer

import matplotlib as mpl
import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader

mpl.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
from autolab_core import BinaryClassificationResult, YamlConfig, keyboard_input

from scenecollisionnet.collision_models import RobotCollisionNet
from scenecollisionnet.datasets import IterableRobotCollisionDataset

sns.set()


def benchmark(device, model, criterion, test_loader, iterations, out_path):
    model.eval()
    time_start = timer()

    test_loss = 0
    preds, trues = np.array([]), np.array([])
    data_loader = iter(test_loader)
    passes = []
    for _ in tqdm.trange(
        iterations,
        desc="Benchmarking",
    ):
        data = next(data_loader)
        data = [Variable(d.float().to(device)) for d in data]
        centers, colls = data
        fw_pass_start = timer()
        with torch.no_grad():
            out = model(centers).squeeze(dim=-1)
        passes.append(timer() - fw_pass_start)

        loss = criterion(out, colls.float())
        pred_batch = out.cpu().numpy().flatten()
        true_batch = colls.cpu().numpy().flatten()
        preds = np.append(preds, pred_batch)
        trues = np.append(trues, true_batch)
        test_loss += loss.item()

    test_loss /= iterations

    # sort by probs
    accs = []
    f1s = []
    tprs = []
    taus = np.linspace(0, 10, 51)
    for t in taus:
        b = BinaryClassificationResult(preds >= t, trues >= t)
        accs.append(b.accuracy)
        f1s.append(b.f1_score)
        tprs.append(b.tpr)

    np.savez_compressed(
        osp.join(out_path, "accuracy_curve.npz"),
        taus=taus,
        accs=accs,
        f1s=f1s,
        tprs=tprs,
    )

    ax = sns.lineplot(x=taus, y=accs, label="Accuracy")
    ax = sns.lineplot(x=taus, y=f1s, label="F1")
    ax = sns.lineplot(x=taus, y=tprs, label="TPR")
    ax.legend()
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.ylim([0.49, 1.01])
    plt.xlim([taus.min() - 0.01, taus.max() + 0.01])
    plt.savefig(osp.join(out_path, "accuracy_curve.png"))

    bcr = BinaryClassificationResult(
        preds >= taus.mean(), trues >= taus.mean()
    )
    with open(osp.join(out_path, "results.txt"), "w") as f:
        elapsed_time = timer() - time_start
        log = (
            ["Images: {:d}".format(iterations)]
            + ["Queries: {:d}".format(iterations * len(colls.flatten()))]
            + ["Loss: {:.5f}".format(test_loss)]
            + ["Accuracy: {:.3f}".format(bcr.accuracy)]
            + ["F1 Score: {:.3f}".format(bcr.f1_score)]
            + ["TPR: {:.3f}".format(bcr.tpr)]
            + ["AP Score: {:.3f}".format(bcr.ap_score)]
            + [
                "FW Pass Time: {:.4f} +- {:.4f} s".format(
                    np.mean(passes), np.std(passes)
                )
            ]
            + ["Time: {:.2f} s".format(elapsed_time)]
        )
        log = map(str, log)
        f.write("\n".join(log))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Benchmark a SelfCollisionNet"
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="cfg/benchmark_robotcollisionnet.yaml",
        help="config file with benchmarking params",
    )
    args = parser.parse_args()

    config = YamlConfig(args.cfg)

    # Create output directory for model
    out = osp.join(config["output"], config["model"]["name"])
    if osp.exists(out):
        response = keyboard_input(
            "A benchmark folder exists at {}. Overwrite?".format(out),
            yesno=True,
        )
        if response.lower() == "n":
            os.exit()
    else:
        os.makedirs(out)
    config.save(osp.join(out, "benchmark.yaml"))

    # Check whether GPU available
    if not torch.cuda.is_available():
        print("No CUDA available!")
        sys.exit(0)
    device = torch.device(config["device"])

    # 1. Model cfg
    model_path = osp.join(config["model"]["path"], config["model"]["name"])
    train_cfg = YamlConfig(osp.join(model_path, "train.yaml"))

    # 2. Dataset
    kwargs = {
        "num_workers": config["num_workers"]
        if "num_workers" in config
        else os.cpu_count(),
        "pin_memory": True,
        "worker_init_fn": lambda _: np.random.seed(),
    }
    train_cfg["dataset"]["batch_size"] = config["dataset"]["batch_size"]
    test_set = IterableRobotCollisionDataset(
        **train_cfg["dataset"],
    )
    test_loader = DataLoader(test_set, batch_size=None, **kwargs)

    # 3. Model
    model = RobotCollisionNet(num_joints=len(test_set.robot.actuated_joints))
    checkpoint = torch.load(osp.join(model_path, "model.pth.tar"))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)

    benchmark(
        device=device,
        model=model,
        criterion=nn.MSELoss(),
        test_loader=test_loader,
        iterations=config["iterations"],
        out_path=out,
    )
